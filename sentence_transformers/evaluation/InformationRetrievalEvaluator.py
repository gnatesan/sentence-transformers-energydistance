from sentence_transformers import SentenceTransformer
from contextlib import nullcontext
from . import SentenceEvaluator
import torch
from torch import Tensor
import logging
from tqdm import trange
from ..util import cos_sim, dot_score, energy_distance, ed_calc
import os
import numpy as np
from typing import List, Dict, Optional, Set, Callable
import heapq
import time


logger = logging.getLogger(__name__)


class InformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(
        self,
        queries: Dict[str, str],  # qid => query
        corpus: Dict[str, str],  # cid => doc
        relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        map_at_k: List[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: Optional[int] = None,
        score_functions: Dict[str, Callable[[Tensor, Tensor], Tensor]] = {
            "energy_distance": energy_distance,
        },  # Score function, higher=more similar
        main_score_function: str = None,
    ):
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function
        self.truncate_dim = truncate_dim

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))

            for k in precision_recall_at_k:
                self.csv_headers.append("{}-Precision@{}".format(score_name, k))
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

            for k in ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))

            for k in map_at_k:
                self.csv_headers.append("{}-MAP@{}".format(score_name, k))

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Information Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            return max([scores[name]["ndcg@k"][max(self.ndcg_at_k)] for name in self.score_function_names])
        else:
            return scores[self.main_score_function]["ndcg@k"][max(self.ndcg_at_k)]

    def precompute_corpus_embeddings(self, corpus, model, batch_size, chunk_size):
        all_corpus_embeddings = []
        print("Length of corpus:", len(corpus))
        print("Batch size:", batch_size)
        for start_idx in range(0, len(corpus), chunk_size):
            end_idx = min(start_idx + chunk_size, len(corpus))
            print("Chunk to document:", end_idx)
            chunk = corpus[start_idx:end_idx]
            embeddings = model.encode(chunk, batch_size=batch_size, convert_to_tensor=True)
            embeddings = embeddings.to('cpu')
            all_corpus_embeddings.append(embeddings)
        return all_corpus_embeddings


    
    def compute_metrices(
        self, model: SentenceTransformer, corpus_model=None, corpus_embeddings: Tensor = None
    ) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Compute embedding for the queries
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            query_embeddings, attention_masks = model.encode(
                self.queries,
                show_progress_bar=self.show_progress_bar,
                output_value="token_embeddings",
                batch_size=self.batch_size,
                convert_to_tensor=True,
            )

        queries_result_list = {}
        #approx_total_queries = len(query_embeddings) * self.batch_size #number of batches times batch size, last batch could be shorter
        actual_total_queries = sum(len(batch) for batch in query_embeddings)
        for name in self.score_functions:
            #queries_result_list[name] = [[] for _ in range(len(query_embeddings))]
            queries_result_list[name] = [[] for _ in range(actual_total_queries)]

        start_time = time.time()

        # Move query embeddings and attention masks to CPU to make space for corpus embeddings
        #query_embeddings = query_embeddings.to('cpu')
        #attention_masks = attention_masks.to('cpu')
        #query_embeddings = [qe.to('cpu') for qe in query_embeddings]
        #attention_masks = [am.to('cpu') for am in attention_masks]

        # End the timer
        end_time = time.time()

        # Calculate and print the elapsed time
        #elapsed_time = end_time - start_time
        #print(f"Time taken to move query embeddings and attention masks to CPU: {elapsed_time:.4f} seconds")



        # Precompute all corpus embeddings
        all_corpus_embeddings = self.precompute_corpus_embeddings(
            corpus=self.corpus,
            model=corpus_model,  # Use the corpus-specific model
            batch_size=self.batch_size,
            chunk_size=self.corpus_chunk_size
        )

        # Iterate over chunks of the corpus along with each query batch 
        for query_batch_index, query_batch in enumerate(query_embeddings):
            #query_batch = query_batch.to('cuda')
            #attention_mask = attention_masks[query_batch_index].to('cuda')
            for chunk_idx, sub_corpus_embeddings in enumerate(all_corpus_embeddings):
                sub_corpus_embeddings = sub_corpus_embeddings.to('cuda')
                chunk_start_idx = chunk_idx * self.corpus_chunk_size  # Calculate the starting index of this chunk
            #for corpus_start_idx in trange(
            #    0, len(self.corpus), self.corpus_chunk_size, desc="Corpus Chunks", disable=not self.show_progress_bar
            #):
                #corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))
    
                # Encode chunk of corpus
                #if corpus_embeddings is None:
                #    with nullcontext() if self.truncate_dim is None else corpus_model.truncate_sentence_embeddings(
                #        self.truncate_dim
                #    ):
                #        sub_corpus_embeddings = corpus_model.encode(
                #            self.corpus[corpus_start_idx:corpus_end_idx],
                #            show_progress_bar=False,
                #            batch_size=self.batch_size,
                #            convert_to_tensor=True,
                #        )
                #else:
                #    sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]
    
                # Compute cosine similarites
                for name, score_function in self.score_functions.items(): #ED should be the only score function in score_functions
                    logger.error(f"Validation! Calling score function: {name}")
                    #pair_scores = score_function(query_embeddings, sub_corpus_embeddings, attention_mask)
                    pair_scores = score_function(query_batch, sub_corpus_embeddings, attention_masks[query_batch_index])
    
                    # Get top-k values
                    pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                        pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                    )
                    pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                    pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()
    
                    for query_itr in range(len(query_batch)):
                        #query_batch_size = len(query_batch)
                        global_query_index = query_itr + (query_batch_index * self.batch_size)
                        for sub_corpus_id, score in zip(
                            pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]
                        ):
                            #corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                            corpus_id = self.corpus_ids[chunk_start_idx + sub_corpus_id]  # Use chunk_start_idx here
                            if len(queries_result_list[name][global_query_index]) < max_k:
                                heapq.heappush(
                                    queries_result_list[name][global_query_index], (score, corpus_id)
                                )  # heaqp tracks the quantity of the first element in the tuple
                            else:
                                heapq.heappushpop(queries_result_list[name][global_query_index], (score, corpus_id))
                sub_corpus_embeddings = sub_corpus_embeddings.to('cpu') 

            # After processing the batch, delete the query_batch tensor to free GPU memory
            del query_batch
            torch.cuda.empty_cache()  # Optionally free up any cached GPU memory



        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {"corpus_id": corpus_id, "score": score}

        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        # Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])

        return scores

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    def output_scores(self, scores):
        for k in scores["accuracy@k"]:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores["accuracy@k"][k] * 100))

        for k in scores["precision@k"]:
            logger.info("Precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100))

        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

        for k in scores["map@k"]:
            logger.info("MAP@{}: {:.4f}".format(k, scores["map@k"][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg
