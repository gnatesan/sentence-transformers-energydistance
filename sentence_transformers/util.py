import functools
import requests
from torch import Tensor, device
from typing import List, Callable, Literal
from tqdm.autonotebook import tqdm
import sys
import importlib
import os
import torch
import numpy as np
import queue
import logging
from typing import Dict, Optional, Union, overload

from transformers import is_torch_npu_available
from huggingface_hub import snapshot_download, hf_hub_download
import heapq

logger = logging.getLogger(__name__)

#Function to calculate the energy of all vectors in a query
def ed_calc_old(x):
    M = x.shape[0]
    x_expanded = x.unsqueeze(1).expand(-1, M, -1)
    ed_sum = torch.norm(x_expanded - x, dim=2).sum()
    return ed_sum / (M * M)

#Function to calcjulate the energy between a single vecctor document and multi-vector query
def energy_calc_old(x, y):
    M = x.shape[0]
    N = y.shape[0]

    # Expand tensors to create all pairs of vectors
    x_expanded = x.unsqueeze(1).expand(-1, N, -1)
    y_expanded = y.unsqueeze(0).expand(M, -1, -1)
    
    # Compute pairwise squared Euclidean distances
    pairwise_diff = x_expanded - y_expanded
    squared_distances = torch.sum(pairwise_diff ** 2, dim=2)

    # Sum up squared distances and scale by (M * N)
    ed_sum = torch.sum(torch.sqrt(squared_distances))

    return 2 * ed_sum / (M * N)

#Function to calculate the energy distance between a batch of queries and a batch of documents.
#Queries are represented as a list of 2d tensors, and documents are represented by a list of 
#1d tensors. Implementation is based off of https://pages.stat.wisc.edu/~wahba/stat860public/pdf4/Energy/EnergyDistance10.1002-wics.1375.pdf
def energy_distance_old(x, y):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move queries and documents to GPU
    x = [query.to(device) for query in x]  # List of 2D query tensors
    y = [doc.to(device) for doc in y]  # List of 1D document tensors

    num_queries = len(x) #number of queries
    num_documents = len(y) #number of documents

    logger.error(f"Num queries in batch: {num_queries}")
    logger.error(f"Num documents in batch: {num_documents}")

    #print("Num queries:", num_queries)
    #print("Num documents:", num_documents)

    #if not isinstance(x, torch.Tensor):
    #    x = torch.tensor(x)

    #if not isinstance(y, torch.Tensor):
    #    y = torch.tensor(y)
    
    #total_queries, sequence_length, query_dim = x.shape
    #total_docs, doc_dim = y.shape

    # Pre-calculate energy for all queries
    ed_queries = torch.stack([ed_calc(query) for query in x]).to(device)
    #logger.error(f"ed_queries device: {ed_queries.device}")

    # Create a tensor of shape M*N filled with zeros
    tensor = torch.zeros(num_queries, num_documents, device=device)

    for i in range(num_queries):
      ed_query = ed_queries[i] #store energy calculation of query to improve runtime
      #logger.error(f"Device containing query: {x[i].device}, Query: {i}")
      for j in range(num_documents):
        tensor[i][j] = (energy_calc(x[i], y[j].reshape(1,-1)).item() - ed_query.item()) * -1
    logger.error(f"Device containing ED tensor: {tensor.device}")
    return tensor

#Takes in a 3d tensor of queries where each query is a 2d tensor that has been padded to the 
#max sequence length in the batch. Also takes in 2d tensor of documents. Multiplied by -1 so larger scores mean higher similarity. 
def energy_distance2(x, y):
    # Shape of x: [num_queries, max_sequence_length, query_dim]
    # Shape of y: [num_docs, doc_dim]
    #print("ED calculation tensors")
    #print(x.device)  # Check device
    #print(y.device)  # Check device

    num_queries, max_sequence_length, query_dim = x.shape
    num_docs, doc_dim = y.shape

    # Check for dimensionality compatibility
    assert query_dim == doc_dim, "Query and document dimensions must match!"

    # Pre-calculate energy for all queries (batch of 2D query tensors)
    ed_queries = torch.stack([ed_calc(query) for query in x])

    # Expand query tensor for broadcasting:
    # x_expanded: [num_queries, num_docs, max_seq_length, query_dim]
    x_expanded = x.unsqueeze(1).expand(-1, num_docs, -1, -1)

    # Expand document tensor for broadcasting:
    # y_expanded: [num_queries, num_docs, query_dim] -> unsqueeze for broadcasting
    y_expanded = y.unsqueeze(0).expand(num_queries, -1, -1)

    # Now, x_expanded has shape [num_queries, num_docs, max_seq_length, query_dim]
    # y_expanded has shape [num_queries, num_docs, query_dim]

    # Calculate energy distances for all query-document pairs in parallel
    pairwise_diff = x_expanded - y_expanded.unsqueeze(2)  # Shape: [num_queries, num_docs, max_seq_length, query_dim]
    squared_distances = torch.sum(pairwise_diff ** 2, dim=3)  # Shape: [num_queries, num_docs, max_seq_length]

    # Compute the sum of sqrt of squared distances for each query-document pair
    ed_sums = torch.sum(torch.sqrt(squared_distances), dim=2)  # Shape: [num_queries, num_docs]

    # Final energy distance calculation (using pre-calculated query energies)
    energy_distances = (2 * ed_sums / (max_sequence_length) - ed_queries.unsqueeze(1)) * -1

    return energy_distances

def ed_calc(x, attention_mask):
    """
    Calculate the energy for all queries in parallel, accounting for padding.

    Args:
	x (torch.Tensor): Query embeddings of shape [num_queries, max_sequence_length, query_dim].
        attention_mask (torch.Tensor): Mask of shape [num_queries, max_sequence_length],
                                        where 1 indicates valid tokens and 0 indicates padding.

    Returns:
	torch.Tensor: Energy values for each query, shape [num_queries].
    """
    # Shape: [num_queries, max_sequence_length, query_dim]
    num_queries, max_sequence_length, query_dim = x.shape

    # Create pairwise differences: [num_queries, max_sequence_length, max_sequence_length, query_dim]
    x_expanded_1 = x.unsqueeze(2)  # Expand along the second dimension
    x_expanded_2 = x.unsqueeze(1)  # Expand along the third dimension
    pairwise_diff = x_expanded_1 - x_expanded_2

    # Compute pairwise distances: [num_queries, max_sequence_length, max_sequence_length]
    pairwise_distances = torch.norm(pairwise_diff, dim=3)

    # Apply the attention mask to exclude padded tokens
    attention_mask_expanded = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)  # [num_queries, max_sequence_length, max_sequence_length]
    pairwise_distances = pairwise_distances * attention_mask_expanded  # Mask padded positions

    # Count valid token pairs for normalization
    valid_pairs = attention_mask_expanded.sum(dim=(1, 2)).clamp(min=1)  # Shape: [num_queries]

    # Sum of distances and normalize
    ed_sums = pairwise_distances.sum(dim=(1, 2))  # Sum across all pairs
    energy = ed_sums / valid_pairs  # Normalize by the number of valid pairs
    #print("ed_calc_new result:", energy)
    return energy  # Shape: [num_queries]

#ED implementation which uses broadcasting operations and no loops at all
def energy_distance_new(x, y, attention_mask):
    # Shape of x: [num_queries, max_sequence_length, query_dim]
    # Shape of y: [num_docs, doc_dim]
    #print("ED calculation tensors")
    #print(x.device)  # Check device
    #print(y.device)  # Check device

    num_queries, max_sequence_length, query_dim = x.shape
    num_docs, doc_dim = y.shape

    # Check for dimensionality compatibility
    assert query_dim == doc_dim, "Query and document dimensions must match!"

    # Pre-calculate energy for all queries (batch of 2D query tensors)
    ed_queries = ed_calc(x, attention_mask)

    # Expand query tensor for broadcasting:
    # x_expanded: [num_queries, num_docs, max_seq_length, query_dim]
    x_expanded = x.unsqueeze(1).expand(-1, num_docs, -1, -1)

    # Expand document tensor for broadcasting:
    # y_expanded: [num_queries, num_docs, query_dim] -> unsqueeze for broadcasting
    y_expanded = y.unsqueeze(0).expand(num_queries, -1, -1)

    # Now, x_expanded has shape [num_queries, num_docs, max_seq_length, query_dim]
    # y_expanded has shape [num_queries, num_docs, query_dim]

    # Calculate energy distances for all query-document pairs in parallel
    pairwise_diff = x_expanded - y_expanded.unsqueeze(2)  # Shape: [num_queries, num_docs, max_seq_length, query_dim]
    pairwise_distances = torch.norm(pairwise_diff, dim=3)

    #squared_distances = torch.sum(pairwise_diff ** 2, dim=3)  # Shape: [num_queries, num_docs, max_seq_length]

    # Apply attention mask to zero out padded embeddings
    # Expand attention_mask for broadcasting: [num_queries, max_sequence_length] -> [num_queries, 1, max_sequence_length]
    attention_mask_expanded = attention_mask.unsqueeze(1).expand(-1, num_docs, -1)
    pairwise_distances = pairwise_distances * attention_mask_expanded

    # Compute the sum of sqrt of squared distances for each query-document pair
    #ed_sums = torch.sum(torch.sqrt(squared_distances), dim=2)  # Shape: [num_queries, num_docs]

    # Sum distances and normalize by valid token count for each query-document pair
    # valid_token_counts: [num_queries, 1, max_sequence_length] -> [num_queries, num_docs]
    valid_token_counts = attention_mask_expanded.sum(dim=2).clamp(min=1)  # Avoid division by zero
    ed_sums = torch.sum(pairwise_distances, dim=2) / valid_token_counts

    # Final energy distance calculation (using pre-calculated query energies)
    energy_distances = 2 * ed_sums - ed_queries.unsqueeze(1)

    return energy_distances

#ED implementation which uses einsum to avoid GPU memory issues
def energy_distance_einsum(x, y, attention_mask):
    """
    Efficient energy distance calculation using einsum.
    """
    #print("Einsum ED implementation!")
    # Norms of x (queries) and y (documents)
    x_norm_sq = torch.einsum('qmk,qmk->qm', x, x)  # [num_queries, max_seq_length]
    y_norm_sq = torch.einsum('nk,nk->n', y, y)  # [num_docs]

    # Cross-term
    cross_term = torch.einsum('qmk,nk->qmn', x, y)  # [num_queries, num_docs, max_seq_length]

    # Reshape norms for broadcasting
    x_norm_sq = x_norm_sq.unsqueeze(2)  # [num_queries, max_seq_length, 1]
    y_norm_sq = y_norm_sq.unsqueeze(0).unsqueeze(0)  # [1, 1, num_docs]

    # Pairwise squared distances
    pairwise_distances = x_norm_sq + y_norm_sq - 2 * cross_term
    pairwise_distances = torch.sqrt(pairwise_distances.clamp(min=0))  # Avoid negative sqrt due to numerical issues

    # Apply attention mask to exclude padded tokens
    attention_mask_expanded = attention_mask.unsqueeze(2).expand(-1, -1, y.shape[0])  # [num_queries, max_seq_length, num_docs]
    pairwise_distances = pairwise_distances * attention_mask_expanded

    # Normalize by valid token counts
    valid_token_counts = attention_mask.sum(dim=1).unsqueeze(1).expand(-1, y.shape[0])  # [num_queries, num_docs]
    valid_token_counts = valid_token_counts.clamp(min=1)  # Avoid division by zero
    ed_sums = pairwise_distances.sum(dim=1) / valid_token_counts

    # Precompute query energies
    ed_queries = ed_calc(x, attention_mask)

    # Final energy distances
    energy_distances = 2 * ed_sums - ed_queries.unsqueeze(1)

    return energy_distances

def energy_distance(x, y, attention_mask):
    """
    Compute energy distance between multivector queries and single vector documents using torch.einsum.

    Args:
        x (torch.Tensor): Query embeddings of shape [num_queries, max_sequence_length, query_dim].
        y (torch.Tensor): Document embeddings of shape [num_docs, doc_dim].
        attention_mask (torch.Tensor): Attention mask of shape [num_queries, max_sequence_length].

    Returns:
        torch.Tensor: Energy distances of shape [num_queries, num_docs].
    """
    # Shapes
    num_queries, max_sequence_length, query_dim = x.shape
    num_docs, doc_dim = y.shape

    # Ensure dimensions are compatible
    assert query_dim == doc_dim, "Query and document dimensions must match!"

    # Step 1: Compute squared norms of the document embeddings (efficient norm calculation)
    # y: [num_docs, query_dim], norm_y: [num_docs]
    norm_y = torch.einsum("nd,nd->n", y, y)  # Shape: [num_docs]

    # Step 2: Compute pairwise squared distances between query tokens and document embeddings
    # x: [num_queries, max_sequence_length, query_dim], y: [num_docs, query_dim]
    # Output shape: [num_queries, max_sequence_length, num_docs]
    dot_product = torch.einsum("qld,nd->qln", x, y)  # Shape: [num_queries, max_sequence_length, num_docs]

    # Squared distance calculation
    # norm_x_tokens: [num_queries, max_sequence_length]
    norm_x_tokens = torch.einsum("qld,qld->ql", x, x)  # Shape: [num_queries, max_sequence_length]
    
    # Applying the squared distance formula: ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * <x_i, y_j>
    squared_distances = (
        norm_x_tokens.unsqueeze(2) + norm_y.unsqueeze(0).unsqueeze(1) - 2 * dot_product
    )  # Shape: [num_queries, max_sequence_length, num_docs]

    # Ensure distances are non-negative due to numerical instability
    squared_distances = squared_distances.clamp(min=0)

    # Step 3: Compute Euclidean distances (L2 norm)
    distances = torch.sqrt(squared_distances)  # Shape: [num_queries, max_sequence_length, num_docs]

    # Step 4: Apply attention mask to the distances
    attention_mask_expanded = attention_mask.unsqueeze(2)  # Shape: [num_queries, max_sequence_length, 1]
    masked_distances = distances * attention_mask_expanded  # Shape: [num_queries, max_sequence_length, num_docs]

    # Step 5: Aggregate distances across sequence length and normalize by valid token count
    valid_token_counts = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(1)  # Shape: [num_queries, 1]
    ed_sums = masked_distances.sum(dim=1) / valid_token_counts  # Shape: [num_queries, num_docs]

    # Step 6: Compute energy for queries and combine with pairwise distances
    ed_queries = ed_calc(x, attention_mask)  # Precomputed energy for each query, shape: [num_queries]
    energy_distances = 2 * ed_sums - ed_queries.unsqueeze(1)  # Shape: [num_queries, num_docs]

    return energy_distances


def pytorch_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)


def cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def pairwise_dot_score(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise dot-product dot_prod(a[i], b[i])

    :return: Vector with res[i] = dot_prod(a[i], b[i])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return (a * b).sum(dim=-1)


def pairwise_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise cossim cos_sim(a[i], b[i])

    :return: Vector with res[i] = cos_sim(a[i], b[i])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))


def pairwise_angle_sim(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the absolute normalized angle distance;
    see AnglELoss or https://arxiv.org/abs/2309.12871v1
    for more information.

    :return: Vector with res[i] = angle_sim(a[i], b[i])
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    # modified from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
    # chunk both tensors to obtain complex components
    a, b = torch.chunk(x, 2, dim=1)
    c, d = torch.chunk(y, 2, dim=1)

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
    re /= dz / dw
    im /= dz / dw

    norm_angle = torch.sum(torch.concat((re, im), dim=1), dim=1)
    return torch.abs(norm_angle)


def normalize_embeddings(embeddings: Tensor) -> Tensor:
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


@overload
def truncate_embeddings(embeddings: np.ndarray, truncate_dim: Optional[int]) -> np.ndarray: ...


@overload
def truncate_embeddings(embeddings: torch.Tensor, truncate_dim: Optional[int]) -> torch.Tensor: ...


def truncate_embeddings(
    embeddings: Union[np.ndarray, torch.Tensor], truncate_dim: Optional[int]
) -> Union[np.ndarray, torch.Tensor]:
    """
    :param embeddings: Embeddings to truncate.
    :param truncate_dim: The dimension to truncate sentence embeddings to. `None` does no truncation.
    :return: Truncated embeddings.
    """
    return embeddings[..., :truncate_dim]


def paraphrase_mining(
    model, sentences: List[str], show_progress_bar: bool = False, batch_size: int = 32, *args, **kwargs
) -> List[List[Union[float, int]]]:
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param model: SentenceTransformer model for embedding computation
    :param sentences: A list of strings (texts or sentences)
    :param show_progress_bar: Plotting of a progress bar
    :param batch_size: Number of texts that are encoded simultaneously by the model
    :param query_chunk_size: Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
    :param corpus_chunk_size: Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of text pairs returned.
    :param top_k: For each sentence, we retrieve up to top_k other sentences
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list of triplets with the format [score, id1, id2]
    """

    # Compute embedding for the sentences
    embeddings = model.encode(
        sentences, show_progress_bar=show_progress_bar, batch_size=batch_size, convert_to_tensor=True
    )

    return paraphrase_mining_embeddings(embeddings, *args, **kwargs)


def paraphrase_mining_embeddings(
    embeddings: Tensor,
    query_chunk_size: int = 5000,
    corpus_chunk_size: int = 100000,
    max_pairs: int = 500000,
    top_k: int = 100,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> List[List[Union[float, int]]]:
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param embeddings: A tensor with the embeddings
    :param query_chunk_size: Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
    :param corpus_chunk_size: Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of text pairs returned.
    :param top_k: For each sentence, we retrieve up to top_k other sentences
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list of triplets with the format [score, id1, id2]
    """

    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            scores = score_function(
                embeddings[query_start_idx : query_start_idx + query_chunk_size],
                embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size],
            )

            scores_top_k_values, scores_top_k_idx = torch.topk(
                scores, min(top_k, len(scores[0])), dim=1, largest=True, sorted=False
            )
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(scores)):
                for top_k_idx, corpus_itr in enumerate(scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr

                    if i != j and scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, sorted_i, sorted_j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list


def information_retrieval(*args, **kwargs) -> List[List[Dict[str, Union[int, float]]]]:
    """This function is deprecated. Use semantic_search instead"""
    return semantic_search(*args, **kwargs)


def semantic_search(
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> List[List[Dict[str, Union[int, float]]]]:
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(
                query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                corpus_embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size],
            )

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(queries_result_list[query_id], (score, corpus_id))

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {"corpus_id": corpus_id, "score": score}
        queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x["score"], reverse=True)

    return queries_result_list


def http_get(url, path) -> None:
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def fullname(o) -> str:
    """
    Gives a full name (package_name.class_name) for a class / object in Python. Will
    be used to load the correct classes from JSON files
    """

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + o.__class__.__name__


def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except Exception:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)


def community_detection(
    embeddings, threshold=0.75, min_community_size=10, batch_size=1024, show_progress_bar=False
) -> List[List[int]]:
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    threshold = torch.tensor(threshold, device=embeddings.device)
    embeddings = normalize_embeddings(embeddings)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in tqdm(
        range(0, len(embeddings), batch_size), desc="Finding clusters", disable=not show_progress_bar
    ):
        # Compute cosine similarity scores
        cos_scores = embeddings[start_idx : start_idx + batch_size] @ embeddings.T

        # Use a torch-heavy approach if the embeddings are on CUDA, otherwise a loop-heavy one
        if embeddings.device.type in ["cuda", "npu"]:
            # Threshold the cos scores and determine how many close embeddings exist per embedding
            threshold_mask = cos_scores >= threshold
            row_wise_count = threshold_mask.sum(1)

            # Only consider embeddings with enough close other embeddings
            large_enough_mask = row_wise_count >= min_community_size
            if not large_enough_mask.any():
                continue

            row_wise_count = row_wise_count[large_enough_mask]
            cos_scores = cos_scores[large_enough_mask]

            # The max is the largest potential community, so we use that in topk
            k = row_wise_count.max()
            _, top_k_indices = cos_scores.topk(k=k, largest=True)

            # Use the row-wise count to slice the indices
            for count, indices in zip(row_wise_count, top_k_indices):
                extracted_communities.append(indices[:count].tolist())
        else:
            # Minimum size for a community
            top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

            # Filter for rows >= min_threshold
            for i in range(len(top_k_values)):
                if top_k_values[i][-1] >= threshold:
                    # Only check top k most similar entries
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    # Check if we need to increase sort_max_size
                    while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                        sort_max_size = min(2 * sort_max_size, len(embeddings))
                        top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    extracted_communities.append(top_idx_large[top_val_large >= threshold].tolist())

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities


##################
#
######################


class disabled_tqdm(tqdm):
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        """Fix for https://github.com/huggingface/huggingface_hub/issues/1603"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise


def is_sentence_transformer_model(
    model_name_or_path: str,
    token: Optional[Union[bool, str]] = None,
    cache_folder: Optional[str] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> bool:
    return bool(
        load_file_path(
            model_name_or_path,
            "modules.json",
            token,
            cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def load_file_path(
    model_name_or_path: str,
    filename: str,
    token: Optional[Union[bool, str]],
    cache_folder: Optional[str],
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> Optional[str]:
    # If file is local
    file_path = os.path.join(model_name_or_path, filename)
    if os.path.exists(file_path):
        return file_path

    # If file is remote
    try:
        return hf_hub_download(
            model_name_or_path,
            filename=filename,
            revision=revision,
            library_name="sentence-transformers",
            token=token,
            cache_dir=cache_folder,
            local_files_only=local_files_only,
        )
    except Exception:
        return


def load_dir_path(
    model_name_or_path: str,
    directory: str,
    token: Optional[Union[bool, str]],
    cache_folder: Optional[str],
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> Optional[str]:
    # If file is local
    dir_path = os.path.join(model_name_or_path, directory)
    if os.path.exists(dir_path):
        return dir_path

    download_kwargs = {
        "repo_id": model_name_or_path,
        "revision": revision,
        "allow_patterns": f"{directory}/**",
        "library_name": "sentence-transformers",
        "token": token,
        "cache_dir": cache_folder,
        "local_files_only": local_files_only,
        "tqdm_class": disabled_tqdm,
    }
    # Try to download from the remote
    try:
        repo_path = snapshot_download(**download_kwargs)
    except Exception:
        # Otherwise, try local (i.e. cache) only
        download_kwargs["local_files_only"] = True
        repo_path = snapshot_download(**download_kwargs)
    return os.path.join(repo_path, directory)


def save_to_hub_args_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # If repo_id not already set, use repo_name
        repo_name = kwargs.pop("repo_name", None)
        if repo_name and "repo_id" not in kwargs:
            logger.warning(
                "Providing a `repo_name` keyword argument to `save_to_hub` is deprecated, please use `repo_id` instead."
            )
            kwargs["repo_id"] = repo_name

        # If positional args are used, adjust for the new "token" keyword argument
        if len(args) >= 2:
            args = (*args[:2], None, *args[2:])

        return func(self, *args, **kwargs)

    return wrapper


def get_device_name() -> Literal["mps", "cuda", "npu", "hpu", "cpu"]:
    """
    Returns the name of the device where this module is running on.
    It's simple implementation that doesn't cover cases when more powerful GPUs are available and
    not a primary device ('cuda:0') or MPS device is available, but not configured properly:
    https://pytorch.org/docs/master/notes/mps.html

    :return: Device name, like 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_torch_npu_available():
        return "npu"
    elif importlib.util.find_spec("habana_frameworks") is not None:
        import habana_frameworks.torch.hpu as hthpu

        if hthpu.is_available():
            return "hpu"
    return "cpu"
