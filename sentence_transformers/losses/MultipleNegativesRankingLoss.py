import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from .. import util

def ed_calc2(x):
    M = x.shape[0]
    x_expanded = x.unsqueeze(1).expand(-1, M, -1)
    ed_sum = torch.norm(x_expanded - x, dim=2).sum()
    return ed_sum / (M * M)

def energy_calc2(x, y):
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

def ed_calc3(x, attention_mask):
    #x = x.to(device)
    #attention_mask = attention_mask.to(device)
    M = x.shape[0]

    #print("Input query tensor (x):")
    #print(x)
    #print("Shape of input query tensor:", x.shape)

    #print("Attention mask for single query:")
    #print(attention_mask)
    #print("Shape of attention mask:", attention_mask.shape)

    attention_mask_expanded = attention_mask.unsqueeze(1).expand(-1, M) * attention_mask.unsqueeze(0).expand(M, -1)

    #print("Attention mask expanded shape:", attention_mask_expanded.shape)
    #print("Expanded attention mask:")
    #print(attention_mask_expanded)

    x_expanded = x.unsqueeze(1).expand(-1, M, -1)
    #print("Expanded input tensor shape:", x_expanded.shape)
    #ed_sum = torch.norm(x_expanded - x, dim=2).sum()
    pairwise_diffs = x_expanded - x
    #print("Pairwise differences shape:", pairwise_diffs.shape)
    masked_diffs = pairwise_diffs * attention_mask_expanded.unsqueeze(-1)
    #print("Masked differences shape:", masked_diffs.shape)
    ed_sum = torch.norm(masked_diffs, dim=2).sum()
    #print("Energy distance sum (ed_sum) padded query embeddings:", ed_sum.item())
    valid_token_count = attention_mask_expanded.sum()
    #print("Valid token count:", valid_token_count)
    #return ed_sum / (M * M)
    #ans = ed_sum / (valid_token_count) if valid_token_count > 0 else 0
    #print("Energy distance of query (padded query embeddings):", ans)
    return ed_sum / (valid_token_count)

def energy_calc3(x, y, attention_mask):
    M = x.shape[0]
    N = y.shape[0]

    #print(x.device)  # Check device
    #print(y.device)  # Check device

    # Expand tensors to create all pairs of vectors
    x_expanded = x.unsqueeze(1).expand(-1, N, -1)
    y_expanded = y.unsqueeze(0).expand(M, -1, -1)

    # Compute pairwise squared Euclidean distances
    pairwise_diff = x_expanded - y_expanded
    attention_mask_expanded = attention_mask.unsqueeze(1).expand(-1, N)
    masked_diffs = pairwise_diff * attention_mask_expanded.unsqueeze(-1)
    #squared_distances = torch.sum(pairwise_diff ** 2, dim=2)
    squared_distances = torch.sum(masked_diffs ** 2, dim=2)

    # Sum up squared distances and scale by (M * N)
    ed_sum = torch.sum(torch.sqrt(squared_distances))
    valid_token_count = attention_mask_expanded.sum()

    #return 2 * ed_sum / (M * N)

    #ans = 2 * ed_sum / (valid_token_count * N) if valid_token_count > 0 else 0
    #print("Energy distance of query-document pair (padded query embeddings):", ans)

    return 2 * ed_sum / valid_token_count

def ed_calc4(x, attention_mask):
    M = x.shape[0]

    attention_mask_expanded = attention_mask.unsqueeze(1).expand(-1, M) * attention_mask.unsqueeze(0).expand(M, -1)
    x_expanded = x.unsqueeze(1).expand(-1, M, -1)
    pairwise_diffs = x_expanded - x

    # Compute squared distances without applying the mask
    squared_distances = torch.sum(pairwise_diffs ** 2, dim=2)

    # Add epsilon only to zero values in squared_distances
    epsilon = 1e-8
    squared_distances += (squared_distances == 0) * epsilon
    #squared_distances = torch.where(squared_distances == 0, squared_distances + epsilon, squared_distances)

    # Take the square root
    sqrt_distances = torch.sqrt(squared_distances)
    if torch.isnan(sqrt_distances).any():
        print("NaN detected in sqrt_distances")


    # Now apply the mask: attention_mask_expanded needs to have the shape [M, M]
    masked_sqrt_distances = sqrt_distances * attention_mask_expanded

    # Sum masked distances and normalize by valid tokens
    ed_sum = torch.sum(masked_sqrt_distances)
    valid_token_count = attention_mask_expanded.sum()

    #print("attention_mask", attention_mask.device)
    #print("attention_mask_expanded", attention_mask_expanded.device)
    #print("valid_token_count", valid_token_count.device)
    #print("sqrt_distances", sqrt_distances.device)
    #print("masked_sqrt_distances", masked_sqrt_distances.device)
    #print("ed_sum", ed_sum.device)

    return ed_sum / valid_token_count

def energy_calc4(x, y, attention_mask):
    M = x.shape[0]
    N = y.shape[0]

    # Expand tensors to create all pairs of vectors
    x_expanded = x.unsqueeze(1).expand(-1, N, -1)
    y_expanded = y.unsqueeze(0).expand(M, -1, -1)
    pairwise_diff = x_expanded - y_expanded

    # Compute squared distances without applying the mask
    squared_distances = torch.sum(pairwise_diff ** 2, dim=2)

    # Take the square root
    sqrt_distances = torch.sqrt(squared_distances)

    # Now apply the mask: attention_mask_expanded needs to have the shape [M, N]
    attention_mask_expanded = attention_mask.unsqueeze(1).expand(-1, N)
    masked_sqrt_distances = sqrt_distances * attention_mask_expanded

    # Sum masked distances and normalize by valid tokens
    ed_sum = torch.sum(masked_sqrt_distances)
    valid_token_count = attention_mask_expanded.sum()

    return 2 * ed_sum / valid_token_count

def ed_calc5(x, attention_mask):
    # Sum of the attention mask to get the index range of valid tokens
    valid_length = attention_mask.sum().item()

    # Slice the query embedding to include only the valid tokens
    valid_embeddings = x[:valid_length]

    M = valid_embeddings.shape[0]
    x_expanded = valid_embeddings.unsqueeze(1).expand(-1, M, -1)
    ed_sum = torch.norm(x_expanded - valid_embeddings, dim=2).sum()
    return ed_sum / (M * M)

def energy_calc5(x, y, attention_mask):
    # Sum of the attention mask to get the index range of valid tokens
    valid_length = attention_mask.sum().item()

    # Slice the query embedding to include only the valid tokens
    valid_x = x[:valid_length]
    M = valid_x.shape[0]  # Number of valid tokens
    N = y.shape[0]  # Number of tokens in document embedding

    # Expand tensors to create all pairs of vectors
    x_expanded = valid_x.unsqueeze(1).expand(-1, N, -1)
    y_expanded = y.unsqueeze(0).expand(M, -1, -1)
    pairwise_diff = x_expanded - y_expanded

    # Compute Euclidean distances with torch.norm
    distances = torch.norm(pairwise_diff, dim=2)

    # Sum distances and normalize by the valid token count
    ed_sum = torch.sum(distances)
    return 2 * ed_sum / (M * N)


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


def energy_distance_old(x, y, attention_mask):
    # Shape of x: [batch_size, num_queries, query_dim]
    # Shape of y: [batch_size, doc_dim] [16, 768]
    device = x.device
    batch_size, num_queries, query_dim = x.shape

    #print("first shape", x[0].shape)
    #print("Shape of tensor containing sentences:", y.shape)
    num_negatives = y.shape[1]
    # Pre-calculate energy for all queries in the batch
    ed_queries = torch.stack([ed_calc(query, attention_mask[i]) for i, query in enumerate(x)]).to(device)

    # Initialize a tensor to store the energy distances
    energy_distances = torch.zeros(batch_size, batch_size, device=device)

    for i in range(batch_size):
        for j in range(batch_size):
            # Calculate energy distance between query i and document i
            ed_query = ed_queries[i]
            energy_distances[i][j] = energy_calc(x[i], y[j].reshape(1,-1), attention_mask[i]) - ed_query

    return energy_distances.requires_grad_()

def energy_distance(x, y, attention_mask):
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

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 1.0, similarity_fct=energy_distance):
        """
        This loss expects as input a batch consisting of sentence pairs ``(a_1, p_1), (a_2, p_2)..., (a_n, p_n)``
        where we assume that ``(a_i, p_i)`` are a positive pair and ``(a_i, p_j)`` for ``i != j`` a negative pair.

        For each ``a_i``, it uses all other ``p_j`` as negative samples, i.e., for ``a_i``, we have 1 positive example
        (``p_i``) and ``n-1`` negative examples (``p_j``). It then minimizes the negative log-likehood for softmax
        normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, relevant_doc)) as it will sample in each batch ``n-1`` negative docs randomly.

        The performance usually increases with increasing batch sizes.

        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        ``(a_1, p_1, n_1), (a_2, p_2, n_2)``. Then, ``n_1`` is a hard negative for ``(a_1, p_1)``. The loss will use for
        the pair ``(a_i, p_i)`` all ``p_j`` for ``j != i`` and all ``n_j`` as negatives.

        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - `Training Examples > Natural Language Inference <../../examples/training/nli/README.html>`_
            - `Training Examples > Paraphrase Data <../../examples/training/paraphrases/README.html>`_
            - `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_
            - `Training Examples > MS MARCO <../../examples/training/ms_marco/README.html>`_
            - `Unsupervised Learning > SimCSE <../../examples/unsupervised_learning/SimCSE/README.html>`_
            - `Unsupervised Learning > GenQ <../../examples/unsupervised_learning/query_generation/README.html>`_

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Relations:
            - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it requires more
              training time.
            - :class:`MultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but with an additional loss term.
            - :class:`GISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `GISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses, InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('distilbert-base-uncased')
                train_examples = [
                    InputExample(texts=['Anchor 1', 'Positive 1']),
                    InputExample(texts=['Anchor 2', 'Positive 2']),
                ]
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
                train_loss = losses.MultipleNegativesRankingLoss(model=model)
                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        embeddings_a = self.model(sentence_features[0])["token_embeddings"]
        #print("Anchor dimensions:", embeddings_a.size())
        #embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        #print("Pos and Neg Sentence dimensions:", embeddings_b.size())
        attention_mask = self.model(sentence_features[0])["attention_mask"]        
        scores = self.similarity_fct(embeddings_a, embeddings_b, attention_mask) * self.scale * -1
        #print("embeddings_a:", embeddings_a.device)
        #print("embeddings_b:", embeddings_b.device)
        #print("attention_mask:", attention_mask.device)
        #print("scores:", scores.device)
        #print("Score tensor dimensions:", scores.size())
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        #print("labels", labels.device)
        return self.cross_entropy_loss(scores, labels)

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
