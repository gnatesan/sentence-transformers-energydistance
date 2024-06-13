from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F
from enum import Enum
from ..SentenceTransformer import SentenceTransformer

def ed_calc(x):
    M = x.shape[0]
    x_expanded = x.unsqueeze(1).expand(-1, M, -1)
    ed_sum = torch.norm(x_expanded - x, dim=2).sum()
    return ed_sum / (M * M)

def energy_calc(x, y):
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

def energy_distance(x, y):
    # Shape of x: [batch_size, num_queries, query_dim]
    # Shape of y: [batch_size, doc_dim]

    batch_size, num_queries, query_dim = x.shape

    #print("first shape", x[0].shape)

    # Pre-calculate energy for all queries in the batch
    ed_queries = torch.stack([ed_calc(query) for query in x])

    # Initialize a tensor to store the energy distances
    energy_distances = torch.zeros(batch_size)

    for i in range(batch_size):
        # Calculate energy distance between query i and document i
        ed_query = ed_queries[i]
        energy_distances[i] = energy_calc(x[i], y[i].reshape(1,-1)).item() - ed_query.item()

    return energy_distances
    
class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """

    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    ENERGY_DISTANCE = lambda x, y: energy_distance(x, y)


class TripletLoss(nn.Module):
    def __init__(
        self, model: SentenceTransformer, distance_metric=TripletDistanceMetric.ENERGY_DISTANCE, triplet_margin: float = 5
    ):
        """
        This class implements triplet loss. Given a triplet of (anchor, positive, negative),
        the loss minimizes the distance between anchor and positive while it maximizes the distance
        between anchor and negative. It compute the following loss function:

        ``loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0)``.

        Margin is an important hyperparameter and needs to be tuned respectively.

        :param model: SentenceTransformerModel
        :param distance_metric: Function to compute distance between two embeddings. The class TripletDistanceMetric
            contains common distance metrices that can be used.
        :param triplet_margin: The negative should be at least this much further away from the anchor than the positive.

        References:
            - For further details, see: https://en.wikipedia.org/wiki/Triplet_loss

        Requirements:
            1. (anchor, positive, negative) triplets

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer,  SentencesDataset, losses
                from sentence_transformers.readers import InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                train_examples = [
                    InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
                    InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2']),
                ]
                train_batch_size = 1
                train_dataset = SentencesDataset(train_examples, model)
                train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
                train_loss = losses.TripletLoss(model=model)
                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(TripletLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "TripletDistanceMetric.{}".format(name)
                break

        return {"distance_metric": distance_metric_name, "triplet_margin": self.triplet_margin}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        #reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        rep_anchor = self.model(sentence_features[0])["token_embeddings"]
        #rep_anchor, rep_pos, rep_neg = reps
        rep_pos = self.model(sentence_features[1])["sentence_embedding"]
        rep_neg = self.model(sentence_features[2])["sentence_embedding"]
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()
