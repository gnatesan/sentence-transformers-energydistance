from enum import Enum
from typing import Iterable, Dict
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from sentence_transformers.SentenceTransformer import SentenceTransformer

#Function to calculate the energy of all vectors in a query
def ed_calc(x):
    M = x.shape[0]
    x_expanded = x.unsqueeze(1).expand(-1, M, -1)
    ed_sum = torch.norm(x_expanded - x, dim=2).sum()
    return ed_sum / (M * M)

#Function to calcjulate the energy between a single vecctor document and multi-vector query
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

#Function to calculate the energy distance between a batch of queries and a batch of documents.
#Queries are represented as a tensor of 2d tensors, and documents are represented by a tensor of 
#1d tensors. Implementation is based off of https://pages.stat.wisc.edu/~wahba/stat860public/pdf4/Energy/EnergyDistance10.1002-wics.1375.pdf
def energy_distance(x, y):
    # Shape of x: [batch_size, num_queries, query_dim]
    # Shape of y: [batch_size, doc_dim]
    device = x.device
    batch_size, num_queries, query_dim = x.shape

    # Pre-calculate energy for all queries in the batch
    ed_queries = torch.stack([ed_calc(query) for query in x], dim=0).to(device)

    # Initialize a tensor to store the energy distances
    energy_distances = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        # Calculate energy distance between query i and document i
        ed_query = ed_queries[i]
        energy_distances[i] = energy_calc(x[i], y[i].reshape(1,-1)).item() - ed_query.item()

    return energy_distances.requires_grad_()

class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)
    ENERGY_DISTANCE = lambda x, y: energy_distance(x, y)


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=SiameseDistanceMetric.ENERGY_DISTANCE,
        margin: float = 0.5,
        size_average: bool = True,
    ):
        """
        Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
        two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

        :param model: SentenceTransformer model
        :param distance_metric: Function that returns a distance between two embeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
        :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
        :param size_average: Average by the size of the mini-batch.

        References:
            * Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            * `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_

        Requirements:
            1. (anchor, positive/negative) pairs

        Relations:
            - :class:`OnlineContrastiveLoss` is similar, but uses hard positive and hard negative pairs.
            It often yields better results.

        Inputs:
            +-----------------------------------------------+------------------------------+
            | Texts                                         | Labels                       |
            +===============================================+==============================+
            | (anchor, positive/negative) pairs             | 1 if positive, 0 if negative |
            +-----------------------------------------------+------------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses
                from sentence_transformers.readers import InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('all-MiniLM-L6-v2')
                train_examples = [
                    InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
                    InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0),
                ]

                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
                train_loss = losses.ContrastiveLoss(model=model)

                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "SiameseDistanceMetric.{}".format(name)
                break

        return {"distance_metric": distance_metric_name, "margin": self.margin, "size_average": self.size_average}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        #encode the query into multiple vectors
        rep_anchor = self.model(sentence_features[0])["token_embeddings"]
        #encode the document into a single vector
        rep_other = self.model(sentence_features[1])["sentence_embedding"]
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (
            labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2)
        )
        return losses.mean() if self.size_average else losses.sum()
