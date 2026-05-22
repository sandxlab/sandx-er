"""sandx-er — Entity Resolution infrastructure."""

from sandx_er.blocking import EmbeddingANNBlocking, LSHBlocking, SortedNeighborhoodBlocking
from sandx_er.clustering import ConnectedComponentsClustering, CorrelationClustering
from sandx_er.matching import CosineSimilarityScorer, JaccardScorer
from sandx_er.resolver import EntityCluster, EntityResolver, ResolutionResult

__version__ = "0.1.0"
__all__ = [
    "EntityResolver",
    "EntityCluster",
    "ResolutionResult",
    "LSHBlocking",
    "SortedNeighborhoodBlocking",
    "EmbeddingANNBlocking",
    "ConnectedComponentsClustering",
    "CorrelationClustering",
    "JaccardScorer",
    "CosineSimilarityScorer",
]
