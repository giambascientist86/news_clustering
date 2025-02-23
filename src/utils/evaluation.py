from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
import numpy as np

class ClusteringEvaluator:
    """
    Class to evaluate clustering performance using various metrics.
    """
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Initialize evaluator with embeddings and labels.

        :param embeddings: np.ndarray, feature representations of the data points.
        :param labels: np.ndarray, cluster labels assigned by the clustering algorithm.
        """
        self.embeddings = embeddings
        self.labels = labels

    def silhouette_score(self) -> float:
        """Compute Silhouette Score."""
        return silhouette_score(self.embeddings, self.labels)

    def davies_bouldin_score(self) -> float:
        """Compute Davies-Bouldin Score."""
        return davies_bouldin_score(self.embeddings, self.labels)

    def calinski_harabasz_score(self) -> float:
        """Compute Calinski-Harabasz Score."""
        return calinski_harabasz_score(self.embeddings, self.labels)

    def adjusted_rand_index(self, ground_truth: np.ndarray) -> float:
        """Compute Adjusted Rand Index (ARI)."""
        return adjusted_rand_score(ground_truth, self.labels)

    def normalized_mutual_info(self, ground_truth: np.ndarray) -> float:
        """Compute Normalized Mutual Information (NMI)."""
        return normalized_mutual_info_score(ground_truth, self.labels)

    def homogeneity_completeness_v_measure(self, ground_truth: np.ndarray) -> dict:
        """Compute Homogeneity, Completeness, and V-measure."""
        h, c, v = homogeneity_completeness_v_measure(ground_truth, self.labels)
        return {"homogeneity": h, "completeness": c, "v_measure": v}

    def purity_score(self, ground_truth: np.ndarray) -> float:
        """Compute Purity Score."""
        contingency_matrix = np.zeros((len(set(ground_truth)), len(set(self.labels))))
        for i, label in enumerate(ground_truth):
            contingency_matrix[label, self.labels[i]] += 1
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def evaluate_all(self, ground_truth: np.ndarray) -> dict:
        """Compute all metrics and return as a dictionary."""
        return {
            "silhouette_score": self.silhouette_score(),
            "davies_bouldin_score": self.davies_bouldin_score(),
            "calinski_harabasz_score": self.calinski_harabasz_score(),
            "adjusted_rand_index": self.adjusted_rand_index(ground_truth),
            "normalized_mutual_info": self.normalized_mutual_info(ground_truth),
            "homogeneity_completeness_v_measure": self.homogeneity_completeness_v_measure(ground_truth),
            "purity_score": self.purity_score(ground_truth),
        }
