import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EmbeddingEvaluator:
    """
    A class to evaluate the quality of embeddings using intrinsic and extrinsic metrics.
    """
    def __init__(self, embeddings: np.ndarray):
        """
        Initializes the evaluator with the given embeddings.
        :param embeddings: A NumPy array of shape (n_samples, embedding_dim)
        """
        self.embeddings = embeddings
        if not isinstance(self.embeddings, np.ndarray):
            raise ValueError("Embeddings must be a NumPy array.")
    
    def cosine_similarity_distribution(self) -> np.ndarray:
        """Computes the cosine similarity distribution of the embeddings."""
        logging.info("Computing cosine similarity distribution...")
        sim_matrix = cosine_similarity(self.embeddings)
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        return upper_triangle

    def nearest_neighbors_consistency(self, n_neighbors: int = 5) -> float:
        """Evaluates nearest neighbor consistency using k-NN."""
        logging.info("Computing nearest neighbor consistency...")
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn_model.fit(self.embeddings)
        distances, indices = nn_model.kneighbors(self.embeddings)
        avg_dist = np.mean(distances[:, 1:])  # Exclude self-distance
        return avg_dist

    def clustering_tendency(self) -> float:
        """Estimates clustering tendency using the Hopkins statistic."""
        logging.info("Computing clustering tendency (Hopkins statistic)...")
        n_samples = self.embeddings.shape[0]
        random_points = np.random.uniform(np.min(self.embeddings, axis=0), np.max(self.embeddings, axis=0), self.embeddings.shape)
        distances = np.min(np.linalg.norm(self.embeddings[:, None, :] - random_points[None, :, :], axis=2), axis=1)
        hopkins_stat = np.sum(distances) / (np.sum(distances) + np.sum(pdist(self.embeddings)))
        return hopkins_stat

    def evaluate_clustering_quality(self, n_clusters: int = 5) -> dict:
        """Evaluates clustering quality using silhouette score and Davies-Bouldin index."""
        logging.info("Evaluating clustering quality...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        silhouette = silhouette_score(self.embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(self.embeddings, cluster_labels)
        return {"silhouette_score": silhouette, "davies_bouldin_index": davies_bouldin}

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    sample_embeddings = np.random.rand(100, 50)  # Simulated embeddings
    evaluator = EmbeddingEvaluator(sample_embeddings)
    
    cosine_sim_dist = evaluator.cosine_similarity_distribution()
    logging.info(f"Mean Cosine Similarity: {np.mean(cosine_sim_dist):.4f}")
    
    nn_consistency = evaluator.nearest_neighbors_consistency()
    logging.info(f"Nearest Neighbor Consistency: {nn_consistency:.4f}")
    
    clustering_tendency = evaluator.clustering_tendency()
    logging.info(f"Clustering Tendency (Hopkins Statistic): {clustering_tendency:.4f}")
    
    clustering_quality = evaluator.evaluate_clustering_quality()
    logging.info(f"Clustering Quality: {clustering_quality}")
