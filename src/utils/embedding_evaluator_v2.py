import logging
import faiss
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EmbeddingEvaluator:
    """
    A class to evaluate the quality of embeddings using intrinsic and extrinsic metrics.
    """
    def __init__(self, embeddings_path: str):
        """
        Initializes the evaluator by loading embeddings from a .pkl file.
        :param embeddings_path: Path to the saved embeddings file.
        """
        self.embeddings_path = embeddings_path
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        """Loads embeddings from a .pkl file."""
        try:
            with open(self.embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
            
            if not isinstance(embeddings, np.ndarray):
                raise ValueError("Loaded embeddings are not a NumPy array.")

            logging.info(f"✅ Embeddings loaded successfully. Shape: {embeddings.shape}")
            return embeddings
        
        except FileNotFoundError:
            logging.error(f"❌ File not found: {self.embeddings_path}")
            raise
        except Exception as e:
            logging.error(f"❌ Error loading embeddings: {str(e)}")
            raise

    def cosine_similarity_faiss(self, top_k: int = 10) -> np.ndarray:
        """
        Computes an approximate cosine similarity distribution using Faiss.
        :param top_k: Number of nearest neighbors to retrieve.
        """
        logging.info("Computing approximate cosine similarity distribution with Faiss...")

        d = self.embeddings.shape[1]  # Dimensionality of embeddings
        index = faiss.IndexFlatIP(d)
        self.embeddings  = self.embeddings.astype(np.float32) # Inner Product index (cosine similarity)
        faiss.normalize_L2(self.embeddings)  # Normalize for cosine similarity
        index.add(self.embeddings)

        _, indices = index.search(self.embeddings, top_k)
        
        return indices

    def nearest_neighbors_consistency(self, n_neighbors: int = 5) -> float:
        """Evaluates nearest neighbor consistency using k-NN."""
        logging.info("Computing nearest neighbor consistency...")
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn_model.fit(self.embeddings)
        distances, _ = nn_model.kneighbors(self.embeddings)
        avg_dist = np.mean(distances[:, 1:])  # Exclude self-distance
        return avg_dist

    def compute_hopkins_statistic(self, n_samples: int = 50) -> float:
        """Computes the Hopkins statistic to measure clustering tendency."""
        logging.info("Computing clustering tendency (Hopkins statistic)...")
        np.random.seed(42)
        sample_indices = np.random.choice(self.embeddings.shape[0], size=n_samples, replace=False)
        sample_points = self.embeddings[sample_indices]

        random_points = np.random.uniform(
            np.min(self.embeddings, axis=0), np.max(self.embeddings, axis=0), sample_points.shape
        )
        
        # Compute distances
        sample_distances = np.min(np.linalg.norm(sample_points[:, None, :] - self.embeddings[None, :, :], axis=2), axis=1)
        random_distances = np.min(np.linalg.norm(random_points[:, None, :] - self.embeddings[None, :, :], axis=2), axis=1)
        
        hopkins_stat = np.sum(sample_distances) / (np.sum(sample_distances) + np.sum(random_distances))
        return hopkins_stat

    def evaluate_clustering_quality(self, n_clusters: int = 5) -> dict:
        """Evaluates clustering quality using multiple metrics."""
        logging.info("Evaluating clustering quality...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)

        silhouette = silhouette_score(self.embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(self.embeddings, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.embeddings, cluster_labels)

        return {
            "silhouette_score": round(silhouette, 4),
            "davies_bouldin_index": round(davies_bouldin, 4),
            "calinski_harabasz_score": round(calinski_harabasz, 4),
        }

# Example usage
if __name__ == "__main__":
    evaluator = EmbeddingEvaluator(r"C:\Users\batti\topic-modeling\data\tfidf_embeddings.pkl")  # Change path if needed

    #cosine_sim_dist = evaluator.cosine_similarity_faiss()
    #logging.info(f"Mean Cosine Similarity: {np.mean(cosine_sim_dist):.4f}")

    nn_consistency = evaluator.nearest_neighbors_consistency()
    logging.info(f"Nearest Neighbor Consistency: {nn_consistency:.4f}")

    hopkins_stat = evaluator.compute_hopkins_statistic()
    logging.info(f"Clustering Tendency (Hopkins Statistic): {hopkins_stat:.4f}")

    clustering_quality = evaluator.evaluate_clustering_quality()
    logging.info(f"Clustering Quality: {clustering_quality}")