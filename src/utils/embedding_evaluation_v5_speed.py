import logging
import time
import faiss
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EmbeddingEvaluator:
    """
    Efficiently evaluates embedding quality with optimized methods.
    """
    def __init__(self, embeddings_path: str, n_clusters: int = 5, reduce_dim: int = 50):
        self.embeddings_path = embeddings_path
        self.embeddings = self.load_embeddings()
        self.n_clusters = n_clusters
        self.reduced_embeddings = self.reduce_dimensionality(reduce_dim)

    def load_embeddings(self) -> np.ndarray:
        """Load embeddings efficiently using memory mapping."""
        try:
            start_time = time.time()
            embeddings = np.load(self.embeddings_path, mmap_mode="r").astype(np.float32)
            logging.info(f"✅ Embeddings loaded. Shape: {embeddings.shape}. Time: {time.time() - start_time:.2f}s")
            return embeddings
        except Exception as e:
            logging.error(f"❌ Error loading embeddings: {e}")
            raise

    def reduce_dimensionality(self, n_components: int = 30) -> np.ndarray:
        """Reduce dimensionality using optimized PCA."""
        logging.info(f"Reducing embeddings to {n_components} dimensions...")
        start_time = time.time()
        pca = PCA(n_components=n_components, svd_solver="randomized")
        reduced = pca.fit_transform(self.embeddings)
        logging.info(f"✅ PCA completed in {time.time() - start_time:.2f}s")
        return reduced

    def cosine_similarity_faiss(self, top_k: int = 10) -> np.ndarray:
        """Compute cosine similarity using optimized Faiss."""
        logging.info("Computing cosine similarity with Faiss...")
        start_time = time.time()
        d = self.reduced_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.reduced_embeddings)  # Normalize before adding
        index.add(self.reduced_embeddings)
        _, indices = index.search(self.reduced_embeddings, top_k)
        logging.info(f"✅ Cosine similarity computed in {time.time() - start_time:.2f}s")
        return indices

    def nearest_neighbors_consistency(self, n_neighbors: int = 5) -> float:
        """Compute k-NN consistency efficiently using Faiss."""
        logging.info("Computing nearest neighbor consistency...")
        start_time = time.time()
        index = faiss.IndexFlatL2(self.reduced_embeddings.shape[1])
        index.add(self.reduced_embeddings)
        _, distances = index.search(self.reduced_embeddings, n_neighbors + 1)
        consistency = np.mean(distances[:, 1:])  # Exclude self-distance
        logging.info(f"✅ k-NN consistency computed in {time.time() - start_time:.2f}s")
        return consistency

    def evaluate_clustering_quality(self) -> dict:
         
        """Use HDBSCAN for clustering and compute quality metrics."""
        logging.info("Evaluating clustering quality with HDBSCAN...")
        # HDBSCAN clustering with faster execution
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom')
        labels = clusterer.fit_predict(self.reduced_embeddings)
        return {
            "num_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "outliers": list(labels).count(-1),
        }

# Example usage
if __name__ == "__main__":
    evaluator = EmbeddingEvaluator(r"C:\Users\batti\topic-modeling\data\word2vec_embeddings.npy")
    logging.info(f"Cosine Faiss similarity: {evaluator.cosine_similarity_faiss()}")
    logging.info(f"Nearest Neighbor Consistency: {evaluator.nearest_neighbors_consistency():.4f}")
    logging.info(f"Clustering Quality: {evaluator.evaluate_clustering_quality()}")
