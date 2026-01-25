import numpy as np
from typing import List


class MetricsCalculator:
    
    @staticmethod
    def retrieval_quality_score(query_embedding: np.ndarray, 
                               retrieved_embeddings: np.ndarray,
                               all_embeddings: np.ndarray) -> float:
        query_embedding = query_embedding.reshape(1, -1)
        
        dist_retrieved = np.linalg.norm(retrieved_embeddings - query_embedding, axis=1)
        mean_dist_retrieved = dist_retrieved.mean()
        
        dist_all = np.linalg.norm(all_embeddings - query_embedding, axis=1)
        mean_dist_all = dist_all.mean()
        
        if mean_dist_all == 0:
            return 1.0
        
        score = 1 / (1 + mean_dist_retrieved / mean_dist_all)
        return score
    
    @staticmethod
    def find_missed_opportunities(query_embedding: np.ndarray,
                                 retrieved_indices: List[int],
                                 all_embeddings: np.ndarray,
                                 threshold_percentile: float = 10) -> List[int]:
        query_embedding = query_embedding.reshape(1, -1)
        
        distances = np.linalg.norm(all_embeddings - query_embedding, axis=1)
        
        threshold = np.percentile(distances, threshold_percentile)
        
        close_indices = np.where(distances <= threshold)[0]
        missed = [idx for idx in close_indices if idx not in retrieved_indices]
        
        return missed
