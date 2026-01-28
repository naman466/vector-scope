import numpy as np
from typing import Optional, List, Literal
from .connectors.base import BaseConnector
from .core.projector import Projector
from .core.clustering import ClusterAnalyzer
from .core.metrics import MetricsCalculator
from .visualizers.plots import PlotGenerator
from .report.html_generator import HTMLReport


class Scope:
    
    def __init__(self, connector : BaseConnector, 
                 projection_method : Literal["umap", "pca", "tsne"] = "umap",
                 n_clusters : int = 5):
        self.connector = connector
        self.projection_method = projection_method
        self.n_clusters = n_clusters
        
        # Internal state
        self.embeddings = None
        self.embeddings_2d = None
        self.documents = None
        self.metadata = None
        self.cluster_labels = None
        
        # Components
        self.projector = Projector(method=projection_method)
        self.cluster_analyzer = ClusterAnalyzer(n_clusters=n_clusters)
        self.metrics = MetricsCalculator()
    
    def ingest(self, limit : int = 1000):
        self.embeddings = self.connector.fetch_embeddings(limit=limit)
        self.metadata = self.connector.fetch_metadata(limit=limit)
        
        # Fetch documents
        doc_ids = list(range(len(self.embeddings)))
        self.documents = self.connector.fetch_documents(doc_ids)
        
        self.embeddings_2d = self.projector.fit_transform(self.embeddings)
        
        self.cluster_labels = self.cluster_analyzer.fit(self.embeddings_2d)
        
        return self
    
    def visualize(self, title : str = "Vector Space Map") -> HTMLReport:
        if self.embeddings_2d is None:
            raise ValueError("Must call ingest() before visualize()")
        
        fig = PlotGenerator.create_galaxy_view(
            self.embeddings_2d,
            self.cluster_labels,
            self.documents,
            self.metadata,
            title=title
        )
        
        return HTMLReport(fig, title)
    
    def trace(self, query_text : str, query_embedding : Optional[np.ndarray] = None,
             top_k: int = 5, retrieved_indices: Optional[List[int]] = None) -> HTMLReport:
        if self.embeddings_2d is None:
            raise ValueError("Must call ingest() before trace()")
        
        if query_embedding is None:
            raise ValueError("query_embedding must be provided (generate from your embedding model)")
        
        query_2d = self.projector.transform(query_embedding.reshape(1, -1))[0]
        
        if retrieved_indices is None:
            distances = np.linalg.norm(self.embeddings - query_embedding.reshape(1, -1), axis=1)
            retrieved_indices = np.argsort(distances)[:top_k].tolist()
        
        missed_indices = self.metrics.find_missed_opportunities(
            query_embedding,
            retrieved_indices,
            self.embeddings,
            threshold_percentile=10
        )
                        
        # Generate visualization
        fig = PlotGenerator.create_query_analysis(
            self.embeddings_2d,
            query_2d,
            retrieved_indices,
            missed_indices,
            self.documents,
            title=f"Query Analysis: '{query_text}'"
        )
        
        return HTMLReport(fig, f"Query Trace: {query_text}")
