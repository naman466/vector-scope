import numpy as np
from typing import List, Dict, Any
from .base import BaseConnector


class ChromaConnector(BaseConnector):
    
    def __init__(self, path : str = "./chroma_db", collection : str = "default"):
        self.path = path
        self.collection_name = collection
        self.client = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.path)
            self.collection = self.client.get_collection(name=self.collection_name)
        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        except Exception as e:
            raise ConnectionError(f"Could not connect to ChromaDB: {e}")
    
    def fetch_embeddings(self, limit : int = 1000) -> np.ndarray:
        results = self.collection.get(limit=limit, include=["embeddings"])
        return np.array(results["embeddings"])
    
    def fetch_metadata(self, limit : int = 1000) -> List[Dict[str, Any]]:
        results = self.collection.get(limit=limit, include=["metadatas"])
        return results["metadatas"]
    
    def fetch_documents(self, ids : List[int]) -> List[str]:
        results = self.collection.get(ids=[str(i) for i in ids], include=["documents"])
        return results["documents"]
