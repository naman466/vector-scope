import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from .base import BaseConnector

class ManualConnector(BaseConnector):

    def __init__(self, embeddings: Union[np.ndarray, pd.DataFrame],
                 documents: Optional[List[str]] = None,
                 metadata: Optional[List[Dict[str, Any]]] = None):
        if isinstance(embeddings, pd.DataFrame):
            self.embeddings = embeddings.values
        else:
            self.embeddings = embeddings
        
        self.documents = documents or [f"Document {i}" for i in range(len(self.embeddings))]
        self.metadata = metadata or [{"id": i} for i in range(len(self.embeddings))]
    
    def fetch_embeddings(self, limit: int = 1000) -> np.ndarray:
        return self.embeddings[:limit]
    
    def fetch_metadata(self, limit: int = 1000) -> List[Dict[str, Any]]:
        return self.metadata[:limit]
    
    def fetch_documents(self, ids: List[int]) -> List[str]:
        return [self.documents[i] for i in ids if i < len(self.documents)]
