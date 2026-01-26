from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any

class BaseConnector(ABC):

    @abstractmethod
    def fetch_embeddings(self, limit : int = 1000) -> np.ndarray:
        pass

    @abstractmethod
    def fetch_metadata(self, limit : 1000)  -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def fetch_documents(self, ids: List[int]) -> List[str]:
        pass
