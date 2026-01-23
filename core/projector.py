import numpy as np
import pickle
from typing import Literal, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    UMAP_AVAILABLE = Truue
except ImportError:
    UMAP_AVAILABLE = False


class Projector: # Dimensionality reduction for visualization

    def __init__(self, method : Literal["umap", "pca", "tsne"] = "umap", n_components : int = 2, random_state : int = 51):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        self._is_fitted = False
    
    def fit(self, data : np.ndarray) -> 'Projector':
        
        if self.method == "umap":
            if not UMAP_AVAILABLE:
                print("UMAP is not installed. Falling back to PCA, please install UMAP to use it.")
                self.method = "pca"
            else:
                self.reducer = UMAP(
                    n_components = self.n_components,
                    random_state = self.random_state
                    n_neighbors = min(15, len(data) - 1)                          
                )
        
        if self.method == "pca":
            self.reducer = PCA(n_components = self.n_components, random_state = self.random_state)
        elif self.method == "tsne":
            self.reducer = TSNE(n_components = self.n_components, random_state = self.random_state)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'umap', 'pca', or 'tsne'.")
        
        self.reducer.fit(data)
        self._is_fitted = True
        return self
    
    def transform(self, data : np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("The projector has not been fitted yet. Call 'fit' with training data before calling 'transform'.")
        return self.reducer.transform(data)
    
    def fit_transform(self, data : np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
    
    def save(self, filepath : str) -> None:
        if not self._is_fitted:
            raise RuntimeError("The projector has not been fitted yet. Nothing to save.")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, path : str):
        with open(path, 'rb') as f:
            self.reducer = pickle.load(f)
        self._is_fitted = True