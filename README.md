# VectorScope

A Python toolkit for visualizing vector embeddings in RAG pipelines.

## Overview

VectorScope provides tools to visualize high-dimensional vector embeddings in 2D/3D space. It helps to analyze retreival quality by showing geometric relationships between queries and documents. 

## Installation

```bash
pip install numpy pandas scikit-learn plotly umap-learn
```

For ChromaDB support, also install:

```bash
pip install chromadb
```

## Usage

## Basic Example

```python
from vectorscope import Scope
from vectorscope.connectors import ManualConnector
import numpy as np

embeddings = np.random.randn(1000, 1536)
documents = ["Your text here" for _ in range(1000)] # Replace with your documents

connector = ManualConnector(embeddings=embeddings, documents=documents)
scope = Scope(connector=connector)
scope.ingest()

report = scope.visualize()
report.show()
```

## Query Example

```python
query_embedding = np.random.randn(1536) # Replace with your query embedding

report = scope.trace(
    query_text="example query",
    query_embedding=query_embedding,
    top_k=5
)
report.show()
```

## ChromaDB Integration

```python
from vectorscope.connectors import ChromaConnector

connector = ChromaConnector(path="./chroma_db", collection="collection_name")
scope = Scope(connector=connector)
scope.ingest()
```

## Features

- Dimensionality reduction using UMAP, PCA, or t-SNE
- Automatic clustering with KMeans or DBSCAN
- Interactive 2D/3D visualizations with Plotly
- Support for manual and ChromaDB connectors
- Local execution

## Repository Structure

```bash
vectorscope/
├── core/          # Projection and clustering logic
├── connectors/    # Database integrations
├── visualizers/   # Plot generation
├── report/        # HTML report generation
└── api.py         # Main user interface
```

## Extending 

You can create custom connectors by implementing the `BaseConnector` interface in `vectorscope.connectors`.

```python
from vectorscope.connectors.base import BaseConnector

class CustomConnector(BaseConnector):
    def fetch_embeddings(self, limit=1000):
        # Implementation
        pass
    
    def fetch_metadata(self, limit=1000):
        # Implementation
        pass
    
    def fetch_documents(self, ids):
        # Implementation
        pass
```

## License 

This project is licensed under the MIT License. See the `LICENSE` file for details.