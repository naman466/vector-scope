import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any
from .styles import ColorPalette


class PlotGenerator:
    
    @staticmethod
    def create_galaxy_view(embeddings_2d : np.ndarray,
                          labels : np.ndarray,
                          documents : List[str],
                          metadata : List[Dict[str, Any]],
                          title : str = "Vector Space Galaxy View") -> go.Figure:

        fig = go.Figure()
        
        # Plot each cluster
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_points = embeddings_2d[mask]
            cluster_docs = [documents[i] for i in np.where(mask)[0]]
            
            hover_text = [f"Cluster {cluster_id}<br>{doc[:100]}..." 
                         for doc in cluster_docs]
            
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=8,
                    color=ColorPalette.DEFAULT[cluster_id % len(ColorPalette.DEFAULT)],
                    opacity=0.7
                ),
                text=hover_text,
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode='closest',
            template='plotly_white',
            width=1000,
            height=700
        )
        
        return fig
    
    @staticmethod
    def create_query_analysis(embeddings_2d : np.ndarray,
                            query_2d : np.ndarray,
                            retrieved_indices : List[int],
                            missed_indices : List[int],
                            documents : List[str],
                            title : str = "Query Analysis View") -> go.Figure:

        fig = go.Figure()
        
        # All documents (background)
        fig.add_trace(go.Scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            mode='markers',
            name='All Documents',
            marker=dict(size=6, color='lightgray', opacity=0.3),
            hoverinfo='skip'
        ))
        
        # Retrieved documents
        if retrieved_indices:
            retrieved_points = embeddings_2d[retrieved_indices]
            retrieved_docs = [documents[i][:100] for i in retrieved_indices]
            fig.add_trace(go.Scatter(
                x=retrieved_points[:, 0],
                y=retrieved_points[:, 1],
                mode='markers',
                name='Retrieved',
                marker=dict(size=12, color=ColorPalette.RETRIEVED_COLOR),
                text=retrieved_docs,
                hoverinfo='text'
            ))
        
        # Missed opportunities
        if missed_indices:
            missed_points = embeddings_2d[missed_indices]
            missed_docs = [documents[i][:100] for i in missed_indices]
            fig.add_trace(go.Scatter(
                x=missed_points[:, 0],
                y=missed_points[:, 1],
                mode='markers',
                name='Missed Opportunities',
                marker=dict(size=12, color=ColorPalette.MISSED_COLOR),
                text=missed_docs,
                hoverinfo='text'
            ))
        
        # Query point
        fig.add_trace(go.Scatter(
            x=[query_2d[0]],
            y=[query_2d[1]],
            mode='markers',
            name='Query',
            marker=dict(size=20, color=ColorPalette.QUERY_COLOR, symbol='star'),
            hoverinfo='name'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode='closest',
            template='plotly_white',
            width=1000,
            height=700
        )
        
        return fig