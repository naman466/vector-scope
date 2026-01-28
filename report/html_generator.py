import plotly.graph_objects as go
from typing import Optional


class HTMLReport:
    
    def __init__(self, figure : go.Figure, title : str = "VectorScope Report"):
        self.figure = figure
        self.title = title
    
    def save(self, filename : str):
        self.figure.write_html(filename)
        print(f"Report saved to {filename}")
    
    def show(self):
        self.figure.show()
