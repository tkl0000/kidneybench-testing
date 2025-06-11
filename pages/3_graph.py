import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

# Create a simple graph
G = nx.DiGraph()
G.add_edge("DonorA", "Patient1")
G.add_edge("DonorB", "Patient2")
G.add_edge("DonorA", "Patient2")  # optional extra edge

# Create and render PyVis network
net = Network(notebook=False, directed=True)
net.from_nx(G)

# Save to temp HTML and display
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    net.save_graph(tmp_file.name)
    components.html(open(tmp_file.name, "r").read(), height=500)