import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Sample data
data = {
    "hospital": ["General_Hospital", "City_Medical", "General_Hospital"],
    "organ_id": ["O0_371", "O0_372", "O0_373"],
    "patient_id": ["P0_771", "P0_772", "P0_773"],
    "matching_score": [4.6, 5.2, 3.9],
    "transplant_outcome": ["Success", "Rejection", "Success"],
    "patient_wait_time": [0, 10, 5],
    "expected_lifespan": [0.9, 1.2, 0.8],
    "donor_type": ["living", "deceased", "living"],
    "donor_subtype": ["related", "SCD", "paired_exchange"],
    "donor_age": [61, 45, 50],
    "organ_quality": [0.32, 0.55, 0.42],
    "patient_urgency": [3, 4, 2]
}
df = pd.DataFrame(data)

# Build graph
G = nx.DiGraph()

for idx, row in df.iterrows():
    donor = f"Donor:{row['organ_id']}"
    patient = f"Patient:{row['patient_id']}"
    G.add_node(donor, type='donor', hospital=row['hospital'])
    G.add_node(patient, type='patient', urgency=row['patient_urgency'])
    G.add_edge(donor, patient, weight=row['matching_score'], outcome=row['transplant_outcome'])

# Visualize
st.title("Kidney Transplant Allocation Graph")
fig, ax = plt.subplots(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
edge_labels = nx.get_edge_attributes(G, 'weight')
node_colors = ['skyblue' if G.nodes[n]['type'] == 'donor' else 'lightgreen' for n in G.nodes]

nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=8, ax=ax)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d:.1f}" for (u, v), d in edge_labels.items()}, ax=ax)

st.pyplot(fig)