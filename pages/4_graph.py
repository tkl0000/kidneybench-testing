import streamlit as st
import pandas as pd
import os
from pyvis.network import Network
import streamlit.components.v1 as components

# Check logs
log_paths = [log.removesuffix('.txt') for log in os.listdir('./logs/decisions')]
if len(log_paths) == 0:
    st.write("No logs saved.")
    st.stop()

selected = st.selectbox("Select log to view:", log_paths)
data= pd.read_csv(f'./logs/outcomes/{selected}.csv')

df = pd.DataFrame(data)

net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")

for _, row in df.iterrows():
    donor = f"Donor:{row['organ_id']}"
    patient = f"Patient:{row['patient_id']}"

    net.add_node(
        donor,
        label=donor,
        title=f"Hospital: {row['hospital']}\nOrgan Quality: {row['organ_quality']}",
        color="blue"
    )

    net.add_node(
        patient,
        label=patient,
        title=f"Urgency: {row['patient_urgency']}\nExpected Lifespan: {row['expected_lifespan']}",
        color="green"
    )

    edge_label = f"{row['matching_score']:.1f} ({row['transplant_outcome']})"
    net.add_edge(donor, patient, value=row['matching_score'], title=edge_label)

net.set_options("""
var options = {
  "edges": {
    "arrows": {
      "to": {
        "enabled": true
      }
    }
  },
  "physics": {
    "stabilization": {
      "iterations": 150
    }
  }
}
""")

net.save_graph("graph.html")
st.title("Kidney Transplant Allocation Network")
components.html(open("graph.html", "r", encoding="utf-8").read(), height=650, scrolling=True)