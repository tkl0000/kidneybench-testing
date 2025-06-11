import streamlit as st

options = ["Greedy", "Optimal", "Dynamic", "Graph-Based"]
policy_type = st.selectbox("Select policy type: ", options)
st.text(policy_type)