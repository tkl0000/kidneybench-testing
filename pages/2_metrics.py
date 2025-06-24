import numpy as np
import pandas as pd
import streamlit as st
import os
import sys
module_path = os.path.relpath('./utils')
sys.path.insert(1, module_path) # Insert at the beginning of the search path
import metrics

log_paths = [log.removesuffix('.txt') for log in os.listdir('./logs/decisions')]
if len(log_paths) == 0:
    st.write("No logs saved.")
    exit()

selected = st.selectbox("Select log to view: ", log_paths)

decisions_path = f'./logs/decisions/{selected}.txt'
outcomes_path = f'./logs/outcomes/{selected}.csv'

outcomes_metrics = metrics.extract_metrics(outcomes_path)
wait_times = metrics.extract_wait_times(outcomes_path)
matching_scores = metrics.extract_matching_scores(outcomes_path)

st.write("**Metrics**")
st.table(outcomes_metrics)

st.write("**Wait Times**")
st.bar_chart(wait_times)

st.write("**Matching Scores**")
st.bar_chart(matching_scores)

# st.write("**Outcomes**")
# st.dataframe(pd.read_csv(outcomes_path))

# st.write("**Decisions**")
# st.dataframe(pd.read_csv(decisions_path))

# for k, v in outcomes_metrics.items():
#     print(k, v)
#     st.write(f'**{k}**')
#     st.write(f'{v}')
