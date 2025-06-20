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

log_1, log_2 = st.columns(2)
selected_1 = log_1.selectbox('Select log 1: ', log_paths, key=1)
selected_2 = log_2.selectbox('Select log 2: ', log_paths, key=2)

outcomes_1_path = f'./logs/outcomes/{selected_1}.csv'
outcomes_2_path = f'./logs/outcomes/{selected_2}.csv'

# metrics_1 = metrics.extract_metrics(outcomes_1_path)
# metrics2 = metrics.extract_metrics(outcomes_2_path)
