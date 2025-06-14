import numpy as np
import pandas as pd
import streamlit as st
import os
import sys

log_paths = [log.removesuffix('.txt') for log in os.listdir('./logs/decisions')]
if len(log_paths) == 0:
    st.write("No logs saved.")
    exit()

selected = st.selectbox("Select log to view: ", log_paths)

def extract_metrics(filepath):
    df = pd.read_csv(filepath)

    avg_wait = df['patient_wait_time'].mean()
    avg_score = df['matching_score'].mean()
    avg_expected_life = df['expected_lifespan'].mean()
    avg_organ_quality = df['organ_quality'].mean()
    avg_donor_age = df['donor_age'].mean()
    avg_patient_urgency = df['patient_urgency'].mean()
    
    match_rate = (df["transplant_outcome"] == "Success").mean()

    return {
        "match_rate" : match_rate,
    }

decisions_path = f'./logs/decisions/{selected}.txt'
outcomes_path = f'./logs/outcomes/{selected}.csv'

outcomes_metrics = extract_metrics(outcomes_path)

# st.write("**Outcomes**")
# st.dataframe(pd.read_csv(outcomes_path))

# st.write("**Decisions**")
# st.dataframe(pd.read_csv(decisions_path))

st.table(outcomes_metrics)

# for k, v in outcomes_metrics.items():
#     print(k, v)
#     st.write(f'**{k}**')
#     st.write(f'{v}')
