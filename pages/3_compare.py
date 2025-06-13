import numpy as np
import pandas as pd
import streamlit as st
import os
import sys

log_paths = [log.removesuffix('.txt') for log in os.listdir('./logs/decisions')]
if len(log_paths) == 0:
    st.write("No logs saved.")
    exit()

log_1, log_2 = st.columns(2)
selected_1 = log_1.selectbox('Select log 1: ', log_paths, key=1)
selected_2 = log_2.selectbox('Select log 2: ', log_paths, key=2)

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
