import streamlit as st
import os
import sys

log_paths = [log.removesuffix('.txt') for log in os.listdir('./logs/decisions')]
if len(log_paths) == 0:
    st.write("No logs saved.")
    exit()

st.selectbox("Select log to view: ", log_paths)

