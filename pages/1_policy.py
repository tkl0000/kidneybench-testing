import sys
import os 
module_path = os.path.relpath('./utils')
sys.path.insert(1, module_path) # Insert at the beginning of the search path

from run_sim import run
import streamlit as st
from datetime import datetime


options = ["greedy", ".pth"]
policy_type = st.selectbox("Select policy type: ", options)

def greedy():
    st.write("**Calculate score as a linear combination of the following factors:**")
    # left, right = st.cols(2)
    # with left:
    #     st.write("*Organ Characteristics*")
    #     st.write("")
    #     st.number_input()

def pth_file():
    st.file_uploader("**Upload .pth file**")

match policy_type:
    case "greedy":
        greedy()
    case ".pth":
        pth_file() 
    case _:
        st.write("Error selecting policy type.")

def generate_paths():
    return './logs/outcomes/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv', './logs/decisions/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt' 

outcomes_path, decisions_path = generate_paths()
st.button("Run", on_click=lambda: run(outcomes_path, decisions_path))