import streamlit as st
import numpy as np
import time

st.set_page_config(page_title="AutoGuard-RL Dashboard", layout="wide")
st.title("AutoGuard-RL Dashboard 🚗")
run_demo = st.sidebar.button("Run Demo")

frame_placeholder = st.empty()
status_placeholder = st.empty()

if run_demo:
    for i in range(200):
        frame_placeholder.image(np.zeros((240,320,3), dtype=np.uint8))
        status_placeholder.write(f"Step {i} | Safety Score: {round((i % 10)/10, 2)}")
        time.sleep(0.05)
else:
    st.info("Press 'Run Demo' to start simulation loop.")
