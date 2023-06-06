import streamlit as st

st.markdown(
"""
# Research Question
How does the model decide "C0" is a valid move?

# Observations
1. MLP neurons (especially Layer 5 and 6) are looking for some pattern in the board state, if the pattern is found, then the neuron will write to W_U saying the target cell is a valid move.
2. (more like a hypothesis for now) The MLP neurons will "destroy" the current board state (by writing to blank direction indicating the target cell is not blank) after confirming the target cell is a valid move.
3. There are also neurons searching for "invalid moves" (deducting the logits of target cell) by checking whether one of the board direction is occupied by the opponent.
"""
)
