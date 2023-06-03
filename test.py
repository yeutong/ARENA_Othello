import streamlit as st

# %%
# %pip install git+https://github.com/neelnanda-io/neel-plotly

# %%
import streamlit as st
import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import einops
import wandb
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
import itertools
import random
from IPython.display import display
import wandb
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from typing import List, Union, Optional, Tuple, Callable, Dict
import typeguard
from functools import partial
import copy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm.notebook import tqdm
from dataclasses import dataclass
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning as pl
from rich import print as rprint
import pandas as pd

from plotly_utils import imshow
from neel_plotly import scatter, line

device = t.device("cuda" if t.cuda.is_available() else "cpu")
working_dir = Path(os.getcwd())

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"
# %%
if MAIN:
    cfg = HookedTransformerConfig(
        n_layers = 8,
        d_model = 512,
        d_head = 64,
        n_heads = 8,
        d_mlp = 2048,
        d_vocab = 61,
        n_ctx = 59,
        act_fn="gelu",
        normalization_type="LNPre",
        device=device,
    )
    model = HookedTransformer(cfg)
# %%
if MAIN:
    sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
    # champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
    model.load_state_dict(sd)

# %%
if MAIN:
    os.chdir(working_dir)

    OTHELLO_ROOT = (working_dir / "othello_world").resolve()
    OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

    # if not OTHELLO_ROOT.exists():
    #     !git clone https://github.com/likenneth/othello_world

    sys.path.append(str(OTHELLO_MECHINT_ROOT))

# %%
from mech_interp_othello_utils import plot_board, plot_single_board, plot_board_log_probs, to_string, to_int, int_to_label, string_to_label, OthelloBoardState
# %%
# Load board data as ints (i.e. 0 to 60)

if MAIN:
    board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
    # Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
    board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)

    assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
    assert board_seqs_int.max() == 60

    num_games, length_of_game = board_seqs_int.shape
    print("Number of games:", num_games)
    print("Length of game:", length_of_game)
# %%
# Define possible indices (excluding the four center squares)

if MAIN:
    stoi_indices = [i for i in range(64) if i not in [27, 28, 35, 36]]

    # Define our rows, and the function that converts an index into a (row, column) label, e.g. `E2`
    alpha = "ABCDEFGH"

def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"

# Get our list of board labels

if MAIN:
    board_labels = list(map(to_board_label, stoi_indices))
# %%
if MAIN:
    moves_int = board_seqs_int[0, :30]

    # This is implicitly converted to a batch of size 1
    logits: Tensor = model(moves_int)
    print("logits:", logits.shape)
# %%
if MAIN:
    logit_vec = logits[0, -1]
    log_probs = logit_vec.log_softmax(-1)
    # Remove the "pass" move (the zeroth vocab item)
    log_probs = log_probs[1:]
    assert len(log_probs)==60

    # Set all cells to -13 by default, for a very negative log prob - this means the middle cells don't show up as mattering
    temp_board_state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
    temp_board_state.flatten()[stoi_indices] = log_probs
# %%
def plot_square_as_board(state, diverging_scale=True, **kwargs):
    '''Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0'''
    kwargs = {
        "y": [i for i in alpha],
        "x": [str(i) for i in range(8)],
        "color_continuous_scale": "RdBu" if diverging_scale else "Blues",
        "color_continuous_midpoint": 0. if diverging_scale else None,
        "aspect": "equal",
        **kwargs
    }
    
    return imshow(state, **kwargs)

fig = plot_square_as_board(temp_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title="Example Log Probs")

# # %%
# if MAIN:
#     plot_single_board(int_to_label(moves_int))
# # %%
# if MAIN:
#     num_games = 50
#     focus_games_int = board_seqs_int[:num_games] # shape: [50, 60] = [50 games, 60 moves each]
#     focus_games_string = board_seqs_string[:num_games]
# # %%
# def one_hot(list_of_ints, num_classes=64):
#     out = t.zeros((num_classes,), dtype=t.float32)
#     out[list_of_ints] = 1.
#     return out


# if MAIN:
#     focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
#     focus_valid_moves = t.zeros((num_games, 60, 64), dtype=t.float32)

#     for i in (range(num_games)):
#         board = OthelloBoardState()
#         for j in range(60):
#             board.umpire(focus_games_string[i, j].item())
#             focus_states[i, j] = board.state
#             focus_valid_moves[i, j] = one_hot(board.get_valid_moves())

#     print("focus states:", focus_states.shape)
#     print("focus_valid_moves", tuple(focus_valid_moves.shape))
# # %%
# if MAIN:
#     imshow(
#         focus_states[0, :16],
#         facet_col=0,
#         facet_col_wrap=8,
#         facet_labels=[f"Move {i}" for i in range(1, 17)],
#         title="First 16 moves of first game",
#         color_continuous_scale="Greys",
#     )
# %%
