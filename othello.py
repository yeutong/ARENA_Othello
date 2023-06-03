# %%
%pip install git+https://github.com/neelnanda-io/neel-plotly

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

    if not OTHELLO_ROOT.exists():
        !git clone https://github.com/likenneth/othello_world

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
    imshow(state, **kwargs)


if MAIN:
    plot_square_as_board(temp_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title="Example Log Probs")
# %%
if MAIN:
    plot_single_board(int_to_label(moves_int))
# %%
if MAIN:
    num_games = 50
    focus_games_int = board_seqs_int[:num_games] # shape: [50, 60] = [50 games, 60 moves each]
    focus_games_string = board_seqs_string[:num_games]
# %%
def one_hot(list_of_ints, num_classes=64):
    out = t.zeros((num_classes,), dtype=t.float32)
    out[list_of_ints] = 1.
    return out


if MAIN:
    focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
    focus_valid_moves = t.zeros((num_games, 60, 64), dtype=t.float32)

    for i in (range(num_games)):
        board = OthelloBoardState()
        for j in range(60):
            board.umpire(focus_games_string[i, j].item())
            focus_states[i, j] = board.state
            focus_valid_moves[i, j] = one_hot(board.get_valid_moves())

    print("focus states:", focus_states.shape)
    print("focus_valid_moves", tuple(focus_valid_moves.shape))
# %%
if MAIN:
    imshow(
        focus_states[0, :16],
        facet_col=0,
        facet_col_wrap=8,
        facet_labels=[f"Move {i}" for i in range(1, 17)],
        title="First 16 moves of first game",
        color_continuous_scale="Greys",
    )
# %%
if MAIN:
    focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))
    focus_logits.shape # torch.Size([50 games, 59 moves, 61 tokens]) 
# %%
if MAIN:
    full_linear_probe = t.load(OTHELLO_MECHINT_ROOT / "main_linear_probe.pth")

    rows = 8
    cols = 8 
    options = 3
    assert full_linear_probe.shape == (3, cfg.d_model, rows, cols, options)
# %%
if MAIN:
    black_to_play_index = 0
    white_to_play_index = 1
    blank_index = 0
    their_index = 1
    my_index = 2

    # Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")
    linear_probe = t.zeros(cfg.d_model, rows, cols, options, device=device)
    linear_probe[..., blank_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 0] + full_linear_probe[white_to_play_index, ..., 0])
    linear_probe[..., their_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 1] + full_linear_probe[white_to_play_index, ..., 2])
    linear_probe[..., my_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 2] + full_linear_probe[white_to_play_index, ..., 1])
# %%
if MAIN:
    layer = 6
    game_index = 0
    move = 29

def plot_probe_outputs(layer, game_index, move, **kwargs):
    residual_stream = focus_cache["resid_post", layer][game_index, move]
    # print("residual_stream", residual_stream.shape)
    probe_out = einops.einsum(residual_stream, linear_probe, "d_model, d_model row col options -> row col options")
    probabilities = probe_out.softmax(dim=-1)
    plot_square_as_board(probabilities, facet_col=2, facet_labels=["P(Empty)", "P(Their's)", "P(Mine)"], **kwargs)

if MAIN:
    plot_probe_outputs(layer, game_index, move, title="Example probe outputs after move 29 (black to play)")

    plot_single_board(int_to_label(focus_games_int[game_index, :move+1]))
# %%
if MAIN:
    layer = 4
    game_index = 0
    move = 29

    plot_probe_outputs(layer, game_index, move, title="Example probe outputs at layer 4 after move 29 (black to play)")
    plot_single_board(int_to_label(focus_games_int[game_index, :move+1]))
# %%
if MAIN:
    layer = 4
    game_index = 0
    move = 30

    plot_probe_outputs(layer, game_index, move, title="Example probe outputs at layer 4 after move 30 (white to play)")

    plot_single_board(focus_games_string[game_index, :31])
# %%
def state_stack_to_one_hot(state_stack):
    '''
    Creates a tensor of shape (games, moves, rows=8, cols=8, options=3), where the [g, m, r, c, :]-th entry
    is a one-hot encoded vector for the state of game g at move m, at row r and column c. In other words, this
    vector equals (1, 0, 0) when the state is empty, (0, 1, 0) when the state is "their", and (0, 0, 1) when the
    state is "my".
    '''
    one_hot = t.zeros(
        state_stack.shape[0], # num games
        state_stack.shape[1], # num moves
        rows,
        cols,
        3, # the options: empty, white, or black
        device=state_stack.device,
        dtype=t.int,
    )
    one_hot[..., 0] = state_stack == 0 
    one_hot[..., 1] = state_stack == -1 
    one_hot[..., 2] = state_stack == 1 

    return one_hot

# We first convert the board states to be in terms of my (+1) and their (-1), rather than black and white

if MAIN:
    alternating = np.array([-1 if i%2 == 0 else 1 for i in range(focus_games_int.shape[1])])
    flipped_focus_states = focus_states * alternating[None, :, None, None]

    # We now convert to one-hot encoded vectors
    focus_states_flipped_one_hot = state_stack_to_one_hot(t.tensor(flipped_focus_states))

    # Take the argmax (i.e. the index of option empty/their/mine)
    focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)
# %%
if MAIN:
    probe_out = einops.einsum(
        focus_cache["resid_post", 6], linear_probe,
        "game move d_model, d_model row col options -> game move row col options"
    )

    probe_out_value = probe_out.argmax(dim=-1)
# %%
if MAIN:
    correct_middle_odd_answers = (probe_out_value.cpu() == focus_states_flipped_value[:, :-1])[:, 5:-5:2]
    accuracies_odd = einops.reduce(correct_middle_odd_answers.float(), "game move row col -> row col", "mean")

    correct_middle_answers = (probe_out_value.cpu() == focus_states_flipped_value[:, :-1])[:, 5:-5]
    accuracies = einops.reduce(correct_middle_answers.float(), "game move row col -> row col", "mean")

    plot_square_as_board(
        1 - t.stack([accuracies_odd, accuracies], dim=0),
        title="Average Error Rate of Linear Probe", 
        facet_col=0, facet_labels=["Black to Play moves", "All Moves"], 
        zmax=0.25, zmin=-0.25
    )
# %%
if MAIN:
    blank_probe = linear_probe[..., 0] - (linear_probe[..., 1] + linear_probe[..., 2]) / 2
    my_probe = linear_probe[..., 2] - linear_probe[..., 1]
# %%
if MAIN:
    pos = 20
    game_index = 0

    # Plot board state
    moves = focus_games_string[game_index, :pos+1]
    plot_single_board(moves)

    # Plot corresponding model predictions
    state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
    state.flatten()[stoi_indices] = focus_logits[game_index, pos].log_softmax(dim=-1)[1:]
    plot_square_as_board(state, zmax=0, diverging_scale=False, title="Log probs")
# %%
if MAIN:
    cell_r = 5
    cell_c = 4
    print(f"Flipping the color of cell {'ABCDEFGH'[cell_r]}{cell_c}")

    board = OthelloBoardState()
    board.update(moves.tolist())
    board_state = board.state.copy()
    valid_moves = board.get_valid_moves()
    flipped_board = copy.deepcopy(board)
    flipped_board.state[cell_r, cell_c] *= -1
    flipped_valid_moves = flipped_board.get_valid_moves()

    newly_legal = [string_to_label(move) for move in flipped_valid_moves if move not in valid_moves]
    newly_illegal = [string_to_label(move) for move in valid_moves if move not in flipped_valid_moves]
    print("newly_legal", newly_legal)
    print("newly_illegal", newly_illegal)
# %%

def apply_scale(resid: Float[Tensor, "batch=1 seq d_model"], flip_dir: Float[Tensor, "d_model"], scale: int, pos: int):
    '''
    Returns a version of the residual stream, modified by the amount `scale` in the 
    direction `flip_dir` at the sequence position `pos`, in the way described above.
    '''
    norm_flip_dir = flip_dir/flip_dir.norm()
    alpha = resid[0, pos].dot(norm_flip_dir)
    resid[0, pos] -= (scale + 1)*alpha*norm_flip_dir
    return resid

# %%

if MAIN:
    flip_dir = my_probe[:, cell_r, cell_c]

    big_flipped_states_list = []
    layer = 4
    scales = [0, 1, 2, 4, 8, 16]

    # Iterate through scales, generate a new facet plot for each possible scale
    for scale in scales:

        # Hook function which will perform flipping in the "F4 flip direction"
        def flip_hook(resid: Float[Tensor, "batch=1 seq d_model"], hook: HookPoint):
            return apply_scale(resid, flip_dir, scale, pos)

        # Calculate the logits for the board state, with the `flip_hook` intervention
        # (note that we only need to use :pos+1 as input, because of causal attention)
        flipped_logits: Tensor = model.run_with_hooks(
            focus_games_int[game_index:game_index+1, :pos+1],
            fwd_hooks=[
                (utils.get_act_name("resid_post", layer), flip_hook),
            ]
        ).log_softmax(dim=-1)[0, pos]

        flip_state = t.zeros((64,), dtype=t.float32, device=device) - 10.
        flip_state[stoi_indices] = flipped_logits[1:]
        big_flipped_states_list.append(flip_state)


if MAIN:
    flip_state_big = t.stack(big_flipped_states_list)
    state_big = einops.repeat(state.flatten(), "d -> b d", b=6)
    color = t.zeros((len(scales), 64)).cuda() + 0.2
    for s in newly_legal:
        color[:, to_string(s)] = 1
    for s in newly_illegal:
        color[:, to_string(s)] = -1

    scatter(
        y=state_big, 
        x=flip_state_big, 
        title=f"Original vs Flipped {string_to_label(8*cell_r+cell_c)} at Layer {layer}", 
        # labels={"x": "Flipped", "y": "Original"}, 
        xaxis="Flipped", 
        yaxis="Original", 

        hover=[f"{r}{c}" for r in "ABCDEFGH" for c in range(8)], 
        facet_col=0, facet_labels=[f"Translate by {i}x" for i in scales], 
        color=color, color_name="Newly Legal", color_continuous_scale="Geyser"
    )
# %%

if MAIN:
    game_index = 1
    move = 20
    layer = 6

    plot_single_board(focus_games_string[game_index, :move+1])
    plot_probe_outputs(layer, game_index, move)

# %%

def plot_contributions(contributions, component: str):
    imshow(
        contributions,
        facet_col=0,
        y=list("ABCDEFGH"),
        facet_labels=[f"Layer {i}" for i in range(7)],
        title=f"{component} Layer Contributions to my vs their (Game {game_index} Move {move})",
        aspect="equal",
        width=1400,
        height=350
    )

def calculate_attn_and_mlp_probe_score_contributions(
    focus_cache: ActivationCache, 
    my_probe: Float[Tensor, "d_model rows cols"],
    layer: int,
    game_index: int, 
    move: int
) -> Tuple[Float[Tensor, "layers rows cols"], Float[Tensor, "layers rows cols"]]:
    attn_contr_all = []
    mlp_contr_all = []
    for l in range(layer + 1):
        attn_out = focus_cache['attn_out', l]
        mlp_out = focus_cache['mlp_out', l]
        
        attn_contr = einops.einsum(attn_out[game_index, move], my_probe, 'd, d r c -> r c')
        mlp_contr = einops.einsum(mlp_out[game_index, move], my_probe, 'd, d r c -> r c')

        attn_contr_all.append(attn_contr)
        mlp_contr_all.append(mlp_contr)
    return (t.stack(attn_contr_all), t.stack(mlp_contr_all))


    


if MAIN:
    attn_contributions, mlp_contributions = calculate_attn_and_mlp_probe_score_contributions(focus_cache, my_probe, layer, game_index, move)

    plot_contributions(attn_contributions, "Attention")
    plot_contributions(mlp_contributions, "MLP")

# %%
if MAIN:
    attn_contributions, mlp_contributions = calculate_attn_and_mlp_probe_score_contributions(focus_cache, blank_probe, layer, game_index, move)

    plot_contributions(attn_contributions, "Attention")
    plot_contributions(mlp_contributions, "MLP")
# %%
# Scale the probes down to be unit norm per cell

if MAIN:
    blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
    my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
    # Set the center blank probes to 0, since they're never blank so the probe is meaningless
    blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.
# %%
def get_w_in(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    '''
    Returns the input weights for the neuron in the list, at each square on the board.

    If normalize is True, the weights are normalized to unit norm.
    '''
    neuron = model.W_in[layer, :, neuron] # shape: [d_model]
    if normalize:
        return neuron / neuron.norm()
    return neuron

def get_w_out(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    '''
    Returns the input weights for the neuron in the list, at each square on the board.
    '''
    neuron = model.W_out[layer, neuron, :] # shape: [d_model]
    if normalize:
        return neuron / neuron.norm()
    return neuron

def calculate_neuron_input_weights(
    model: HookedTransformer, 
    probe: Float[Tensor, "d_model row col"], 
    layer: int, 
    neuron: int
) -> Float[Tensor, "rows cols"]:
    '''
    Returns tensor of the input weights for each neuron in the list, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    '''
    neuron = get_w_in(model, layer, neuron, normalize=True)
    return einops.einsum(neuron, probe, 'd_model, d_model row col -> row col')


def calculate_neuron_output_weights(
    model: HookedTransformer, 
    probe: Float[Tensor, "d_model row col"], 
    layer: int, 
    neuron: int
) -> Float[Tensor, "rows cols"]:
    '''
    Returns tensor of the output weights for each neuron in the list, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    '''
    neuron = get_w_out(model, layer, neuron, normalize=True)
    return einops.einsum(neuron, probe, 'd_model, d_model row col -> row col')

def neuron_output_weight_map_to_unemb(
    model: HookedTransformer,
    layer: int,
    neuron: int
) -> Float[Tensor, 'rows cols']:
    neuron = get_w_out(model, layer, neuron, normalize=True)
    W_U_norm = model.W_U / model.W_U.norm(dim=0, keepdim=True)
    out = einops.einsum(neuron, W_U_norm, 'd_model, d_model n_vocab -> n_vocab')
    board = t.zeros(8, 8).to(model.cfg.device)
    board.flatten()[stoi_indices] = out[1:]
    return board
# %%
def neuron_and_blank_my_emb(layer, neuron):

    w_in_L5N1393_blank = calculate_neuron_input_weights(model, blank_probe_normalised, layer, neuron)
    w_in_L5N1393_my = calculate_neuron_input_weights(model, my_probe_normalised, layer, neuron)

    imshow(
        t.stack([w_in_L5N1393_blank, w_in_L5N1393_my]),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
        facet_labels=["Blank In", "My In"],
        width=750,
    )

    w_out_L5N1393_blank = calculate_neuron_output_weights(model, blank_probe_normalised, layer, neuron)
    w_out_L5N1393_my = calculate_neuron_output_weights(model, my_probe_normalised, layer, neuron)
    w_out_L5N1393_unemb = neuron_output_weight_map_to_unemb(model, layer, neuron)

    imshow(
        t.stack([w_out_L5N1393_blank, w_out_L5N1393_my, w_out_L5N1393_unemb]),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Output weights in terms of the probe for neuron L{layer}N{neuron}",
        facet_labels=["Blank Out", "My Out", 'Unemb'],
        width=1100,
    )

if MAIN:
    layer = 5
    neuron = 1393
    neuron_and_blank_my_emb(layer, neuron)

# %%
"""
Research Question: How the model decide "C0" is a valid move?
Method: Find neurons that write to W_U, and see whether they are looking for some pattern according to blank and my probe.
1. For each neuron, calculate the cosine similarity between 
    - the input weights and blank probe
    - the input weights and my probe
    - the output weights and W_U
2. Sort by the cosine similarity between the output weights and W_U, take the top 5 neurons
    - here we can have different metric to sort the neurons
3. using `neuron_and_blank_my_emb` to visualize the input and output weights for these neurons
"""

def possible_based_cell(target_cell: Tuple[int, int]):
    """
    return a list of cells that are possible based on the target cell
    """
    tr, tc = target_cell
    possible_cells = []

    for r in range(8):
        for c in range(8):
            if abs(tr - r) == abs(tc - c) > 1:
                possible_cells.append((r, c))
            if (abs(tr - r) == 0  and abs(tc - c) > 1):
                possible_cells.append((r, c))
            if (abs(tr - r) > 1  and abs(tc - c) == 0):
                possible_cells.append((r, c))
    return possible_cells

# %%
def gen_detector_vec(target_cell: Tuple[int, int], util_cell: Tuple[int, int]):
    "return 8 * 8 map, 1 for based_cell and -1 for cells between based_cell and target_cell, 0 for other cells"
    tr, tc = target_cell
    ur, uc = util_cell
    detector_vec = t.zeros(8, 8)
    detector_vec[ur, uc] = 1

    r_dir = 1 if ur > tr else -1
    r_dir = 0 if ur == tr else r_dir
    c_dir = 1 if uc > tc else -1
    c_dir = 0 if uc == tc else c_dir

    r, c = tr + r_dir, tc + c_dir
    while (r, c) != (ur, uc):
        detector_vec[r, c] = -1
        r += r_dir
        c += c_dir

    return detector_vec

def score_read_blank(
    w_in_blank: Float[Tensor, 'd_mlp row col'],
    target_cell: Tuple[int, int],
):
    row, col = target_cell
    w_in_blank_norm = w_in_blank / w_in_blank.norm(dim=(1, 2), keepdim=True)
    return w_in_blank_norm[:, row, col]

def score_write_unemb(
    w_out_unemb: Float[Tensor, 'd_mlp n_vocab'],
    target_cell: Tuple[int, int],
):
    row, col = target_cell
    cell_string = row * 8 + col
    cell_label = string_to_label(cell_string)
    cell_int = to_int(cell_label)

    w_out_unemb_norm = w_out_unemb / w_out_unemb.norm(dim=-1, keepdim=True)
    return w_out_unemb_norm[:, cell_int]

def score_read_my(
    w_in_my: Float[Tensor, 'd_mlp row col'],
    target_cell: Tuple[int, int],
    util_cell: Tuple[int, int],
):
    row, col = target_cell
    ur, uc = util_cell
    detector_vec = gen_detector_vec(target_cell, util_cell)
    w_in_my_norm = w_in_my / w_in_my.norm(dim=(1, 2), keepdim=True)
    return einops.einsum(w_in_my_norm, detector_vec, 'd_mlp row col, row col -> d_mlp')

# %%
# calculate the cosine similarity between the output weights and W_U for each neuron in specific layer
layer = 5
cell_label = 'E5'
cell = (ord(cell_label[0]) - ord('A'), int(cell_label[-1])) # row and column of the cell

w_in_L5 = model.W_in[layer, :, :] # shape: [d_model, d_mlp]
w_in_L5_norm = w_in_L5 / w_in_L5.norm(dim=0, keepdim=True)

w_out_L5 = model.W_out[layer, :, :] # shape: [d_mlp, d_model]
w_out_L5_norm = w_out_L5 / w_out_L5.norm(dim=-1, keepdim=True)

w_U_norm = model.W_U / model.W_U.norm(dim=0, keepdim=True) # shape: [d_model, n_vocab]

w_in_L5_my = einops.einsum(w_in_L5_norm, my_probe_normalised, 'd_model d_mlp, d_model row col -> d_mlp row col')
w_in_L5_blank = einops.einsum(w_in_L5_norm, blank_probe_normalised, 'd_model d_mlp, d_model row col -> d_mlp row col')[:, row, col]
w_out_L5_umemb = einops.einsum(w_out_L5_norm, w_U_norm, 'd_mlp d_model, d_model n_vocab -> d_mlp n_vocab')[:, to_int(cell)]

# select the top 5 neurons that have the highest cosine similarity between the output weights and W_U
top_neurons = (w_out_L5_umemb + w_in_L5_blank).argsort(descending=True)[:20]

# visualize the input and output weights for these neurons
for neuron in top_neurons:
    neuron_and_blank_my_emb(layer, neuron)

# 

# %%
if MAIN:
    w_in_L5N1393 = get_w_in(model, layer, neuron, normalize=True)
    w_out_L5N1393 = get_w_out(model, layer, neuron, normalize=True)

    U, S, Vh = t.svd(t.cat([
        my_probe.reshape(cfg.d_model, 64),
        blank_probe.reshape(cfg.d_model, 64)
    ], dim=1))

    U_unemb, _, _ = t.svd(
        model.W_U[:, :]
    )

    # Remove the final four dimensions of U, as the 4 center cells are never blank and so the blank probe is meaningless there
    probe_space_basis = U[:, :-4]

    print("Fraction of input weights in probe basis:", (w_in_L5N1393 @ probe_space_basis).norm().item()**2)
    print("Fraction of output weights in probe basis:", (w_out_L5N1393 @ probe_space_basis).norm().item()**2)
    print("Fraction of output weights in unemb basis:", (w_out_L5N1393 @ U_unemb).norm().item()**2)

# %%

def kurtosis(tensor: Tensor, reduced_axes, fisher=True):
    '''
    Computes the kurtosis of a tensor over specified dimensions.
    '''
    return (((tensor - tensor.mean(dim=reduced_axes, keepdim=True)) / tensor.std(dim=reduced_axes, keepdim=True))**4).mean(dim=reduced_axes, keepdim=False) - fisher*3




if MAIN:
    unembed_normalized = model.W_U / model.W_U.norm(dim=0)
    square_unembed = t.zeros(model.cfg.d_model, 8, 8).to(model.cfg.device)
    square_unembed.flatten(start_dim=1)[:, stoi_indices] = unembed_normalized[:, 1:]

    layer = 3
    top_layer_3_neurons = focus_cache["post", layer][:, 3:-3].std(dim=[0, 1]).argsort(descending=True)[:10]
    # top_layer_3_neurons = einops.reduce(focus_cache["post", layer][:, 3:-3], "game move neuron -> neuron", reduction=kurtosis).argsort(descending=True)[:10]

    heatmaps_blank = []
    heatmaps_my = []
    heatmaps_U = []

    for neuron in top_layer_3_neurons:
        neuron = neuron.item()
        # heatmaps_blank.append(calculate_neuron_output_weights(model, blank_probe_normalised, layer, neuron))
        # heatmaps_my.append(calculate_neuron_output_weights(model, my_probe_normalised, layer, neuron))
        # heatmaps_U.append(calculate_neuron_output_weights(model, square_unembed, layer, neuron))

        heatmaps_blank.append(calculate_neuron_input_weights(model, blank_probe_normalised, layer, neuron))
        heatmaps_my.append(calculate_neuron_input_weights(model, my_probe_normalised, layer, neuron))
        heatmaps_U.append(calculate_neuron_input_weights(model, square_unembed, layer, neuron))
        
    imshow(
        t.stack(heatmaps_blank),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Cosine sim of Output weights and the 'blank color' probe for top layer {layer} neurons",
        facet_labels=[f"L3N{n.item()}" for n in top_layer_3_neurons],
        width=1600, height=300,
    )

    imshow(
        t.stack(heatmaps_my),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Cosine sim of Output weights and the 'my color' probe for top layer {layer} neurons",
        facet_labels=[f"L3N{n.item()}" for n in top_layer_3_neurons],
        width=1600, height=300,
    )

    imshow(
        t.stack(heatmaps_U),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Cosine sim of Output weights and Unembed for top layer {layer} neurons",
        facet_labels=[f"L3N{n.item()}" for n in top_layer_3_neurons],
        width=1600, height=300,
    )
# %%
if MAIN:
    game_index = 4
    move = 20

    # plot_single_board(focus_games_string[game_index, :move+1], title="Original Game (black plays E0)")
    # plot_single_board(focus_games_string[game_index, :move].tolist()+[16], title="Corrupted Game (blank plays C0)")

    plot_single_board(focus_games_string[game_index, :move+1], title="Original Game (black plays E0)")
    plot_single_board(focus_games_string[game_index, :move-1].tolist()+
                      [to_string('G7'), focus_games_string[game_index, move+1].item()], title="Corrupted Game (blank plays C0)")

# %%

if MAIN:
    clean_input = focus_games_int[game_index, :move+1].clone()
    corrupted_input = focus_games_int[game_index, :move+1].clone()
    corrupted_input[-1] = to_int("C0")
    print("Clean:     ", ", ".join(int_to_label(corrupted_input)))
    print("Corrupted: ", ", ".join(int_to_label(clean_input)))

if MAIN:
    clean_logits, clean_cache = model.run_with_cache(clean_input)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_input)

    clean_log_probs = clean_logits.log_softmax(dim=-1)
    corrupted_log_probs = corrupted_logits.log_softmax(dim=-1)
# %%

if MAIN:
    f0_index = to_int("F0")
    clean_f0_log_prob = clean_log_probs[0, -1, f0_index]
    corrupted_f0_log_prob = corrupted_log_probs[0, -1, f0_index]

    print("Clean log prob", clean_f0_log_prob.item())
    print("Corrupted log prob", corrupted_f0_log_prob.item(), "\n")

def patching_metric(patched_logits: Float[Tensor, "batch=1 seq=21 d_vocab=61"]):
    '''
    Function of patched logits, calibrated so that it equals 0 when performance is 
    same as on corrupted input, and 1 when performance is same as on clean input.

    Should be linear function of the logits for the F0 token at the final move.
    '''
    patch_f0_log_prob = patched_logits.log_softmax(-1)[0, -1, f0_index]
    return  ((corrupted_f0_log_prob - patch_f0_log_prob) / (corrupted_f0_log_prob - clean_f0_log_prob))


# %%

def patch_final_move_output(
    activation: Float[Tensor, "batch seq d_model"], 
    hook: HookPoint,
    clean_cache: ActivationCache,
) -> Float[Tensor, "batch seq d_model"]:
    '''
    Hook function which patches activations at the final sequence position.

    Note, we only need to patch in the final sequence position, because the
    prior moves in the clean and corrupted input are identical (and this is
    an autoregressive model).
    '''
    activation[:, -1] = clean_cache[hook.name][:, -1]
    return activation

def get_act_patch_resid_pre(
    model: HookedTransformer, 
    corrupted_input: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[Tensor, "batch seq d_model"]], Float[Tensor, ""]]
) -> Float[Tensor, "2 d_model"]:
    '''
    Returns an array of results, corresponding to the results of patching at
    each (attn_out, mlp_out) for all layers in the model.
    '''
    model.reset_hooks()
    patch_results = []

    for layer in range(model.cfg.n_layers):
        patch_hook = partial(patch_final_move_output, clean_cache=clean_cache)
        patch_logits_attn = model.run_with_hooks(corrupted_input, fwd_hooks=[(
            utils.get_act_name('attn_out', layer), patch_hook
        )])
        patch_logits_mlp = model.run_with_hooks(corrupted_input, fwd_hooks=[(
            utils.get_act_name('mlp_out', layer), patch_hook
        )])
        patch_results.append((patching_metric(patch_logits_attn).item(),
                              patching_metric(patch_logits_mlp).item()))
    return np.array(patch_results).T

if MAIN:
    patching_results = get_act_patch_resid_pre(model, corrupted_input, clean_cache, patching_metric)

    line(patching_results, title="Layer Output Patching Effect on F0 Log Prob", line_labels=["attn", "mlp"], width=750)

# %%

if MAIN:
    layer = 5
    neuron = 1393

    w_out = get_w_out(model, layer, neuron, normalize=False)
    state = t.zeros(8, 8, device=device)
    state.flatten()[stoi_indices] = w_out @ model.W_U[:, 1:]
    plot_square_as_board(state, title=f"Output weights of Neuron {layer}N{neuron} in the output logit basis", width=600)

# %%

print(t.cosine_similarity(model.W_U[:, to_int('C0')], model.W_U[:, to_int('D1')], dim=0))

# %%

w_out = get_w_out(model, layer, neuron, normalize=True)
U, S, Vh = t.svd(model.W_U[:, 1:])
print("Fraction of variance captured by W_U", (w_out @ U).norm().item()**2)
# %%

if MAIN:
    neuron_acts = focus_cache["post", layer, "mlp"][:, :, neuron]

    imshow(
        neuron_acts,
        title=f"{layer}N{neuron} Activations over 50 games",
        labels={"x": "Move", "y": "Game"},
        aspect="auto",
        width=900
    )
# %%

if MAIN:
    top_moves = neuron_acts > neuron_acts.quantile(0.95)

    focus_states_flipped_value = focus_states_flipped_value.to(device)
    board_state_at_top_moves = t.stack([
        (focus_states_flipped_value == 2)[:, :-1][top_moves].float().mean(0),
        (focus_states_flipped_value == 1)[:, :-1][top_moves].float().mean(0),
        (focus_states_flipped_value == 0)[:, :-1][top_moves].float().mean(0)
    ])

    plot_square_as_board(
        board_state_at_top_moves, 
        facet_col=0,
        facet_labels=["Mine", "Theirs", "Blank"],
        title=f"Aggregated top 30 moves for neuron {layer}N{neuron}", 
    )
# %%

if MAIN:
    focus_states_flipped_pm1 = t.zeros_like(focus_states_flipped_value, device=device)
    focus_states_flipped_pm1[focus_states_flipped_value==2] = 1.
    focus_states_flipped_pm1[focus_states_flipped_value==1] = -1.

    board_state_at_top_moves = focus_states_flipped_pm1[:, :-1][top_moves].float().mean(0)

    plot_square_as_board(
        board_state_at_top_moves, 
        title=f"Aggregated top 30 moves for neuron {layer}N{neuron} (1 = theirs, -1 = mine)",
    )
# %%

unembed_normalized = model.W_U / model.W_U.norm(dim=0)
square_unembed = t.zeros(model.cfg.d_model, 8, 8).to(model.cfg.device)
square_unembed.flatten(start_dim=1)[:, stoi_indices] = unembed_normalized[:, 1:]

layer = 5
top_neurons = focus_cache["post", layer][:, 3:-3].std(dim=[0, 1]).argsort(descending=True)[:10]
# top_neurons = einops.reduce(focus_cache["post", layer][:, 3:-3], "game move neuron -> neuron", reduction=kurtosis).argsort(descending=True)[:10]
output_weights_list = []
board_states_list = []


for neuron in top_neurons:
    output_weights_list.append(calculate_neuron_input_weights(model, square_unembed, layer, neuron))

    neuron_acts = focus_cache["post", layer][..., neuron]
    top_moves_neuron = neuron_acts > neuron_acts.quantile(0.95)
    board_states_list.append(focus_states_flipped_pm1[:, :-1][top_moves_neuron].float().mean(0))

output_weights_in_logit_basis = t.stack(output_weights_list)
board_states = t.stack(board_states_list)


if MAIN:
    # Your code here - investigate the top 10 neurons by std dev of activations, see what you can find!
    plot_square_as_board(
        output_weights_in_logit_basis, 
        title=f"Output weights of top 10 neurons in layer 5, in the output logit basis",
        facet_col=0, 
        facet_labels=[f"L5N{n.item()}" for n in top_neurons],
        width=1500
    )
    plot_square_as_board(
        board_states, 
        title=f"Aggregated top 30 moves for each top 10 neuron in layer 5", 
        facet_col=0, 
        facet_labels=[f"L5N{n.item()}" for n in top_neurons],
        width=1500
    )
# %%

if MAIN:
    c0 = focus_states_flipped_pm1[:, :, 2, 0]
    d1 = focus_states_flipped_pm1[:, :, 3, 1]
    e2 = focus_states_flipped_pm1[:, :, 4, 2]

    label = (c0==0) & (d1==-1) & (e2==1)

    neuron_acts = focus_cache["post", 5][:, :, 1393]

def make_spectrum_plot(
    neuron_acts: Float[Tensor, "batch"],
    label: Bool[Tensor, "batch"],
    **kwargs
) -> None:
    '''
    Generates a spectrum plot from the neuron activations and a set of labels.
    '''
    px.histogram(
        pd.DataFrame({"acts": neuron_acts.tolist(), "label": label.tolist()}), 
        x="acts", color="label", histnorm="percent", barmode="group", nbins=100, 
        title="Spectrum plot for neuron L5N1393 testing C0==BLANK & D1==THEIRS & E2==MINE",
        color_discrete_sequence=px.colors.qualitative.Bold
    ).show()


if MAIN:
    make_spectrum_plot(neuron_acts.flatten(), label[:, :-1].flatten())
# %%
true_pos = (neuron_acts > 0) & (label[:, :-1] == True)
false_neg = (neuron_acts < 0) & (label[:, :-1] == True)

false_neg_states = focus_states_flipped_pm1[:, :-1][false_neg]
true_pos_states = focus_states_flipped_pm1[:, :-1][true_pos]

false_neg_states_reshape = false_neg_states[:20].reshape(5, 4, 8, 8)
# imshow(false_neg_states_reshape, animation_frame=0, facet_col=1)

imshow(utils.to_numpy(t.stack([false_neg_states.float().mean(0), 
                               true_pos_states.float().mean(0)])),
                               facet_col=0)
imshow(false_neg_states.float().std(0))


# %%
imshow(
    focus_states[0, :16],
    facet_col=0,
    facet_col_wrap=8,
    facet_labels=[f"Move {i}" for i in range(1, 17)],
    title="First 16 moves of first game",
    color_continuous_scale="Greys",
)
