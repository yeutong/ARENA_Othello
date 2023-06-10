
# %pip install git+https://github.com/neelnanda-io/neel-plotly




from neel_plotly import scatter, line
from plotly_utils import imshow
import pandas as pd
from rich import print as rprint
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from dataclasses import dataclass
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformer_lens.hook_points import HookedRootModule, HookPoint
import transformer_lens.utils as utils
import transformer_lens
from IPython.display import HTML
import datasets
import dataclasses
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import copy
from functools import partial
import typeguard
from typing import List, Union, Optional, Tuple, Callable, Dict
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from IPython.display import display
import random
import itertools
from pathlib import Path
import plotly.express as px
from ipywidgets import interact
import wandb
import einops
import numpy as np
from torch.utils.data import DataLoader
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch as t
import sys
import os
import random

# os.environ["ACCELERATE_DISABLE_RICH"] = "1"



ENABLE_STREAMLIT = True # allow the script to be run in streamlit

if ENABLE_STREAMLIT:
    from streamlit.runtime.scriptrunner import get_script_run_ctx # this will affect the plotly figure
    import streamlit as st
    from st_pages import show_pages_from_config, add_page_title

    if not get_script_run_ctx():
        in_streamlit = False # whether the script is run in streamlit
    else:
        in_streamlit = True
        st.set_page_config(layout="wide")
        show_pages_from_config()
else:
    in_streamlit = False 

# Make sure exercises are in the path

working_dir = Path(f"{os.getcwd()}").resolve()

t.set_grad_enabled(False)




# if not OTHELLO_ROOT.exists():
#     !git clone https://github.com/likenneth/othello_world

if OTHELLO_MECHINT_ROOT not in sys.path:
    sys.path.append(str(OTHELLO_MECHINT_ROOT))

from mech_interp_othello_utils import plot_board, plot_single_board, plot_board_log_probs, to_string, to_int, int_to_label, string_to_label, OthelloBoardState




















# Get our list of board labels


# board_labels = list(map(to_board_label, stoi_indices))
# moves_int = board_seqs_int[0, :30]

# This is implicitly converted to a batch of size 1
# logits: Tensor = model(moves_int)



# logit_vec = logits[0, -1]
# log_probs = logit_vec.log_softmax(-1)
# Remove the "pass" move (the zeroth vocab item)
# log_probs = log_probs[1:]
# assert len(log_probs) == 60

# Set all cells to -13 by default, for a very negative log prob - this means the middle cells don't show up as mattering
# temp_board_state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
# temp_board_state.flatten()[stoi_indices] = log_probs















    # focus_logits.shape # torch.Size([50 games, 59 moves, 61 tokens])



layer = 6
game_index = 0
move = 29


# def plot_probe_outputs(layer, game_index, move, **kwargs):
#     residual_stream = focus_cache["resid_post", layer][game_index, move]
#     # print("residual_stream", residual_stream.shape)
#     probe_out = einops.einsum(residual_stream, linear_probe,
#                               "d_model, d_model row col options -> row col options")
#     probabilities = probe_out.softmax(dim=-1)
#     plot_square_as_board(probabilities, facet_col=2, facet_labels=[
#                          "P(Empty)", "P(Their's)", "P(Mine)"], **kwargs)



def state_stack_to_one_hot(state_stack):
    '''
    Creates a tensor of shape (games, moves, rows=8, cols=8, options=3), where the [g, m, r, c, :]-th entry
    is a one-hot encoded vector for the state of game g at move m, at row r and column c. In other words, this
    vector equals (1, 0, 0) when the state is empty, (0, 1, 0) when the state is "their", and (0, 0, 1) when the
    state is "my".
    '''
    one_hot = t.zeros(
        state_stack.shape[0],  # num games
        state_stack.shape[1],  # num moves
        rows,
        cols,
        3,  # the options: empty, white, or black
        device=state_stack.device,
        dtype=t.int,
    )
    one_hot[..., 0] = state_stack == 0
    one_hot[..., 1] = state_stack == -1
    one_hot[..., 2] = state_stack == 1

    return one_hot

# We first convert the board states to be in terms of my (+1) and their (-1), rather than black and white


alternating = np.array(
    [-1 if i % 2 == 0 else 1 for i in range(focus_games_int.shape[1])])
flipped_focus_states = focus_states * alternating[None, :, None, None]

# We now convert to one-hot encoded vectors
focus_states_flipped_one_hot = state_stack_to_one_hot(
    t.tensor(flipped_focus_states))

# Take the argmax (i.e. the index of option empty/their/mine)
focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)


probe_out = einops.einsum(
    focus_cache["resid_post", 6], linear_probe,
    "game move d_model, d_model row col options -> game move row col options"
)

probe_out_value = probe_out.argmax(dim=-1)

correct_middle_odd_answers = (probe_out_value.cpu(
) == focus_states_flipped_value[:, :-1])[:, 5:-5:2]
accuracies_odd = einops.reduce(
    correct_middle_odd_answers.float(), "game move row col -> row col", "mean")

correct_middle_answers = (probe_out_value.cpu(
) == focus_states_flipped_value[:, :-1])[:, 5:-5]
accuracies = einops.reduce(
    correct_middle_answers.float(), "game move row col -> row col", "mean")

# plot_square_as_board(
#     1 - t.stack([accuracies_odd, accuracies], dim=0),
#     title="Average Error Rate of Linear Probe",
#     facet_col=0, facet_labels=["Black to Play moves", "All Moves"],
#     zmax=0.25, zmin=-0.25
# )

blank_probe = linear_probe[..., 0] - \
    (linear_probe[..., 1] + linear_probe[..., 2]) / 2
my_probe = linear_probe[..., 2] - linear_probe[..., 1]

pos = 20
game_index = 0

# Plot board state
moves = focus_games_string[game_index, :pos+1]
# plot_single_board(moves)

# Plot corresponding model predictions
state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
state.flatten()[stoi_indices] = focus_logits[game_index,
                                                pos].log_softmax(dim=-1)[1:]
# plot_square_as_board(state, zmax=0, diverging_scale=False, title="Log probs")

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

newly_legal = [string_to_label(
    move) for move in flipped_valid_moves if move not in valid_moves]
newly_illegal = [string_to_label(
    move) for move in valid_moves if move not in flipped_valid_moves]
print("newly_legal", newly_legal)
print("newly_illegal", newly_illegal)



def apply_scale(resid: Float[Tensor, "batch=1 seq d_model"], flip_dir: Float[Tensor, "d_model"], scale: int, pos: int):
    '''
    Returns a version of the residual stream, modified by the amount `scale` in the 
    direction `flip_dir` at the sequence position `pos`, in the way described above.
    '''
    norm_flip_dir = flip_dir/flip_dir.norm()
    alpha = resid[0, pos].dot(norm_flip_dir)
    resid[0, pos] -= (scale + 1)*alpha*norm_flip_dir
    return resid





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


flip_state_big = t.stack(big_flipped_states_list)
state_big = einops.repeat(state.flatten(), "d -> b d", b=6)
color = t.zeros((len(scales), 64)).to(device) + 0.2
for s in newly_legal:
    color[:, to_string(s)] = 1
for s in newly_illegal:
    color[:, to_string(s)] = -1


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

        attn_contr = einops.einsum(
            attn_out[game_index, move], my_probe, 'd, d r c -> r c')
        mlp_contr = einops.einsum(
            mlp_out[game_index, move], my_probe, 'd, d r c -> r c')

        attn_contr_all.append(attn_contr)
        mlp_contr_all.append(mlp_contr)
    return (t.stack(attn_contr_all), t.stack(mlp_contr_all))



# Scale the probes down to be unit norm per cell
blank_probe_normalised = blank_probe / \
    blank_probe.norm(dim=0, keepdim=True)
my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
# Set the center blank probes to 0, since they're never blank so the probe is meaningless
blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.



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
    neuron = model.W_in[layer, :, neuron]  # shape: [d_model]
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
    neuron = model.W_out[layer, neuron, :]  # shape: [d_model]
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
    out = einops.einsum(
        neuron, W_U_norm, 'd_model, d_model n_vocab -> n_vocab')
    board = t.zeros(8, 8).to(model.cfg.device)
    board.flatten()[stoi_indices] = out[1:]
    return board



def neuron_and_blank_my_emb(layer, neuron, score=None, sub_score=None, top_detector=None):

    if score is not None:
        score = score[neuron]
        # print(f"Score: {score:2.2%}")
        # st.write(f'Score: {score:2.2%}')
        st.title(f'Neuron L{layer}N{neuron} (Score: {score:2.2%})')

    if sub_score is not None:
        score_read_blank = sub_score['read_blank'][neuron]
        score_read_my = sub_score['read_my'][neuron]
        score_write_unemb = sub_score['write_unemb'][neuron]

        scores_str = f'read blank: {score_read_blank:2.2%}, read my: {score_read_my:2.2%}, write unemb: {score_write_unemb:2.2%}'
        if in_streamlit:
            st.write(scores_str)
        else:
            print(scores_str)

    w_in_L5N1393_blank = calculate_neuron_input_weights(
        model, blank_probe_normalised, layer, neuron)
    w_in_L5N1393_my = calculate_neuron_input_weights(
        model, my_probe_normalised, layer, neuron)

    fig1 = imshow(
        t.stack([w_in_L5N1393_blank, w_in_L5N1393_my, top_detector/3]),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
        facet_labels=["Blank In", "My In", "My In Top Detector"],
        # width=800,
        height=350,
    )

    w_out_L5N1393_blank = calculate_neuron_output_weights(
        model, blank_probe_normalised, layer, neuron)
    w_out_L5N1393_my = calculate_neuron_output_weights(
        model, my_probe_normalised, layer, neuron)
    w_out_L5N1393_unemb = neuron_output_weight_map_to_unemb(
        model, layer, neuron)

    fig2 = imshow(
        t.stack([w_out_L5N1393_blank, w_out_L5N1393_my, w_out_L5N1393_unemb]),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Output weights in terms of the probe for neuron L{layer}N{neuron}",
        facet_labels=["Blank Out", "My Out", 'Unemb'],
        # width=800,
        height=350,
    )

    if in_streamlit:
        col1, col2 = st.columns([1, 1])
        col1.plotly_chart(fig1)
        col2.plotly_chart(fig2)
    else:
        fig1.show()
        fig2.show()

    # Spectrum plots
    top_detector_values = detector_to_values(top_detector)
    label_board = (focus_states_flipped_value ==
                   top_detector_values) | top_detector_values.isnan()
    label = label_board.reshape(*label_board.shape[:2], 8*8).all(dim=-1)
    neuron_acts = focus_cache['post', layer][..., neuron]

    spectrum_data = pd.DataFrame({"acts": neuron_acts.flatten().tolist(),
                                  "label": label[:, :-1].flatten().tolist()})

    hist_fig = px.histogram(
        spectrum_data, x="acts", color="label",
        histnorm="percent", barmode="group", nbins=100,
        title=f"Spectrum plot for neuron L{layer}N{neuron}",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=350
    )

    confusion_matrix = pd.crosstab(
        spectrum_data['acts'] > ACTIVATION_THRES,
        spectrum_data['label']
    )
    cross_fig = px.imshow(confusion_matrix, text_auto=True, height=350)

    if in_streamlit:
        col1, col2 = st.columns([2, 2])
        col1.plotly_chart(hist_fig)
        col2.plotly_chart(cross_fig)
    else:
        cross_fig.show()

    # Max Activating boards
    # acts_by_label = spectrum_data.groupby('label')['acts']
    # top_activation_idx =   



def possible_util_cell(target_cell: Tuple[int, int]):
    """
    return a list of cells that are possible based on the target cell
    """
    tr, tc = target_cell
    possible_cells = []

    for r in range(8):
        for c in range(8):
            if abs(tr - r) == abs(tc - c) > 1:
                possible_cells.append((r, c))
            if (abs(tr - r) == 0 and abs(tc - c) > 1):
                possible_cells.append((r, c))
            if (abs(tr - r) > 1 and abs(tc - c) == 0):
                possible_cells.append((r, c))
    return possible_cells


def gen_detector_vec(target_cell: Tuple[int, int], util_cell: Tuple[int, int]):
    "return 8 * 8 map, 1 for based_cell and -1 for cells between based_cell and target_cell, 0 for other cells"
    tr, tc = target_cell
    ur, uc = util_cell
    detector_vec = t.full((8, 8), t.nan, dtype=t.float32)
    detector_vec[ur, uc] = 1
    detector_vec[tr, tc] = 0

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


def gen_detector_vecs(target_cell: Tuple[int, int], util_cells: List[Tuple[int, int]]):
    detector_vecs = []
    for util_cell in util_cells:
        detector_vec = gen_detector_vec(target_cell, util_cell)
        detector_vecs.append(detector_vec)
    return t.stack(detector_vecs)


def detector_to_values(detector: Float[Tensor, 'row col']):
    """
    Map the values of detector to the focus_states_flipped_value convention
    """
    values = t.full((8, 8), t.nan, dtype=t.float32)
    values[detector == 0] = 0
    values[detector == 1] = 2
    values[detector == -1] = 1
    return values


def cal_score_weight_probe(
    w_in_blank: Float[Tensor, 'd_mlp row col'],
    target_cell: Tuple[int, int],
):
    row, col = target_cell
    w_in_blank_norm = w_in_blank / w_in_blank.norm(dim=(1, 2), keepdim=True)
    return w_in_blank_norm[:, row, col]


def cal_score_write_unemb(
    w_out_unemb: Float[Tensor, 'd_mlp n_vocab'],
    target_cell: Tuple[int, int],
):
    row, col = target_cell
    cell_string = row * 8 + col
    cell_label = string_to_label(cell_string)
    cell_int = to_int(cell_label)

    w_out_unemb_norm = w_out_unemb / w_out_unemb.norm(dim=-1, keepdim=True)
    return w_out_unemb_norm[:, cell_int]


def cal_score_read_my(
    w_in_my: Float[Tensor, 'd_mlp row col'],
    target_cell: Tuple[int, int],
):
    w_in_my_norm = w_in_my / w_in_my.norm(dim=(1, 2), keepdim=True)

    util_cells = possible_util_cell(target_cell)
    d_mlp = w_in_my.shape[0]

    detectors = gen_detector_vecs(target_cell, util_cells).to(w_in_my.device)
    detectors_num = t.nan_to_num(detectors, nan=0.0)
    detectors_norm = detectors_num / \
        detectors_num.reshape(-1, 8*8).norm(dim=-
                                            1)[:, None, None]  # shape: [util_cells, 8, 8]
    util_score = einops.einsum(
        w_in_my_norm, detectors_norm, 'd_mlp row col, cells row col -> cells d_mlp')
    top_score, top_idx = util_score.max(dim=0)

    # assert score.shape == (d_mlp,)
    return top_score, detectors[top_idx]


# calculate cosine similarity between blank probe, my probe and W_U

SHOW_COSINE_SIM = False
if in_streamlit:
    SHOW_COSINE_SIM = st.checkbox('Show cosine similarity between probes and W_U')

def plot_cosine_sim(a: str, b: str):
    """
    Plot the cosine similarity between a and b. 
    Options: blank, my, W_U
    """
    assert a in ['blank', 'my', 'W_U']
    assert b in ['blank', 'my', 'W_U']

    W_U_norm = model.W_U / model.W_U.norm(dim=0, keepdim=True)
    W_U_board = t.zeros((model.cfg.d_model, 8, 8)).to(model.cfg.device)
    W_U_board.flatten(start_dim=1)[:, stoi_indices] = W_U_norm[:, 1:]
    
    name2tensor = {
        'blank': blank_probe_normalised,
        'my': my_probe_normalised,
        'W_U': W_U_board
    }

    cosine_sim = einops.einsum(
        name2tensor[a], name2tensor[b], 'd_model row col, d_model row col -> row col'
    )

    fig = imshow(
        cosine_sim,
        y=[i for i in "ABCDEFGH"],
        title=f"Cosine similarity between {a} and {b}",
        zmin=-0.5,
        zmax=0.5,
        aspect="equal",
        height=350,
        width=300,
    )
    return fig

if in_streamlit and SHOW_COSINE_SIM:
    cols = st.columns(3)
    pairs = [('blank', 'my'), ('blank', 'W_U'), ('my', 'W_U')]
    for col, (a, b) in zip(cols, pairs):
        col.plotly_chart(plot_cosine_sim(a, b), aspect='equal')




# calculate the cosine similarity between the output weights and W_U for each neuron in specific layer
def relu(x: Tensor):
    return x.clamp(min=0)

layer = 5
cell_label = 'C1'
ACTIVATION_THRES = 0
# choose from ['read_blank', 'write_unemb', 'read_my', 'all']
SCORING_METRIC = 'multiply all'

if in_streamlit:
    with st.sidebar:
        # cell_label = st.text_input('Target Cell', 'C0')
        cell_label = st.selectbox('Target Cell', [
                                  f'{alp}{num}' for alp in 'ABCDEFGH' for num in range(8)], index=16)
        layer = st.slider('Layer', 0, 7, 5)

        ACTIVATION_THRES = st.slider(
            'Activation threshold', min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        SCORING_METRIC = st.selectbox(
            'Scoring metric', [
                'read_blank', 
                'write_unemb', 
                'read_my', 
                'multiply all', 
                'Find invalid moves: -write_unemb', 
                'Destroy: -(read_blank * write_blank)', 
                'Destroy: -(read_my * write_my)',
                'Destroy: -(write_unemb * write_blank)',
                'Enhance: read_blank * write_blank',
                'Enhance: read_my * write_my'
            ], index=3
        )
        

# row and column of the cell
cell = (ord(cell_label[0]) - ord('A'), int(cell_label[-1]))

w_in_L5 = model.W_in[layer, :, :]  # shape: [d_model, d_mlp]
w_in_L5_norm = w_in_L5 / w_in_L5.norm(dim=0, keepdim=True)

w_out_L5 = model.W_out[layer, :, :]  # shape: [d_mlp, d_model]
w_out_L5_norm = w_out_L5 / w_out_L5.norm(dim=-1, keepdim=True)

# shape: [d_model, n_vocab]
W_U_norm = model.W_U / model.W_U.norm(dim=0, keepdim=True)

w_in_L5_my = einops.einsum(w_in_L5_norm, my_probe_normalised,
                           'd_model d_mlp, d_model row col -> d_mlp row col')
w_out_L5_my = einops.einsum(w_out_L5_norm, my_probe_normalised,
                            'd_mlp d_model, d_model row col -> d_mlp row col')
w_in_L5_blank = einops.einsum(
    w_in_L5_norm, blank_probe_normalised, 'd_model d_mlp, d_model row col -> d_mlp row col')
w_out_L5_blank = einops.einsum(
    w_out_L5_norm, blank_probe_normalised, 'd_mlp d_model, d_model row col -> d_mlp row col')
w_out_L5_umemb = einops.einsum(
    w_out_L5_norm, W_U_norm, 'd_mlp d_model, d_model n_vocab -> d_mlp n_vocab')

# calculate scores
score_read_blank = cal_score_weight_probe(w_in_L5_blank, cell)
score_write_blank = cal_score_weight_probe(w_out_L5_blank, cell)

score_read_my = cal_score_weight_probe(w_in_L5_my, cell)
score_write_my = cal_score_weight_probe(w_out_L5_my, cell)

score_write_unemb = cal_score_write_unemb(w_out_L5_umemb, cell) 
score_detector_match, top_detector = cal_score_read_my(w_in_L5_my, cell)

sub_score = {
    'read_blank': score_read_blank,
    'write_unemb': score_write_unemb,
    'read_my': score_detector_match,
}

if SCORING_METRIC == 'multiply all':
    score = relu(score_read_blank) * relu(score_write_unemb) * relu(score_detector_match)
elif SCORING_METRIC == 'read_blank':
    score = score_read_blank
elif SCORING_METRIC == 'write_unemb':
    score = score_write_unemb
elif SCORING_METRIC == 'read_my':
    score = score_read_my
elif SCORING_METRIC == 'Find invalid moves: -write_unemb':
    score = -score_write_unemb
elif SCORING_METRIC == 'Destroy: -(read_blank * write_blank)':
    score = -(score_read_blank * score_write_blank) 
elif SCORING_METRIC == 'Destroy: -(read_my * write_my)':
    score = -(score_read_my * score_write_my)
elif SCORING_METRIC == 'Destroy: -(write_unemb * write_blank)':
    score = -(score_write_unemb * score_write_blank)
elif SCORING_METRIC == 'Enhance: read_blank * write_blank':
    score = score_read_blank * score_write_blank
elif SCORING_METRIC == 'Enhance: read_my * write_my':
    score = score_read_my * score_write_my

n_top_neurons = 1
top_neurons = score.argsort(descending=True)[:n_top_neurons]

# visualize the input and output weights for these neurons
if in_streamlit:
    tabs = st.tabs([f'L{layer}N{neuron}' for neuron in top_neurons])
    for neuron, tab in zip(top_neurons, tabs):
        with tab:
            neuron_and_blank_my_emb(layer, neuron.item(),
                                    score, sub_score, top_detector[neuron])

  
