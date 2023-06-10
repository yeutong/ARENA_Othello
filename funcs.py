import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union, Callable

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch as t
import transformer_lens.utils as utils
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.notebook import tqdm
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from functools import partial

from plotly_utils import imshow
import transformer_lens.patching as patching

working_dir = Path(f"{os.getcwd()}").resolve()
device = t.device("cuda" if t.cuda.is_available() else "cpu")


from othello_world.mechanistic_interpretability.mech_interp_othello_utils import (
    OthelloBoardState,
    int_to_label,
    plot_board,
    plot_board_log_probs,
    plot_single_board,
    string_to_label,
    to_int,
    to_string,
)


def relu(x: Tensor):
    return x.clamp(min=0)


def load_model():
    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
        device=device,
    )
    model = HookedTransformer(cfg)
    sd = utils.download_file_from_hf(
        "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
    )
    model.load_state_dict(sd)
    return model


def load_board_seq(path="othello_world/mechanistic_interpretability/"):
    board_seqs_int = t.tensor(np.load(path + "board_seqs_int_small.npy"), dtype=t.long)
    # Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
    board_seqs_string = t.tensor(
        np.load(path + "board_seqs_string_small.npy"), dtype=t.long
    )

    return board_seqs_int, board_seqs_string


# Define possible indices (excluding the four center squares)
stoi_indices = [i for i in range(64) if i not in [27, 28, 35, 36]]

# Define our rows, and the function that converts an index into a (row, column) label, e.g. `E2`
alpha = "ABCDEFGH"


def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"


def plot_square_as_board(state, diverging_scale=True, **kwargs):
    """Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0"""
    kwargs = {
        "y": [i for i in alpha],
        "x": [str(i) for i in range(8)],
        "color_continuous_scale": "RdBu" if diverging_scale else "Blues",
        "color_continuous_midpoint": 0.0 if diverging_scale else None,
        "aspect": "equal",
        **kwargs,
    }
    imshow(state, **kwargs)


def one_hot(list_of_ints, num_classes=64):
    out = t.zeros((num_classes,), dtype=t.float32)
    out[list_of_ints] = 1.0
    return out


def board_color_to_state(board: OthelloBoardState):
    """
    board state: 1 is black, -1 is white, 0 is blank
    next_hand_color: 1 is black, -1 is white

    Return:
    1 is mine, -1 is theirs, 0 is blank
    """
    next_hand_color = board.next_hand_color
    return board.state * next_hand_color


def detect_board_match(
    board: OthelloBoardState,
    target_str: int,
    target_pos: Tuple[int, int],
    target_state: str,
):
    """
    target_str: cell_int (0-60)
    target_pos: row, col
    target_state: 'valid', 'invalid', 'blank', 'mine', 'theirs'
    """

    board_state = board_color_to_state(board)

    # Map target states to corresponding conditions
    target_conditions = {
        "valid": lambda: target_str in board.get_valid_moves(),
        "invalid": lambda: target_str not in board.get_valid_moves(),
        "blank": 0,
        "mine": 1,
        "theirs": -1,
    }

    assert (
        target_state in target_conditions
    ), "target_state should be 'valid', 'invalid', 'blank', 'mine', or 'theirs'"

    condition = target_conditions.get(target_state)

    # For the 'valid' and 'invalid' cases
    if callable(condition):
        return condition()
    # For the 'blank', 'mine', and 'theirs' cases
    else:
        return board_state[target_pos[0], target_pos[1]] == condition


def select_board_states(
    target_label: List[str],
    target_state: List[str],
    pos: Optional[int] = None,
    batch_size=10,
    game_str_gen=None,
):
    """
    target_pos: str, e.g. 'C0'
    target_state: str, ['blank', 'mine', 'theirs', 'valid', 'invalid']

    the model predict white first
    board state: 1 is black, -1 is white, 0 is blank
    """
    assert all(
        t_state in ["blank", "mine", "theirs", "valid", "invalid"]
        for t_state in target_state
    )

    target_str = to_string(target_label)
    target_pos = [(s // 8, s % 8) for s in target_str]
    n_found = 0

    for game_string in game_str_gen:
        board = OthelloBoardState()

        if pos is None:
            for idx, move in enumerate(game_string):
                board.umpire(move)

                if all(
                    [
                        detect_board_match(board, *target_args)
                        for target_args in zip(target_str, target_pos, target_state)
                    ]
                ):
                    n_found += 1
                    yield game_string[: idx + 1]

                    if n_found >= batch_size:
                        return
        else:
            board.update(game_string[:pos])

            if all(
                [
                    detect_board_match(board, *target_args)
                    for target_args in zip(target_str, target_pos, target_state)
                ]
            ):
                n_found += 1
                yield game_string[:pos]

                if n_found >= batch_size:
                    return


def yield_similar_boards(target_game_str: Int[Tensor, "seq"], game_str_gen, sim_threshold: float = 0.0,
                         batch_size=10, by_valid_moves=False, match_valid_moves_number=False):
    target_board = OthelloBoardState()
    target_board.update(target_game_str)
    target_board_state = board_color_to_state(target_board)

    n_found = 0
    for board_str in game_str_gen:
        board = OthelloBoardState()
        board.update(board_str)

        if by_valid_moves:
            valid_moves = board.get_valid_moves()
            target_valid_moves = target_board.get_valid_moves()
            
            if match_valid_moves_number and len(valid_moves) != len(target_valid_moves):
                continue

            valid_moves_in_target = sum([float(move in target_valid_moves) for move in valid_moves])
            similarity = valid_moves_in_target/len(target_valid_moves)
            if similarity >= sim_threshold:
                n_found += 1
                # print(similarity)
                yield board_str
        else:
            board_state = board_color_to_state(board)
            if (board_state == target_board_state).mean() >= sim_threshold:
                n_found += 1
                yield board_str

        if n_found >= batch_size:
            return


def extend_game_string(starting_game: Int[Tensor, "seq"], final_length: int):
    board = OthelloBoardState()
    board.update(starting_game)

    new_moves = []
    for _ in range(final_length - starting_game.shape[0]):
        valid_moves = board.get_valid_moves()
        next_move = random.choice(valid_moves)
        board.umpire(next_move)
        new_moves.append(next_move)

    return t.cat([starting_game, t.Tensor(new_moves).long()])


def yield_extended_boards(
    starting_game: Int[Tensor, "seq"], final_length: int, batch_size=10
):
    for _ in range(batch_size):
        yield extend_game_string(starting_game, final_length)


def yield_tree_from_game_string(
    starting_game: Int[Tensor, "seq"], final_length: int, batch_size=10
):
    pass


def plot_probe_outputs(layer, game_index, move, focus_cache, linear_probe, **kwargs):
    residual_stream = focus_cache["resid_post", layer][game_index, move]
    # print("residual_stream", residual_stream.shape)
    probe_out = einops.einsum(
        residual_stream,
        linear_probe,
        "d_model, d_model row col options -> row col options",
    )
    probabilities = probe_out.softmax(dim=-1)
    plot_square_as_board(
        probabilities,
        facet_col=2,
        facet_labels=["P(Empty)", "P(Their's)", "P(Mine)"],
        **kwargs,
    )


def state_stack_to_one_hot(state_stack, rows, cols):
    """
    Creates a tensor of shape (games, moves, rows=8, cols=8, options=3), where the [g, m, r, c, :]-th entry
    is a one-hot encoded vector for the state of game g at move m, at row r and column c. In other words, this
    vector equals (1, 0, 0) when the state is empty, (0, 1, 0) when the state is "their", and (0, 0, 1) when the
    state is "my".
    """
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


def apply_scale(
    resid: Float[Tensor, "batch=1 seq d_model"],
    flip_dir: Float[Tensor, "d_model"],
    scale: int,
    pos: int,
):
    """
    Returns a version of the residual stream, modified by the amount `scale` in the
    direction `flip_dir` at the sequence position `pos`, in the way described above.
    """
    norm_flip_dir = flip_dir / flip_dir.norm()
    alpha = resid[0, pos].dot(norm_flip_dir)
    resid[0, pos] -= (scale + 1) * alpha * norm_flip_dir
    return resid


def plot_contributions(contributions, component: str, game_index: int, move: int):
    imshow(
        contributions,
        facet_col=0,
        y=list("ABCDEFGH"),
        facet_labels=[f"Layer {i}" for i in range(7)],
        title=f"{component} Layer Contributions to my vs their (Game {game_index} Move {move})",
        aspect="equal",
        width=1400,
        height=350,
    )


def calculate_attn_and_mlp_probe_score_contributions(
    focus_cache: ActivationCache,
    my_probe: Float[Tensor, "d_model rows cols"],
    layer: int,
    game_index: int,
    move: int,
) -> Tuple[Float[Tensor, "layers rows cols"], Float[Tensor, "layers rows cols"]]:
    attn_contr_all = []
    mlp_contr_all = []
    for l in range(layer + 1):
        attn_out = focus_cache["attn_out", l]
        mlp_out = focus_cache["mlp_out", l]

        attn_contr = einops.einsum(
            attn_out[game_index, move], my_probe, "d, d r c -> r c"
        )
        mlp_contr = einops.einsum(
            mlp_out[game_index, move], my_probe, "d, d r c -> r c"
        )

        attn_contr_all.append(attn_contr)
        mlp_contr_all.append(mlp_contr)
    return (t.stack(attn_contr_all), t.stack(mlp_contr_all))


def get_w_in(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    """
    Returns the input weights for the neuron in the list, at each square on the board.

    If normalize is True, the weights are normalized to unit norm.
    """
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
    """
    Returns the input weights for the neuron in the list, at each square on the board.
    """
    neuron = model.W_out[layer, neuron, :]  # shape: [d_model]
    if normalize:
        return neuron / neuron.norm()
    return neuron


def calculate_neuron_input_weights(
    model: HookedTransformer,
    probe: Float[Tensor, "d_model row col"],
    layer: int,
    neuron: int,
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the input weights for each neuron in the list, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    neuron = get_w_in(model, layer, neuron, normalize=True)
    return einops.einsum(neuron, probe, "d_model, d_model row col -> row col")


def calculate_neuron_output_weights(
    model: HookedTransformer,
    probe: Float[Tensor, "d_model row col"],
    layer: int,
    neuron: int,
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the output weights for each neuron in the list, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    neuron = get_w_out(model, layer, neuron, normalize=True)
    return einops.einsum(neuron, probe, "d_model, d_model row col -> row col")


def neuron_output_weight_map_to_unemb(
    model: HookedTransformer, layer: int, neuron: int
) -> Float[Tensor, "rows cols"]:
    neuron = get_w_out(model, layer, neuron, normalize=True)
    W_U_norm = model.W_U / model.W_U.norm(dim=0, keepdim=True)
    out = einops.einsum(neuron, W_U_norm, "d_model, d_model n_vocab -> n_vocab")
    board = t.zeros(8, 8).to(model.cfg.device)
    board.flatten()[stoi_indices] = out[1:]
    return board


def neuron_and_blank_my_emb(
    layer,
    neuron,
    model,
    blank_probe_normalised,
    my_probe_normalised,
    focus_states_flipped_value,
    focus_cache,
    ACTIVATION_THRES,
    score=None,
    sub_score=None,
    top_detector=None,
    in_streamlit=True,
):
    if score is not None:
        score = score[neuron]
        # print(f"Score: {score:2.2%}")
        # st.write(f'Score: {score:2.2%}')
        st.title(f"Neuron L{layer}N{neuron} (Score: {score:2.2%})")

    if sub_score is not None:
        score_read_blank = sub_score["read_blank"][neuron]
        score_read_my = sub_score["read_my"][neuron]
        score_write_unemb = sub_score["write_unemb"][neuron]

        scores_str = f"read blank: {score_read_blank:2.2%}, read my: {score_read_my:2.2%}, write unemb: {score_write_unemb:2.2%}"
        if in_streamlit:
            st.write(scores_str)
        else:
            print(scores_str)

    w_in_L5N1393_blank = calculate_neuron_input_weights(
        model, blank_probe_normalised, layer, neuron
    )
    w_in_L5N1393_my = calculate_neuron_input_weights(
        model, my_probe_normalised, layer, neuron
    )

    fig1 = imshow(
        t.stack([w_in_L5N1393_blank, w_in_L5N1393_my, top_detector / 3]),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
        facet_labels=["Blank In", "My In", "My In Top Detector"],
        # width=800,
        height=350,
    )

    w_out_L5N1393_blank = calculate_neuron_output_weights(
        model, blank_probe_normalised, layer, neuron
    )
    w_out_L5N1393_my = calculate_neuron_output_weights(
        model, my_probe_normalised, layer, neuron
    )
    w_out_L5N1393_unemb = neuron_output_weight_map_to_unemb(model, layer, neuron)

    fig2 = imshow(
        t.stack([w_out_L5N1393_blank, w_out_L5N1393_my, w_out_L5N1393_unemb]),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Output weights in terms of the probe for neuron L{layer}N{neuron}",
        facet_labels=["Blank Out", "My Out", "Unemb"],
        # width=800,
        height=350,
    )

    # if in_streamlit:
    col1, col2 = st.columns([1, 1])
    col1.plotly_chart(fig1)
    col2.plotly_chart(fig2)
    # else:
    #     fig1.show()
    #     fig2.show()

    # Spectrum plots
    top_detector_values = detector_to_values(top_detector)
    label_board = (
        focus_states_flipped_value == top_detector_values
    ) | top_detector_values.isnan()
    label = label_board.reshape(*label_board.shape[:2], 8 * 8).all(dim=-1)
    neuron_acts = focus_cache["post", layer][..., neuron]

    spectrum_data = pd.DataFrame(
        {
            "acts": neuron_acts.flatten().tolist(),
            "label": label[:, :-1].flatten().tolist(),
        }
    )

    hist_fig = px.histogram(
        spectrum_data,
        x="acts",
        color="label",
        histnorm="percent",
        barmode="group",
        nbins=100,
        title=f"Spectrum plot for neuron L{layer}N{neuron}",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=350,
    )

    confusion_matrix = pd.crosstab(
        spectrum_data["acts"] > ACTIVATION_THRES, spectrum_data["label"]
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
            if abs(tr - r) == 0 and abs(tc - c) > 1:
                possible_cells.append((r, c))
            if abs(tr - r) > 1 and abs(tc - c) == 0:
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


def detector_to_values(detector: Float[Tensor, "row col"]):
    """
    Map the values of detector to the focus_states_flipped_value convention
    """
    values = t.full((8, 8), t.nan, dtype=t.float32)
    values[detector == 0] = 0
    values[detector == 1] = 2
    values[detector == -1] = 1
    return values


def cal_score_weight_probe(
    w_in_blank: Float[Tensor, "d_mlp row col"],
    target_cell: Tuple[int, int],
):
    row, col = target_cell
    w_in_blank_norm = w_in_blank / w_in_blank.norm(dim=(1, 2), keepdim=True)
    return w_in_blank_norm[:, row, col]


def cal_score_write_unemb(
    w_out_unemb: Float[Tensor, "d_mlp n_vocab"],
    target_cell: Tuple[int, int],
):
    row, col = target_cell
    cell_string = row * 8 + col
    cell_label = string_to_label(cell_string)
    cell_int = to_int(cell_label)

    w_out_unemb_norm = w_out_unemb / w_out_unemb.norm(dim=-1, keepdim=True)
    return w_out_unemb_norm[:, cell_int]


def cal_score_read_my(
    w_in_my: Float[Tensor, "d_mlp row col"],
    target_cell: Tuple[int, int],
):
    w_in_my_norm = w_in_my / w_in_my.norm(dim=(1, 2), keepdim=True)

    util_cells = possible_util_cell(target_cell)
    d_mlp = w_in_my.shape[0]

    detectors = gen_detector_vecs(target_cell, util_cells).to(w_in_my.device)
    detectors_num = t.nan_to_num(detectors, nan=0.0)
    detectors_norm = (
        detectors_num / detectors_num.reshape(-1, 8 * 8).norm(dim=-1)[:, None, None]
    )  # shape: [util_cells, 8, 8]
    util_score = einops.einsum(
        w_in_my_norm, detectors_norm, "d_mlp row col, cells row col -> cells d_mlp"
    )
    top_score, top_idx = util_score.max(dim=0)

    # assert score.shape == (d_mlp,)
    return top_score, detectors[top_idx]


def plot_cosine_sim(a: str, b: str, model, blank_probe_normalised, my_probe_normalised):
    """
    Plot the cosine similarity between a and b.
    Options: blank, my, W_U
    """
    assert a in ["blank", "my", "W_U"]
    assert b in ["blank", "my", "W_U"]

    W_U_norm = model.W_U / model.W_U.norm(dim=0, keepdim=True)
    W_U_board = t.zeros((model.cfg.d_model, 8, 8)).to(model.cfg.device)
    W_U_board.flatten(start_dim=1)[:, stoi_indices] = W_U_norm[:, 1:]

    name2tensor = {
        "blank": blank_probe_normalised,
        "my": my_probe_normalised,
        "W_U": W_U_board,
    }

    cosine_sim = einops.einsum(
        name2tensor[a], name2tensor[b], "d_model row col, d_model row col -> row col"
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


def patching_metric(patched_logits: Float[Tensor, "batch seq d_vocab"], pos, answer_index, corrupted_log_prob, clean_log_prob):
	'''
	Function of patched logits, calibrated so that it equals 0 when performance is 
	same as on corrupted input, and 1 when performance is same as on clean input.

	Should be linear function of the logits for the F0 token at the final move.
	'''
	patched_log_probs = patched_logits.log_softmax(dim=-1)
	return (patched_log_probs[0, pos, answer_index] - corrupted_log_prob) / (clean_log_prob - corrupted_log_prob)

def neuron_patch(activations: Float[Tensor, 'batch seq neuron'], hook: HookPoint, 
                 index: Union[List, Int, Int[Tensor, 'd_mlp']], pos: int, clean_cache=ActivationCache) -> Float[Tensor, 'batch seq neuron']:
    activations[:, pos, index] = clean_cache['post', hook.layer()][:, pos, index]
    return activations


def get_act_patch_mlp_post(model: HookedTransformer, corrupted_tokens: Int[Tensor, "batch seq"],
                           clean_cache: ActivationCache, patching_metric: Callable, answer_index, 
                           corrupted_log_prob, clean_log_prob, layer=5, pos=-1,
                           ) -> Float[Tensor, 'pos neuron']:
    layer = [layer] if isinstance(layer, int) else layer

    model.reset_hooks()
    result = t.zeros(len(layer), model.cfg.d_mlp).to(model.cfg.device)

    for idx, lay in enumerate(layer):
        for neuron in tqdm(range(model.cfg.d_mlp)):
            neuron_hook = partial(neuron_patch, index=neuron, clean_cache=clean_cache, pos=pos)
            patched_logits = model.run_with_hooks(
                corrupted_tokens, fwd_hooks=[(utils.get_act_name('post', lay), neuron_hook)])
            result[idx, neuron] = patching_metric(patched_logits, pos, answer_index, corrupted_log_prob, clean_log_prob)

    return result.squeeze()

