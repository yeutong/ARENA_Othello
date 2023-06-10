import copy

import einops
import numpy as np
import streamlit as st
import torch as t

from funcs import (
    cal_score_read_my,
    cal_score_weight_probe,
    cal_score_write_unemb,
    load_board_seq,
    load_model,
    neuron_and_blank_my_emb,
    one_hot,
    plot_probe_outputs,
    relu,
    state_stack_to_one_hot,
    plot_cosine_sim,
    select_board_states,
    yield_similar_boards,
    yield_extended_boards,
    get_act_patch_mlp_post,
    patching_metric,

)
from othello_world.mechanistic_interpretability.mech_interp_othello_utils import (
    OthelloBoardState,
    string_to_label,
    to_int,
    plot_single_board,
)

# %%
t.set_grad_enabled(False)
stoi_indices = [i for i in range(64) if i not in [27, 28, 35, 36]]

# Define our rows, and the function that converts an index into a (row, column) label, e.g. `E2`
alpha = "ABCDEFGH"

device = "cuda" if t.cuda.is_available() else "cpu"
model = load_model()
board_seqs_int, board_seqs_string = load_board_seq(
    "othello_world/mechanistic_interpretability/"
)

num_games = 50
# shape: [50, 60] = [50 games, 60 moves each]
focus_games_int = board_seqs_int[:num_games]
focus_games_string = board_seqs_string[:num_games]


focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
focus_valid_moves = t.zeros((num_games, 60, 64), dtype=t.float32)

for i in range(num_games):
    board = OthelloBoardState()
    for j in range(60):
        board.umpire(focus_games_string[i, j].item())
        focus_states[i, j] = board.state
        focus_valid_moves[i, j] = one_hot(board.get_valid_moves())

focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))

OTHELLO_ROOT = "./othello_world"
OTHELLO_MECHINT_ROOT = OTHELLO_ROOT + "/mechanistic_interpretability"

full_linear_probe = t.load(
    OTHELLO_MECHINT_ROOT + "/main_linear_probe.pth", map_location=device
)

rows = 8
cols = 8
options = 3
assert full_linear_probe.shape == (3, model.cfg.d_model, rows, cols, options)

black_to_play_index = 0
white_to_play_index = 1
blank_index = 0
their_index = 1
my_index = 2

# Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")
linear_probe = t.zeros(model.cfg.d_model, rows, cols, options, device=device)
linear_probe[..., blank_index] = 0.5 * (
    full_linear_probe[black_to_play_index, ..., 0]
    + full_linear_probe[white_to_play_index, ..., 0]
)
linear_probe[..., their_index] = 0.5 * (
    full_linear_probe[black_to_play_index, ..., 1]
    + full_linear_probe[white_to_play_index, ..., 2]
)
linear_probe[..., my_index] = 0.5 * (
    full_linear_probe[black_to_play_index, ..., 2]
    + full_linear_probe[white_to_play_index, ..., 1]
)


alternating = np.array(
    [-1 if i % 2 == 0 else 1 for i in range(focus_games_int.shape[1])]
)
flipped_focus_states = focus_states * alternating[None, :, None, None]

# We now convert to one-hot encoded vectors
focus_states_flipped_one_hot = state_stack_to_one_hot(
    t.tensor(flipped_focus_states), rows=rows, cols=cols
)

# Take the argmax (i.e. the index of option empty/their/mine)
focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)


probe_out = einops.einsum(
    focus_cache["resid_post", 6],
    linear_probe,
    "game move d_model, d_model row col options -> game move row col options",
)

probe_out_value = probe_out.argmax(dim=-1)


correct_middle_odd_answers = (
    probe_out_value.cpu() == focus_states_flipped_value[:, :-1]
)[:, 5:-5:2]
accuracies_odd = einops.reduce(
    correct_middle_odd_answers.float(), "game move row col -> row col", "mean"
)

correct_middle_answers = (probe_out_value.cpu() == focus_states_flipped_value[:, :-1])[
    :, 5:-5
]
accuracies = einops.reduce(
    correct_middle_answers.float(), "game move row col -> row col", "mean"
)


blank_probe = linear_probe[..., 0] - (linear_probe[..., 1] + linear_probe[..., 2]) / 2
my_probe = linear_probe[..., 2] - linear_probe[..., 1]


pos = 20
game_index = 0

# Plot board state
moves = focus_games_string[game_index, : pos + 1]
# plot_single_board(moves)

# Plot corresponding model predictions
state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.0
state.flatten()[stoi_indices] = focus_logits[game_index, pos].log_softmax(dim=-1)[1:]


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

newly_legal = [
    string_to_label(move) for move in flipped_valid_moves if move not in valid_moves
]
newly_illegal = [
    string_to_label(move) for move in valid_moves if move not in flipped_valid_moves
]
print("newly_legal", newly_legal)
print("newly_illegal", newly_illegal)


st.write("got to here")

flip_dir = my_probe[:, cell_r, cell_c]

big_flipped_states_list = []
layer = 4
scales = [0, 1, 2, 4, 8, 16]

# Scale the probes down to be unit norm per cell
blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
# Set the center blank probes to 0, since they're never blank so the probe is meaningless
blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

with st.sidebar:
    # cell_label = st.text_input('Target Cell', 'C0')
    cell_label = st.selectbox(
        "Target Cell",
        [f"{alp}{num}" for alp in "ABCDEFGH" for num in range(8)],
        index=16,
    )
    layer = st.slider("Layer", 0, 7, 5)

    ACTIVATION_THRES = st.slider(
        "Activation threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05
    )
    SCORING_METRIC = st.selectbox(
        "Scoring metric",
        [
            "read_blank",
            "write_unemb",
            "read_my",
            "multiply all",
            "Find invalid moves: -write_unemb",
            "Destroy: -(read_blank * write_blank)",
            "Destroy: -(read_my * write_my)",
            "Destroy: -(write_unemb * write_blank)",
            "Enhance: read_blank * write_blank",
            "Enhance: read_my * write_my",
            "Patching",
            "Random"
        ],
        index=3,
    )


# row and column of the cell
cell = (ord(cell_label[0]) - ord("A"), int(cell_label[-1]))

w_in_L5 = model.W_in[layer, :, :]  # shape: [d_model, d_mlp]
w_in_L5_norm = w_in_L5 / w_in_L5.norm(dim=0, keepdim=True)

w_out_L5 = model.W_out[layer, :, :]  # shape: [d_mlp, d_model]
w_out_L5_norm = w_out_L5 / w_out_L5.norm(dim=-1, keepdim=True)

# shape: [d_model, n_vocab]
W_U_norm = model.W_U / model.W_U.norm(dim=0, keepdim=True)

w_in_L5_my = einops.einsum(
    w_in_L5_norm, my_probe_normalised, "d_model d_mlp, d_model row col -> d_mlp row col"
)
w_out_L5_my = einops.einsum(
    w_out_L5_norm,
    my_probe_normalised,
    "d_mlp d_model, d_model row col -> d_mlp row col",
)
w_in_L5_blank = einops.einsum(
    w_in_L5_norm,
    blank_probe_normalised,
    "d_model d_mlp, d_model row col -> d_mlp row col",
)
w_out_L5_blank = einops.einsum(
    w_out_L5_norm,
    blank_probe_normalised,
    "d_mlp d_model, d_model row col -> d_mlp row col",
)
w_out_L5_umemb = einops.einsum(
    w_out_L5_norm, W_U_norm, "d_mlp d_model, d_model n_vocab -> d_mlp n_vocab"
)

# calculate scores
score_read_blank = cal_score_weight_probe(w_in_L5_blank, cell)
score_write_blank = cal_score_weight_probe(w_out_L5_blank, cell)

score_read_my = cal_score_weight_probe(w_in_L5_my, cell)
score_write_my = cal_score_weight_probe(w_out_L5_my, cell)

score_write_unemb = cal_score_write_unemb(w_out_L5_umemb, cell)
score_detector_match, top_detector = cal_score_read_my(w_in_L5_my, cell)


# calculating patching score
if SCORING_METRIC == "Patching":
    for datapoint in select_board_states(["C0", "D1", "E2"], ["valid", "theirs", "mine"], pos=None, batch_size=1000, game_str_gen=board_seqs_string):
        print('inside datapoint loop')
        orig_extensions = yield_extended_boards(
            datapoint[:-1], datapoint.shape[0], batch_size=25
        )
        selected_board_states = select_board_states(
            ["C0", "C0"],
            ["blank", "invalid"],
            game_str_gen=orig_extensions,
            pos=datapoint.shape[0],
            batch_size=25,
        )
        alter_dataset = list(
            yield_similar_boards(
                datapoint,
                selected_board_states,
                sim_threshold=0.0,
                by_valid_moves=True,
                match_valid_moves_number=True,
                batch_size=25,
            )
        )
        if alter_dataset != []:
            orig_datapoint = datapoint
            alter_datapoint = alter_dataset[0]
            break

    
    clean_input = t.tensor(to_int(orig_datapoint))
    corrupted_input = t.tensor(to_int(alter_datapoint))

    clean_logits, clean_cache = model.run_with_cache(clean_input)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_input)

    clean_log_probs = clean_logits.log_softmax(dim=-1)
    corrupted_log_probs = corrupted_logits.log_softmax(dim=-1)


    pos = -1

    answer_index = to_int("C0")
    clean_log_prob = clean_log_probs[0, pos, answer_index]
    corrupted_log_prob = corrupted_log_probs[0, pos, answer_index]

    print('Everything is fine, prepared to patch')
    act_patch = get_act_patch_mlp_post(
        model,
        corrupted_input,
        clean_cache,
        patching_metric,
        answer_index,
        corrupted_log_prob,
        clean_log_prob,
        layer=layer,
        pos=pos,
    )


    col1, col2 = st.columns(2)
    col1.write('Clean board')
    col1.plotly_chart(plot_single_board(string_to_label(orig_datapoint), return_fig=True), aspect='equal')
    col2.write('Corrupt board')
    col2.plotly_chart(plot_single_board(string_to_label(alter_datapoint), return_fig=True), aspect='equal')


# act_patch shape: [d_mlp]
# top_neurons = act_patch.argsort(descending=True)[:5]

sub_score = {
    "read_blank": score_read_blank,
    "write_unemb": score_write_unemb,
    "read_my": score_detector_match,
}

if SCORING_METRIC == "multiply all":
    score = (
        relu(score_read_blank) * relu(score_write_unemb) * relu(score_detector_match)
    )
elif SCORING_METRIC == "read_blank":
    score = score_read_blank
elif SCORING_METRIC == "write_unemb":
    score = score_write_unemb
elif SCORING_METRIC == "read_my":
    score = score_read_my
elif SCORING_METRIC == "Find invalid moves: -write_unemb":
    score = -score_write_unemb
elif SCORING_METRIC == "Destroy: -(read_blank * write_blank)":
    score = -(score_read_blank * score_write_blank)
elif SCORING_METRIC == "Destroy: -(read_my * write_my)":
    score = -(score_read_my * score_write_my)
elif SCORING_METRIC == "Destroy: -(write_unemb * write_blank)":
    score = -(score_write_unemb * score_write_blank)
elif SCORING_METRIC == "Enhance: read_blank * write_blank":
    score = score_read_blank * score_write_blank
elif SCORING_METRIC == "Enhance: read_my * write_my":
    score = score_read_my * score_write_my
elif SCORING_METRIC == "Patching":
    score = t.abs(act_patch)
elif SCORING_METRIC == "Random":
    score = t.rand_like(score_read_blank)

n_top_neurons = 10
top_neurons = score.argsort(descending=True)[:n_top_neurons]

# visualize the input and output weights for these neurons
tabs = st.tabs([f"L{layer}N{neuron}" for neuron in top_neurons])
for neuron, tab in zip(top_neurons, tabs):
    if SCORING_METRIC == "Patching":
        detector = t.full((8, 8), t.nan).to(device)
        detector[2, 0] = 0
        detector[3, 1] = -1
        detector[4, 2] = 1
    else:
        detector = top_detector[neuron]

    with tab:
        neuron_and_blank_my_emb(
            layer,
            neuron.item(),
            model,
            blank_probe_normalised,
            my_probe_normalised,
            focus_states_flipped_value,
            focus_cache,
            ACTIVATION_THRES,
            score,
            sub_score,
            detector,
        )


# calculate cosine similarity between blank probe, my probe and W_U

SHOW_COSINE_SIM = False

SHOW_COSINE_SIM = st.checkbox("Show cosine similarity between probes and W_U")

if SHOW_COSINE_SIM:
    cols = st.columns(3)
    pairs = [("blank", "my"), ("blank", "W_U"), ("my", "W_U")]
    for col, (a, b) in zip(cols, pairs):
        col.plotly_chart(
            plot_cosine_sim(a, b, model, blank_probe_normalised, my_probe_normalised),
            aspect="equal",
        )


