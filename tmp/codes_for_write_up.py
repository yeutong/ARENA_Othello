# example board (exlaining pattern)
plot_single_board(int_to_label([20, 33, 39, 19, 32, 38, 41, 31, 37, 34, 25, 28, 21, 27, 11, 48, 40, 47,
        43, 13,  6,  5, 42, 26,]))

# get clean, corrupted example board
plot_single_board(int_to_label(corrupted_input[0, :end_position[0]+1]), title='Corrupted (C0 is invalid)')
plot_single_board(int_to_label(clean_input[0, :end_position[0]+1]), title='Clean (C0 is valid and pattern present)')

