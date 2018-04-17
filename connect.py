## Old development version
import numpy as np
import utils

class Connect:
    def __init__(self, num_cols=5, num_rows=3, num_connect=3, verbose=True):
        """
        Define a new Connect object
        """

        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_connect = num_connect
        self.verbose = verbose

        self.players = ['o', 'x']
        self.other_player = {'o': 'x', 'x': 'o'}

    def reset(self, first_player='random'):
        self.grid = np.full(fill_value=" ", shape=(self.num_rows, self.num_cols), dtype=str)

        # Each column index is one action.
        self.available_actions = np.arange(self.num_cols)

        # Keep track of the lowest free row position per column (where a disk would land if dropped in that column)
        self.lowest_free_rows = np.zeros(self.num_cols, dtype=int)

        if first_player == 'random':
            self.player_at_turn = np.random.choice(self.players)
        elif first_player in self.players:
            self.player_at_turn = first_player
        else:
            raise ValueError("The argument first_player has to be either 'random', 'x', or 'o'.")

        # Keep track of the last action played (simplifies checking for terminal states).
        self.last_action = None

        self.game_over = False
        if self.verbose:
            print("Game has been reset.")
            print(self.grid[::-1, ])

    def change_turn(self):
        self.player_at_turn = self.other_player[self.player_at_turn]

    def act(self, action):
        """
        Given an action (a column index; known to be a valid action!), generate the new board

        :param action: an integer referring to the column index where a new token/disk should be dropped
        """
        self.grid[self.lowest_free_rows[action], action] = self.player_at_turn
        self.lowest_free_rows[action] += 1
        if self.lowest_free_rows[action] == self.num_rows:
            self.available_actions = np.setdiff1d(self.available_actions, action)
        self.last_action = action

        if self.verbose:
            print(self.grid[::-1, ])

    def grid_is_full(self):
        return np.all(self.lowest_free_rows == self.num_rows)

    def was_winning_move(self):
        """
        Check if the move that has just been made wins the game.

        Determine in which row the disk (token) landed using self.last_action and look at that row,
        column and both diagonals including this token. Check whether there is any sequence of
        length 'num_connect' of the same token type.

        For example, if num_connect == 3

        ' 'd' ' ' 'c' ' ' 'u' ' '
        ' ' ' 'd' 'c' 'u' ' ' ' '
        ' 'r' 'r' 'x' 'r' 'r' ' '
        ' ' ' 'u' 'c' 'd' ' ' ' '
        ' 'u' ' ' 'c' ' ' 'd' ' '
        ' ' ' ' ' ' ' ' ' ' ' ' '
        ' ' ' ' ' ' ' ' ' ' ' ' '

        and "x" is the position the token has dropped, check whether there is a sequence of 'x' of length 3
        in the corresponding row (r), column (c), upward-diagonal (u), or downward diagonal (d).

        [This function could be made MUCH more efficient by excluding some of the checks beforehand, for
         example, based on the row height of the last_action.]

        :return: a boolean, True if the last move was a winning move
        """
        game_is_won = False

        action_row = self.lowest_free_rows[self.last_action] - 1
        action_col = self.last_action
        winning_sequence = np.full(shape=self.num_connect, fill_value=self.player_at_turn)

        # Calculate candidate vectors
        row_candidates = self.grid[action_row, max(0, action_col - self.num_connect + 1) : min(self.num_cols, action_col + self.num_connect)]
        if utils.search_sequence_numpy(row_candidates, winning_sequence):
            game_is_won = True
        else:
            col_candidates = self.grid[max(0, action_row - self.num_connect + 1): min(self.num_rows, action_row + self.num_connect), action_col]
            if utils.search_sequence_numpy(col_candidates, winning_sequence):
                game_is_won = True
            else:
                diag_index_up = action_col - action_row
                diag_up_candidates = np.diagonal(self.grid, diag_index_up)
                if utils.search_sequence_numpy(diag_up_candidates, winning_sequence):
                    game_is_won = True
                else:
                    diag_index_down = action_row + action_col - (self.num_rows - 1)
                    diag_down_candidates = np.diagonal(self.grid[::-1], diag_index_down)
                    if utils.search_sequence_numpy(diag_down_candidates, winning_sequence):
                        game_is_won = True

        if self.verbose and game_is_won:
            print("Player '", self.player_at_turn, "' has won the game!")
        return game_is_won

