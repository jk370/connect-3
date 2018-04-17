import numpy as np


def get_downward_diagonal_indices(startRow, startCol, length=3):
    list_of_indices = [np.zeros(2) for _ in range(length)]
    for i in range(0, length):
        list_of_indices[i] = [startRow - 1 - i, startCol + 1 + i]
    return list_of_indices


def get_upward_diagonal_indices(startRow, startCol, length=3):
    list_of_indices = [np.zeros(2) for _ in range(length)]
    for i in range(0, length):
        list_of_indices[i] = [startRow + 1 + i, startCol + 1 + i]
    return list_of_indices


def get_grid_values(list_of_indices, array):
    return np.array([array[x, y] for x, y in list_of_indices])


def print_grid(grid):
    print(grid[::-1, ])


def other_token(current_token):
    if current_token == 'x':
        return 'o'
    else:
        return 'x'


def search_sequence_numpy(arr, seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create 2D array of sliding indices across entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:, None] + r_seq] == seq).all(1)

    return np.any(M)