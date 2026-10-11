"""Solve the 8-puzzle with A* search.

The puzzle is a 3x3 grid holding tiles 1-8 and one empty space, written 'e'.
A state is stored as a string: cells in a row are joined by '-' and rows are
joined by newlines. For example, the goal state is:

    1-2-3
    4-5-6
    7-8-e

A move slides a tile next to the empty space into it. This script asks the
user for a starting puzzle, checks that it is valid and solvable, then uses
simpleai's A* search, with Manhattan distance as the heuristic, to find the
shortest sequence of moves to GOAL.
"""

import random

from simpleai.search import astar, SearchProblem

GOAL = '''1-2-3
4-5-6
7-8-e'''

INITIAL = '''4-1-2
7-e-3
8-5-6'''


def list_to_string(list_):
    """Convert a grid of tiles into its string form.

    Args:
        list_: A list of rows, where each row is a list of tile strings.

    Returns:
        The grid as a string, with cells joined by '-' and rows by newlines.
    """
    return '\n'.join(['-'.join(row) for row in list_])

def string_to_list(string_):
    """Convert a puzzle state string into a grid of tiles.

    Args:
        string_: A state string, with cells separated by '-' and rows by
            newlines.

    Returns:
        A list of rows, where each row is a list of tile strings. The lists
        are new objects, so the caller can change them freely.
    """
    return [row.split('-') for row in string_.split('\n')]

def get_location(rows, element):
    """Find the row and column of a tile in the grid.

    Args:
        rows: The grid, as returned by string_to_list.
        element: The tile to look for, such as '5' or 'e'.

    Returns:
        A (row, column) tuple of zero-based indexes, or None if the tile is
        not in the grid.
    """
    for i, row in enumerate(rows):
        for j, item in enumerate(row):
            if item == element:
                return i, j


def parse_puzzle(text):
    """Turn the user's input into a puzzle state string.

    The input lists the 9 cells in row order, top-left to bottom-right. Cells
    may be separated by spaces, commas or dashes, or not separated at all,
    so '4 1 2 7 e 3 8 5 6', '4-1-2-7-e-3-8-5-6' and '4127e3856' are all
    accepted. 'E' is treated the same as 'e'.

    Args:
        text: The line the user typed.

    Returns:
        The puzzle as a state string, such as '4-1-2\\n7-e-3\\n8-5-6'.

    Raises:
        ValueError: If the input does not contain exactly the tiles 1-8 and
            'e', each once. The message says what is wrong.
    """
    cells = [ch for ch in text.lower() if ch not in ' ,-\t']
    if len(cells) != 9:
        raise ValueError(f'Expected 9 cells but found {len(cells)}.')

    invalid = sorted(set(cells) - set('12345678e'))
    if invalid:
        raise ValueError(f'Invalid characters: {", ".join(invalid)}. '
                         'Use only the digits 1-8 and e.')

    repeated = sorted(ch for ch in set(cells) if cells.count(ch) > 1)
    if repeated:
        raise ValueError(f'Each cell must appear exactly once, but these '
                         f'appear more than once: {", ".join(repeated)}.')

    return list_to_string([cells[0:3], cells[3:6], cells[6:9]])


def is_solvable(state):
    """Check whether the goal can be reached from a state.

    Only half of all 8-puzzle layouts can be solved. Ignoring the empty
    space, read the tiles in row order and count the inversions: pairs where
    a larger number comes before a smaller one. On a 3x3 grid every move
    changes this count by an even number, and the goal has 0 inversions, so
    a state is solvable only if its count is even.

    Args:
        state: A valid puzzle state string.

    Returns:
        True if the puzzle can be solved, otherwise False.
    """
    tiles = [int(t) for row in string_to_list(state) for t in row if t != 'e']
    inversions = sum(1 for i in range(len(tiles))
                     for j in range(i + 1, len(tiles))
                     if tiles[i] > tiles[j])
    return inversions % 2 == 0


def random_state():
    """Generate a random puzzle that can be solved.

    Shuffles the tiles until the layout is solvable and is not already the
    goal. About half of all shuffles are solvable, so this finishes quickly.

    Returns:
        A random, solvable puzzle state string that differs from GOAL.
    """
    cells = list('12345678e')
    while True:
        random.shuffle(cells)
        state = list_to_string([cells[0:3], cells[3:6], cells[6:9]])
        if state != GOAL and is_solvable(state):
            return state


MENU = '''
Choose a starting puzzle:
  1. Enter a puzzle manually
  2. Pick a random puzzle
  3. Use the example puzzle
  4. Quit'''


def read_manual_state():
    """Ask the user to type a puzzle until they enter a valid one.

    Input that is malformed or cannot be solved is rejected with a message,
    and the user is asked again. Pressing Enter without typing anything
    goes back to the menu.

    Returns:
        A valid, solvable puzzle state string, or None if the user went
        back to the menu.
    """
    print('\nEnter the puzzle as 9 cells in row order (top-left to')
    print('bottom-right), using 1-8 for the tiles and e for the empty space.')
    print('Example: 4 1 2 7 e 3 8 5 6')
    while True:
        text = input('\nPuzzle (press Enter to go back): ').strip()
        if not text:
            return None
        try:
            state = parse_puzzle(text)
        except ValueError as error:
            print(f'Invalid input: {error}')
            continue
        if not is_solvable(state):
            print('This puzzle cannot be solved: no sequence of moves reaches')
            print(f'the goal from it. Please try another.\n{state}')
            continue
        return state


def choose_initial_state():
    """Show the menu and return the starting puzzle the user picks.

    Keeps showing the menu until the user picks a puzzle or quits. Any
    choice other than 1-4 is rejected with a message.

    Returns:
        A valid, solvable puzzle state string, or None if the user chose
        to quit.
    """
    while True:
        print(MENU)
        choice = input('Option (1-4): ').strip()
        if choice == '1':
            state = read_manual_state()
            if state is not None:
                return state
        elif choice == '2':
            return random_state()
        elif choice == '3':
            return INITIAL
        elif choice == '4':
            return None
        else:
            print(f'Invalid option: {choice!r}. Please enter 1, 2, 3 or 4.')


def solve(initial):
    """Solve a puzzle with A* and print each step of the solution.

    Args:
        initial: A valid, solvable puzzle state string.
    """
    result = astar(PuzzleSolver(initial), graph_search=True)
    for i, (action, state) in enumerate(result.path()):
        print()
        if action is None:
            print('Initial configuration')
        else:
            print(f'Step {i}: move tile {action} into the empty space')
        print(state)
    moves = len(result.path()) - 1
    print(f"\nSolved in {moves} move{'' if moves == 1 else 's'}.")


# Goal (row, column) of every tile, computed once so the heuristic does not
# have to search the goal grid on every call.
GOAL_POSITIONS = {}
for tile in '12345678e':
    GOAL_POSITIONS[tile] = get_location(string_to_list(GOAL), tile)


class PuzzleSolver(SearchProblem):
    """The 8-puzzle as a simpleai search problem.

    States are puzzle strings (see the module docstring). An action is the
    tile, such as '5', that slides into the empty space. Every move costs 1,
    which is simpleai's default cost, so A* finds the solution with the
    fewest moves.

    Pass the starting state to the constructor, for example
    PuzzleSolver(INITIAL).
    """

    def actions(self, state):
        """List the moves that can be made from a state.

        Args:
            state: The current puzzle state string.

        Returns:
            The tiles directly above, below, left of, or right of the empty
            space. Each of these can slide into it.
        """
        rows = string_to_list(state)
        row_e, col_e = get_location(rows, 'e')
        moves = []
        if row_e > 0: moves.append(rows[row_e - 1][col_e])
        if row_e < 2: moves.append(rows[row_e + 1][col_e])
        if col_e > 0: moves.append(rows[row_e][col_e - 1])
        if col_e < 2: moves.append(rows[row_e][col_e + 1])
        return moves

    def result(self, state, action):
        """Return the state reached by making a move.

        Args:
            state: The current puzzle state string.
            action: The tile to slide into the empty space. It must be one of
                the values returned by actions(state).

        Returns:
            A new state string with the tile and the empty space swapped.
        """
        rows = string_to_list(state)
        row_e, col_e = get_location(rows, 'e')
        row_t, col_t = get_location(rows, action)
        rows[row_e][col_e], rows[row_t][col_t] = rows[row_t][col_t], rows[row_e][col_e]
        return list_to_string(rows)

    def is_goal(self, state):
        """Return True if the state is the solved puzzle (GOAL)."""
        return state == GOAL

    def heuristic(self, state):
        """Estimate how many moves are left to reach the goal.

        Adds up the Manhattan distance (rows apart plus columns apart) from
        each tile to its goal position. The empty space is not counted. Each
        move shifts one tile by one cell, so this never overestimates the
        true number of moves. That makes it admissible, and A* is guaranteed
        to find an optimal solution.

        Args:
            state: The puzzle state string to estimate from.

        Returns:
            The total Manhattan distance of tiles 1-8, as an int.
        """
        rows = string_to_list(state)
        distance = 0
        for tile in '12345678':
            r, c = get_location(rows, tile)
            gr, gc = GOAL_POSITIONS[tile]
            distance += abs(r - gr) + abs(c - gc)
        return distance


if __name__ == '__main__':
    try:
        while True:
            initial = choose_initial_state()
            if initial is None:
                break
            solve(initial)
    except (EOFError, KeyboardInterrupt):
        print()
    print('Goodbye.')
