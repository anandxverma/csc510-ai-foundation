# 8-Puzzle Solver

`puzzle-solver.py` solves the classic 8-puzzle using A* search.

The puzzle is a 3x3 grid with tiles `1`–`8` and one empty space, shown as `e`. A move slides a tile next to the empty space into it. The goal is to reach this layout:

```
1-2-3
4-5-6
7-8-e
```

The program asks for a starting puzzle and checks that it is valid and solvable. It then uses the A* search from the [simpleai](https://github.com/simpleai-team/simpleai) library to find the shortest solution. A* is guided by the Manhattan distance heuristic. Every step of the solution is printed, followed by the total number of moves.

## Requirements

- Python 3.6 or later
- The `simpleai` package

## Setup

Install the dependency, optionally inside a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install simpleai
```

## Running the program

From this directory, run:

```bash
python3 puzzle-solver.py
```

You will see a menu:

```
Choose a starting puzzle:
  1. Enter a puzzle manually
  2. Pick a random puzzle
  3. Use the example puzzle
  4. Quit
```

- **1. Enter a puzzle manually**: Type the 9 cells in row order, from top-left to bottom-right. Use `1`–`8` for the tiles and `e` for the empty space. Cells can be separated by spaces, commas or dashes, or not separated at all. For example, these are all the same puzzle:
  ```
  4 1 2 7 e 3 8 5 6
  4-1-2-7-e-3-8-5-6
  4127e3856
  ```
  If the input is invalid or the puzzle cannot be solved, the program explains why and asks again. Press Enter on an empty line to go back to the menu.
- **2. Pick a random puzzle**: Generates a random puzzle that is guaranteed to be solvable.
- **3. Use the example puzzle**: Solves this built-in puzzle:
  ```
  4-1-2
  7-e-3
  8-5-6
  ```
- **4. Quit**: Exits the program. You can also press `Ctrl+C` or `Ctrl+D` at any time.

After a puzzle is solved, the menu appears again so you can solve another one.

## Example output

Output for the built-in example puzzle (option 3), shortened:

```
Initial configuration
4-1-2
7-e-3
8-5-6

Step 1: move tile 5 into the empty space
4-1-2
7-5-3
8-e-6

...

Solved in 8 moves.
```
