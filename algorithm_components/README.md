# Algorithm Components

These `.py` files are algorithm material for `autosolver_agent.py`.

The agent reads this directory by default and does not read the submit file while running. The files here are not evaluated as a baseline; they are provided so the LLM can choose, rewrite, and combine methods into a new standalone solver.

Files:

- `01_io_core.py`: parsing, cleaning, and core data structures.
- `02_objective.py`: proxy objective and comparison helpers.
- `03_initial_builders.py`: greedy, exact-cover, singleton, pair, bundle, and optional CP-SAT seed builders.
- `04_matching_skeleton.py`: pair skeleton and min-cost-flow style assignment helpers.
- `05_local_search.py`: local replacement, rider rearrangement, LNS, and polishing helpers.
- `06_search_orchestration.py`: annealing, main search scheduling, and legacy search material.
