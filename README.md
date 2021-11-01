# Preferential access and fairness in waste management
This project is based on the paper **Balancing preferential access and fairness with an application to waste management: mathematical models, optimality conditions, and heuristics** by C. Schmitt and B. Singh. There, we propose a Mixed Integer Quadratic Programming (MIQP) Model for a Facility Location Problem that seeks to assign users to facilities in a manner that balances access for users and fairness among facilities. We present a case study of this problem for the allocation of recycling centers in Bavaria. We refer to the paper for further information on methods and assumptions.
## Repository content
The repository contains the following content:
- `data` contains the two input data files: `users_and_facilities.xlsx` contains all ZIP codes and recycling centers related data like the population, centroid and regional spatial type (rural/urban) of each ZIP code as well as the capacity, centroid and regional spatial type of each recycling center. `travel_dict.json.pbz2` is a compressed json file that contains the travel probabilities from each ZIP code to each recycling center.
- `model` contains scripts for the optimization of the MIQP model and visualization of the results: `model.py` contains functions for building and optimizing the MIQP model, while `greedy_heuristic.py` applies a greedy heuristic to achieve a feasible solution. `plotting.py` and `results.py` contain functions tasked with visualizing results through various different plots as well as excel tables. They also contain superordinate functions that create the corresponding results first by running functions from `model.py` and/or `greedy_heuristic.py` before creating the corresponding visualization. These functions are called by functions in `tables_and_figures.py` to create the exact same figures and tables that are included in the paper from scratch. Lastly, `utils.py` contains helper and utility functions.
- `results` contains the excel tables and figures visualizing the results that are included in the paper.

