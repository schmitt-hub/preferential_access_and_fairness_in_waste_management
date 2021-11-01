"""
Module containing functions for solving the MIP with a greedy heuristic
"""

from time import time
from utils import *
import numpy as np


def greedily_assign_users(users_and_facs_df, travel_dict, users, facs, open_facs, exp_travelers_dict, cap_dict):
    """
    assignment step of the greedy heuristic
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param open_facs: list of open facilities
    :param exp_travelers_dict: dictionary stating the population expected to travel from each user to each facility
    :param cap_dict: dictionary of the facilities' capacities
    :return: boolean indicating whether a feasible assignment could be constructed;
        a dictionary of assignments, facility utilizations and the objective value
    """

    # initialize
    cap_left = cap_dict.copy()
    user_assignment = {}
    unassigned_users = users.copy()

    while unassigned_users:
        users_not_assignable = True
        most_preferred_users = {j: [] for j in open_facs}
        # match every user with their most preferred facility
        for i in unassigned_users:
            possible_facs = [j for j in open_facs if cap_left[j] >= exp_travelers_dict[i][j]]
            if not possible_facs:
                continue
            most_preferred_fac = possible_facs[np.argmax([travel_dict[i][j] for j in possible_facs])]
            most_preferred_prob = travel_dict[i][most_preferred_fac]
            most_preferred_users[most_preferred_fac].append((i, most_preferred_prob))
            users_not_assignable = False
        # if no user could be assigned in this iteration, return without a new feasible assignment
        if users_not_assignable:
            return 0, {}
        # assign users to their most preferred facility in decreasing rank of their preference to this facility
        for j in most_preferred_users:
            sorted_users = sorted(most_preferred_users[j], key=lambda x: -x[1])
            for (i, prob) in sorted_users:
                if cap_left[j] >= users_and_facs_df.at[i, 'population'] * prob:
                    unassigned_users.remove(i)
                    user_assignment[i] = j
                    cap_left[j] -= users_and_facs_df.at[i, 'population'] * prob

    utilization = {j: sum(exp_travelers_dict[i][j] for i in users if user_assignment[i] == j) /
                 cap_dict[j] for j in facs}
    objective = sum(cap_dict[j] * (1 - utilization[j]) ** 2 for j in facs)
    return 1, {'assignment': user_assignment, 'utilization': {j: utilization[j] for j in open_facs}, 'obj': objective}


def solve_greedily(users_and_facs_df, travel_dict, users, facs, lb=0.0, budget_factor=1.0, cap_factor=1.5,
                   turnover_factor=0.02, tolerance=5e-3, time_limit=20000, iteration_limit=20):
    """
    greedy heuristic
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param lb: a lower bound on the optimal objective value
    :param budget_factor: ratio of facilities that are allowed to be opened
    :param cap_factor: factor by which all capacities are scaled
    :param turnover_factor: maximal fraction of facilities that are swapped in each iteration
    :param tolerance: tolerance on the optimality of the solution
    :param time_limit: time limit in seconds
    :param iteration_limit: limit for the number of consecutive iterations without improvement
    :return: dictionary of results
    """
    print('Solving greedily...')
    start_time = time()
    # filter the data frame by the used facilities and sort it by their capacities
    sorted_users_and_facs_df = users_and_facs_df.iloc[facs].sort_values(by=['capacity'], ascending=False)

    # initialize
    nr_of_users = len(users)
    nr_of_facs = len(facs)
    budget = round(budget_factor * nr_of_facs)
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    best_open_facs = sorted_users_and_facs_df.index[:budget]
    best_obj = sum(users_and_facs_df['capacity'])
    best_gap = (best_obj-lb)/best_obj
    best_assignment = {}
    it_ctr = 0
    open_facs = best_open_facs.copy()
    turnover = round(turnover_factor * budget)
    is_feasible = False

    # create dictionary of expected travelers
    travel_matrix = np.zeros((nr_of_users, nr_of_facs))
    for i in range(nr_of_users):
        for j in range(nr_of_facs):
            travel_matrix[i][j] = travel_dict[users[i]][facs[j]]
    exp_travelers_matrix = np.array(users_and_facs_df.loc[users]['population']).reshape(-1, 1) * travel_matrix
    exp_travelers_dict = {users[i]: {facs[j]: exp_travelers_matrix[i][j] for j in range(nr_of_facs)}
                          for i in range(nr_of_users)}

    while time()-start_time <= time_limit:
        # assign users
        is_feasible_assignment, assignment_results = greedily_assign_users(users_and_facs_df, travel_dict, users, facs,
                                                                           open_facs, exp_travelers_dict, cap_dict)

        # update best solution
        if is_feasible_assignment and assignment_results['obj'] < best_obj:
            is_feasible = True
            best_obj = assignment_results['obj']
            best_gap = (best_obj-lb)/best_obj
            best_open_facs = open_facs.copy()
            best_assignment = assignment_results['assignment']
            it_ctr = 0
        else:
            it_ctr += 1

        # check stop criteria
        if it_ctr == iteration_limit or best_gap < tolerance:
            break
        # update open facilities
        sorted_facs = dict(sorted(assignment_results['utilization'].items(), key=lambda item: item[1]))
        ditched_facs = list(sorted_facs.keys())[:turnover]
        ditched_zipcodes = [area for (area, fac) in assignment_results['assignment'].items() if fac in ditched_facs]
        open_facs = list(sorted_facs.keys())[turnover:]
        access = {j: sum(exp_travelers_dict[i][j] for i in ditched_zipcodes) for j in facs if j not in open_facs}
        sorted_access = dict(sorted(access.items(), key=lambda item: -item[1]))
        open_facs += list(sorted_access.keys())[:turnover]
        open_facs = pd.Index(open_facs)

    solving_time = time()-start_time
    if not is_feasible:
        print('no feasible solution could be constructed.')
        return is_feasible, {}

    # write dictionary with results
    results = {"solution_details":
                   {"assignment": best_assignment, "open_facs": list(best_open_facs), "objective_value": best_obj,
                    "lower_bound": lb, "solving_time": solving_time},
               "model_details":
                   {"users": users, "facs": facs, "cap_factor": cap_factor, "budget_factor": budget_factor,
                    "turnover_factor": turnover_factor, "tolerance": tolerance, "time_limit": time_limit,
                    "iteration_limit": iteration_limit}
               }
    return is_feasible, results


