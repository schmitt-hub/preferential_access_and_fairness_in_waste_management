"""
Module containing functions for optimizing the MIP
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from time import time
from copy import deepcopy
from utils import *


def exp_travelers_init(m, i, j):
    return m.users_and_facs_df.at[i, 'population'] * m.travel_dict[i][j]


def cap_init(m, j):
    return m.users_and_facs_df.at[j, 'capacity'] * m.cap_factor.value


def utilization_cstr(m, j):
    #if m.different_objective:
    #    return m.u[j] <= 1
    #else:
    return sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j) / m.cap[
    j] <= 1

def define_utilization(m,j):
    return m.u[j] == sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j) / m.cap[j]

def assign_to_one_cstr(m, i):
    if m.strict_assign_to_one:
        return sum(m.x[(i, j)] for (i2, j) in m.travel_pairs if i2 == i) == 1
    else:
        return sum(m.x[(i, j)] for (i2, j) in m.travel_pairs if i2 == i) <= 1


def assign_to_open_cstr(m, i, j):
    return m.y[j] >= m.x[(i, j)]

def min_access_cstr(m):
    return sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j) in m.travel_pairs) >= m.access * m.discount_factor

def obj_expression(m):
    if m.max_access:
        return sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j) in m.travel_pairs)
    elif m.different_objective:
         #return sum(m.cap[j] * m.y[j] * (
         #   1 - (sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j)
         #        / m.cap[j])) for j in m.facs)
        #  return sum((
        #     1 - (sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j)
        #          / m.cap[j])) ** 2 for j in m.facs)
        return sum(m.cap[j] * (
            1 - (sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j)
                 / m.cap[j])) ** 2 for j in m.facs) - sum(m.cap[j]*(1-m.y[j]) for j in m.facs)
    else:
        return sum(m.cap[j] * (
            1 - (sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j)
                 / m.cap[j])) ** 2 for j in m.facs)

# seperate objective needed since y_j not defined anymore
def obj_expression_post(m):
    if m.max_access:
        return sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j) in m.travel_pairs)
    elif m.different_objective:
        return sum(m.cap[j] * (
            1 - (sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j)
                 / m.cap[j])) ** 2 for j in m.assignable_facs)
    else:
        return sum(m.cap[j] * (
            1 - (sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j)
                 / m.cap[j])) ** 2 for j in m.facs)

def obj_expression_access_constrained(m):
    if m.different_objective:
        return sum(m.cap[j] * (
            1 - (sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j)
                 / m.cap[j])) ** 2 for j in m.facs) - sum(m.cap[j]*(1-m.y[j]) for j in m.facs)
    else:
        return sum(m.y[j] for j in m.assignable_facs)


def obj_sense(m):
    if m.max_access:
        return -1
    else:
        return 1


def budget_cstr(m):
    if m.different_objective:
        return pyo.summation(m.y) == m.budget
    #else:
    #if m.rural_budget >= 0:
    #    return pyo.summation(m.y[i] for i in m.rural_facs) <= m.rural_budget, pyo.summation(m.y[i] for i in m.urban_facs) <= m.budget - m.rural_budget
    else:
        return pyo.summation(m.y) <= m.budget

def budget_cstr_inequality(m):
    return pyo.summation(m.y) <= m.budget

def budget_cstr_reserve_rural(m):
    return sum(m.y[i] for i in m.rural_facs) >= m.rural_budget

def budget_cstr_rural(m):
    return sum(m.y[i] for i in m.rural_facs) <= m.rural_budget

def budget_cstr_urban(m):
    return sum(m.y[i] for i in m.urban_facs) <= m.budget - m.rural_budget

def update_budget(m, budget_factor):
    m.del_component(m.budget)
    m.budget = pyo.Param(initialize=round(budget_factor * len(list(m.facs))))
    m.del_component(m.budget_cstr)
    m.budget_cstr = pyo.Constraint(rule=budget_cstr)
    return m

def build_model_rural(users_and_facs_df, travel_dict, users, facs, strict_assign_to_one=False,
                cap_factor=1.5, cutoff=0.2, max_access=False, facs_to_open = "rural"):
    """
    build the main optimization model
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned
        to exactly 1 facility in the main model
    :param cap_factor: factor by which all capacities are scaled
    :param cutoff: travel combinations with a probability smaller than this value will be removed
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :param facs_to_open: "rural" or "urban" all open
    :param proportion_rural: proportion of the budget that should be rural facilities, -1 indicates no constraint on urban and rural seperately
    :return: model: the built model
    """
    print('Build model...')
    users_rural, facs_rural, users_urban, facs_urban = split_rural_urban(users_and_facs_df, users, facs)

    m = pyo.ConcreteModel()

    facs_open = facs_rural
    if facs_to_open == "rural":
        facs_open = facs_rural
    elif facs_to_open == "urban":
        facs_open = facs_urban

    # declare sets, variables and parameters
    m.users_and_facs_df = users_and_facs_df
    m.users = pyo.Set(initialize=users)
    m.facs = pyo.Set(initialize=facs)
    m.facs_open = pyo.Set(initialize=facs_rural)
    m.cutoff = pyo.Param(initialize=cutoff)
    m.travel_dict = travel_dict
    m.travel_pairs = pyo.Set(initialize=[(i, j) for i in m.users for j in m.facs_open
                                         if m.travel_dict[i][j] > m.cutoff.value])
    # the area-facility combinations whose travel willingness is higher than cutoff
    m.assignable_users = pyo.Set(initialize=list(set(i for (i, _) in m.travel_pairs)))
    # the users that have a facility to which the travel willingness is higher than cutoff
    m.assignable_facs = pyo.Set(initialize=list(set(j for (_, j) in m.travel_pairs)))
    # the facilities that have a user from which the travel willingness is higher than cutoff

    m.cap_factor = pyo.Param(initialize=cap_factor)
    m.cap = pyo.Param(m.facs, initialize=cap_init)
    m.exp_travelers = pyo.Param(m.travel_pairs, initialize=exp_travelers_init)
    m.strict_assign_to_one = pyo.Param(initialize=strict_assign_to_one)
    m.max_access = pyo.Param(initialize=max_access)
    m.obj_sense = pyo.Param(initialize=obj_sense)
    m.x = pyo.Var(m.travel_pairs, within=pyo.Binary)
    m.z = pyo.Var()

    # constraints and objective
    m.utilization_cstr = pyo.Constraint(m.assignable_facs, rule=utilization_cstr)
    m.assign_to_one_cstr = pyo.Constraint(m.assignable_users, rule=assign_to_one_cstr)
    m.assign_to_open_cstr = pyo.Constraint(m.travel_pairs, rule=assign_to_open_cstr)
    m.obj = pyo.Objective(rule=obj_expression, sense=m.obj_sense.value)
    print('Setup complete')
    return m
def build_model(users_and_facs_df, travel_dict, users, facs, budget_factor=1.0, strict_assign_to_one=False,
                cap_factor=1.5, cutoff=0.2, max_access=False, different_objective = False, proportion_rural = -1, closed_facilities_list = []):
    """
    build the main optimization model
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: ratio of facilities that are allowed to be opened
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned
        to exactly 1 facility in the main model
    :param cap_factor: factor by which all capacities are scaled
    :param cutoff: travel combinations with a probability smaller than this value will be removed
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :param different_objective: Malena trying out a different objective with no square but multiplied by y_j
    :param proportion_rural: proportion of the budget that should be rural facilities, -1 indicates no constraint on urban and rural seperately
    :param closed_facilities_list: list of facilities that we want to have closed
    :return: model: the built model
    """
    print('Build model...')
    users_rural, facs_rural, users_urban, facs_urban = split_rural_urban(users_and_facs_df, users, facs)

    m = pyo.ConcreteModel()

    # declare sets, variables and parameters
    m.users_and_facs_df = users_and_facs_df
    m.users = pyo.Set(initialize=users)
    m.facs = pyo.Set(initialize=facs)
    m.rural_facs = pyo.Set(initialize=facs_rural)
    m.urban_facs = pyo.Set(initialize=facs_urban)
    m.cutoff = pyo.Param(initialize=cutoff)
    m.travel_dict = travel_dict
    m.travel_pairs = pyo.Set(initialize=[(i, j) for i in m.users for j in m.facs
                                         if m.travel_dict[i][j] > m.cutoff.value and j not in closed_facilities_list])
    # the area-facility combinations whose travel willingness is higher than cutoff
    m.assignable_users = pyo.Set(initialize=list(set(i for (i, _) in m.travel_pairs)))
    # the users that have a facility to which the travel willingness is higher than cutoff
    m.assignable_facs = pyo.Set(initialize=list(set(j for (_, j) in m.travel_pairs)))
    # the facilities that have a user from which the travel willingness is higher than cutoff

    m.cap_factor = pyo.Param(initialize=cap_factor)
    m.cap = pyo.Param(m.facs, initialize=cap_init)
    m.exp_travelers = pyo.Param(m.travel_pairs, initialize=exp_travelers_init)
    m.budget_factor = pyo.Param(initialize=budget_factor)
    m.budget = pyo.Param(initialize=round(m.budget_factor * len(list(m.facs))))
    m.proportion_rural = pyo.Param(initialize=proportion_rural)
    m.rural_budget = pyo.Param(initialize=round(m.budget*m.proportion_rural))
    m.strict_assign_to_one = pyo.Param(initialize=strict_assign_to_one)
    m.max_access = pyo.Param(initialize=max_access)
    m.obj_sense = pyo.Param(initialize=obj_sense)
    m.different_objective = pyo.Param(initialize=different_objective)
    m.y = pyo.Var(m.facs, bounds=(0, 1))
    m.x = pyo.Var(m.travel_pairs, within=pyo.Binary)
    m.z = pyo.Var()

    # to fix certain facilities to be closed and open
    for j in closed_facilities_list:
        m.y[j].fix(0)
    if len(closed_facilities_list) > 0:
        #sm.budget_factor = pyo.Param(initialize=1-len(closed_facilities_list)/len(facs))
        #m.budget = pyo.Param(initialize=round(m.budget_factor * len(list(m.facs))))
        for j in facs:
            if j not in closed_facilities_list:
                m.y[j].fix(1)

    # constraints and objective
    m.utilization_cstr = pyo.Constraint(m.assignable_facs, rule=utilization_cstr)
    m.budget_cstr = pyo.Constraint(rule=budget_cstr)
    if m.rural_budget > 0:
        m.budget_cstr_rural_reservation = pyo.Constraint(rule=budget_cstr_reserve_rural)
    m.assign_to_one_cstr = pyo.Constraint(m.assignable_users, rule=assign_to_one_cstr)
    m.assign_to_open_cstr = pyo.Constraint(m.travel_pairs, rule=assign_to_open_cstr)
    m.obj = pyo.Objective(rule=obj_expression, sense=m.obj_sense.value)
    print('Setup complete')
    return m


def optimize_model(m, threads=1, tolerance=5e-3, time_limit=20000, print_sol=False, log_file=None, preqlinearize=-1):
    """
    solve the main optimization step
    :param m: the main model
    :param threads: number of threads
    :param tolerance: tolerance on the optimality of the solution
    :param time_limit: time limit in seconds for the optimization
    :param print_sol: boolean indicating whether the solution should be printed
    :param log_file: name of the log file. If "None", no log file will be produced
    :param preqlinearize: setting for Gurobi's PreQLinearize parameter
    :return: dictionary of results
    """

    print("Optimize model...")
    opt = pyo.SolverFactory("gurobi")
    opt.options["threads"] = threads
    opt.options["MIPGap"] = tolerance
    opt.options["Time_limit"] = time_limit
    if log_file is not None:
        opt.options["Logfile"] = log_file
    opt.options["PreQLinearize"] = preqlinearize
    res = opt.solve(m, tee=True, warmstart=True)

    if res.solver.termination_condition == TerminationCondition.infeasible:
        return False, {}

    if print_sol:
        # print the decision variables that are used in the solution and the objective value
        print('Open facilities:')
        for j in m.facs:
            if m.y[j].value > 1e-4:
                print(j, m.y[j].value)
                if j not in m.assignable_facs:
                    print('Facility', j, 'with utilization: 0')
                else:
                    print('Facility', j, 'with utilization:', 1 - m.utilization_cstr[j].uslack())
        print()
        print('Selected travel pairs:')
        for k in m.travel_pairs:
            if m.x[k].value > 1e-4:
                print('Travel pair', k)
        print()
        print('Objective value:', pyo.value(m.obj))
        print()

    # write dictionary with results
    results = {"solution_details":
                   {"assignment": {}, "open_facs": [], "objective_value": pyo.value(m.obj),
                    "lower_bound": None, "solving_time": res.Solver[0]['Time']},
               "model_details":
                   {"users": list(m.users), "facs": list(m.facs), "cap_factor": m.cap_factor.value,
                    "budget_factor": m.budget_factor.value, "proportion_rural": m.proportion_rural.value, "cutoff": m.cutoff.value,
                    "strict_assign_to_one": m.strict_assign_to_one.value, "tolerance": tolerance,
                    "time_limit": time_limit}
               }

    unassigned_ctr = 0
    for j in m.facs:
        if m.y[j].value > 1e-4:
            results["solution_details"]["open_facs"].append(j)
    for i in m.users:
        if i not in m.assignable_users:
            results["solution_details"]["assignment"][i] = None
            unassigned_ctr += 1
        else:
            assignment_found = False
            for j in m.assignable_facs:
                if (i, j) in m.travel_pairs and m.x[(i, j)].value > 1e-4:
                    results["solution_details"]["assignment"][i] = j
                    assignment_found = True
                    break
            if not assignment_found:
                results["solution_details"]["assignment"][i] = None
                unassigned_ctr += 1
    print(unassigned_ctr, 'users need postprocessing')
    # the lower bound returned by the solver is only a lower bound for the full model when no postprocessing step
    # is necessary
    if unassigned_ctr == 0:
        results["solution_details"]["lower_bound"] = res['Problem'][0]['Lower bound']
    return True, results


def build_postprocessing_model(users_and_facs_df, travel_dict, users, facs,
                        cap_factor, max_access, different_objective, results_mainstep, continous = False):
    """
    build the post processing model
    :param model: the main model
    :param results_mainstep: dictionary of the results of the main optimization step
    :return: the built post processing model
    """
    print('Build postprocessing model...')
    start = time()
    m = pyo.ConcreteModel()
    m.assignment = results_mainstep['solution_details']["assignment"]
    m.users_and_facs_df = users_and_facs_df
    m.travel_dict = travel_dict

    # declare sets, variables and parameters
    m.users = pyo.Set(initialize=users)
    m.assignable_users = pyo.Set(initialize=[int(i) for i in m.assignment if m.assignment[i] is None])
    m.facs = pyo.Set(initialize=facs)
    m.assignable_facs = pyo.Set(initialize=results_mainstep['solution_details']["open_facs"])
    m.cutoff = pyo.Param(initialize=0.0)
    m.travel_pairs = pyo.Set(initialize=[(i, j) for i in m.assignable_users for j in m.assignable_facs
                                         if m.travel_dict[i][j] > m.cutoff.value] +
                                        [(i, j) for (i, j) in m.assignment.items() if j is not None])

    m.cap_factor = pyo.Param(initialize=cap_factor)
    m.cap = pyo.Param(m.facs, initialize=cap_init)
    m.exp_travelers = pyo.Param(m.travel_pairs, initialize=exp_travelers_init)
    m.strict_assign_to_one = pyo.Param(initialize=True)
    m.max_access = pyo.Param(initialize=max_access)
    m.obj_sense = pyo.Param(initialize=obj_sense)
    m.continous = pyo.Param(initialize=continous)
    m.different_objective = pyo.Param(initialize=different_objective)
    if continous:
        m.x = pyo.Var(m.travel_pairs, bounds=(0,1))
    else:
        m.x = pyo.Var(m.travel_pairs, within=pyo.Binary)

    # fix assignments that have been made before
    for (i, j) in m.assignment.items():
        if j is not None:
            m.x[(i, j)].fix(1)

    m.utilization_cstr = pyo.Constraint(m.assignable_facs, rule=utilization_cstr)
    m.assign_to_one_cstr = pyo.Constraint(m.assignable_users, rule=assign_to_one_cstr)
    m.obj = pyo.Objective(rule=obj_expression_post, sense=m.obj_sense.value)
    print('setup complete in ', time() - start, 'seconds')
    return m


def optimize_postprocessing_model(m, results_mainstep, threads=1, tolerance=0.0, print_sol=False, log_file=None,
                                  preqlinearize=-1):
    """
    solve the postprocessing model that contains only the unassigned zip code areas and the open facilities;
    all constraints related to the facilities are removed;
    the objective value is the objective value of the full model
    :param m: the model to be optimized
    :param results_mainstep: dictionary of results of the main optimization step
    :param threads: number of threads
    :param tolerance: tolerance on the optimality of the solution
    :param print_sol: boolean indicating whether the solution should be printed
    :param log_file: name of the log file. If "None", no log file will be produced
    :param preqlinearize: setting for Gurobi's PreQLinearize parameter
    :return: boolean indicating whether the model is feasible; updated results dictionary
    """
    print('Optimize post processing model...')
    opt = pyo.SolverFactory("gurobi")
    opt.options["threads"] = threads
    opt.options["MIPGap"] = tolerance
    opt.options["NodefileStart"] = 0.5
    opt.options["Time_limit"] = 5000

    if log_file is not None:
        opt.options["Logfile"] = log_file
    opt.options["PreQLinearize"] = preqlinearize
    # solve with enabling logging of solver
    #TODO: set tee back to true
    res = opt.solve(m, tee=True)

    if res.solver.termination_condition == TerminationCondition.infeasible:
        return False, {}

    if print_sol:
        # print the decision variables that are used in the solution and the objective value
        print('Open facilities:')
        for j in m.assignable_facs:
            print('Facility', j, 'with utilization:', 1 - m.utilization_cstr[j].uslack())
        print()
        print('Selected travel pairs:')
        for k in m.travel_pairs:
            if m.x[k].value > 1e-4:
                print('Travel pair', k)
        print()
        print('Objective value:', pyo.value(m.obj))
        print()

    # create dictionary for the results after postprocessing
    results = deepcopy(results_mainstep)
    results["solution_details"]["objective_value"] = pyo.value(m.obj)
    results["solution_details"]["solving_time"] += res.Solver[0]['Time']
    for i in m.assignable_users:
        for j in m.assignable_facs:
            if m.x[(i, j)].value > 1e-4:
                results["solution_details"]["assignment"][i] = j
    return True, results


def solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor=1.0, strict_assign_to_one=False,
                        cap_factor=1.5, cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3,
                        main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                        post_threads=1, post_tolerance=0.0, post_print_sol=False, post_log_file=None,
                        post_preqlinearize=-1, different_objective = False, proportion_rural = -1, closed_facilities_list = [], continous = False):
    """
    solve the optimization model naively
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: ratio of facilities that are allowed to be opened
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned
        to exactly 1 facility in the main model
    :param cap_factor: factor by which all capacities are scaled
    :param cutoff: travel combinations with a probability smaller than this value will be removed
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :param main_threads: number of threads used in the main optimization step
    :param main_tolerance: tolerance on the optimality of the solution used in the main optimization step
    :param main_time_limit: time limit in seconds for the optimization used in the main optimization step
    :param main_print_sol: boolean indicating whether the solution should be printed used in the main optimization step
    :param main_log_file: name of the log file for the main optimization step; if "None", no log file will be produced
    :param main_preqlinearize: setting for Gurobi's PreQLinearize parameter in the main optimization step
    :param post_threads: number of threads used in the post processing step
    :param post_tolerance: tolerance on the optimality of the solution used in the post processing step
    :param post_print_sol: boolean indicating whether the solution should be printed used in the post processing step
    :param post_log_file: name of the log file for the post processing step; if "None", no log file will be produced
    :param post_preqlinearize: setting for Gurobi's PreQLinearize parameter used in the post processing step
    :param different_objective: Malena trying out a different objective with no square but multiplied by y_j
    :param proportion_rural: proportion of the budget that should be rural facilities, -1 indicates no constraint on urban and rural seperately
    :param proportion_rural: proportion of the budget that should be rural facilities, -1 indicates no constraint on urban and rural seperately
    :return: dictionary containing the results
    """
    model = build_model(users_and_facs_df, travel_dict, users, facs, budget_factor, strict_assign_to_one,
                        cap_factor, cutoff, max_access, different_objective, proportion_rural, closed_facilities_list)
    is_feasible, results = optimize_model(model, main_threads, main_tolerance, main_time_limit, main_print_sol,
                                          main_log_file, main_preqlinearize)
    del(model)
    # TODO: debugging...delete
    #print(results)
    if not is_feasible:
        print('The model is infeasible')
        return is_feasible, {}
    if None in results['solution_details']['assignment'].values():
        print()
        print('Postprocessing...')
        post_model = build_postprocessing_model(users_and_facs_df, travel_dict, users, facs,
                        cap_factor, max_access, different_objective, results, continous)
        is_feasible, results = optimize_postprocessing_model(post_model, results, post_threads, post_tolerance,
                                                             post_print_sol, post_log_file, post_preqlinearize)
        if not is_feasible:
            print('The post processing model is infeasible')
            return is_feasible, {}
    return is_feasible, results

def build_adapted_access_model(users_and_facs_df, travel_dict, users, facs, budget_factor, access, discount_factor=0.95, strict_assign_to_one=False,
                cap_factor=1.5, cutoff=0.2, different_objective = False):
    """
    build the adapted model where we are given an overall access we almost want to achieve
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param access: the access we want this solution to (almost) achieve, i.e. sum_i,jU_iP_ijx_ij
    :param discount_factor: how much deviation to the access we want to achieve we allow
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned
        to exactly 1 facility in the main model
    :param cap_factor: factor by which all capacities are scaled
    :param cutoff: travel combinations with a probability smaller than this value will be removed
    :param different_objective: False means minimising facilities in first, with C_j objective in secondary; True means only open facilities count in both
    """
    print('Build model...')

    m = pyo.ConcreteModel()

    # declare sets, variables and parameters
    m.users_and_facs_df = users_and_facs_df
    m.users = pyo.Set(initialize=users)
    m.facs = pyo.Set(initialize=facs)
    m.cutoff = pyo.Param(initialize=cutoff)
    m.travel_dict = travel_dict
    m.travel_pairs = pyo.Set(initialize=[(i, j) for i in m.users for j in m.facs
                                         if m.travel_dict[i][j] > m.cutoff.value])
    # the area-facility combinations whose travel willingness is higher than cutoff
    m.assignable_users = pyo.Set(initialize=list(set(i for (i, _) in m.travel_pairs)))
    # the users that have a facility to which the travel willingness is higher than cutoff
    m.assignable_facs = pyo.Set(initialize=list(set(j for (_, j) in m.travel_pairs)))
    # the facilities that have a user from which the travel willingness is higher than cutoff

    m.cap_factor = pyo.Param(initialize=cap_factor)
    m.cap = pyo.Param(m.facs, initialize=cap_init)
    m.exp_travelers = pyo.Param(m.travel_pairs, initialize=exp_travelers_init)
    m.strict_assign_to_one = pyo.Param(initialize=strict_assign_to_one)
    m.access = pyo.Param(initialize=access)
    m.discount_factor = pyo.Param(initialize=discount_factor)
    m.obj_sense = pyo.Param(initialize=1)
    m.different_objective = pyo.Param(initialize=different_objective)
    m.budget_factor = pyo.Param(initialize=budget_factor)
    m.budget = pyo.Param(initialize=round(m.budget_factor * len(list(m.facs))))
    m.y = pyo.Var(m.facs, bounds=(0, 1))
    m.x = pyo.Var(m.travel_pairs, within=pyo.Binary)
    m.z = pyo.Var()

    # constraints and objective
    m.utilization_cstr = pyo.Constraint(m.assignable_facs, rule=utilization_cstr)
    # edited to allow setting seperate budgets for rural and urban
    m.assign_to_one_cstr = pyo.Constraint(m.assignable_users, rule=assign_to_one_cstr)
    m.assign_to_open_cstr = pyo.Constraint(m.travel_pairs, rule=assign_to_open_cstr)
    m.min_access_cstr = pyo.Constraint(rule=min_access_cstr)
    m.budget_cstr_inequality = pyo.Constraint(rule=budget_cstr_inequality)
    m.obj = pyo.Objective(rule=obj_expression_access_constrained, sense=m.obj_sense.value)
    print('Setup complete')
    return m

def optimize_adapted_access_model(m, threads=1, tolerance=5e-3, time_limit=20000, print_sol=False, log_file=None, preqlinearize=-1):
    """
    solve the main optimization step
    :param m: the main model
    :param threads: number of threads
    :param tolerance: tolerance on the optimality of the solution
    :param time_limit: time limit in seconds for the optimization
    :param print_sol: boolean indicating whether the solution should be printed
    :param log_file: name of the log file. If "None", no log file will be produced
    :param preqlinearize: setting for Gurobi's PreQLinearize parameter
    :return: dictionary of results
    """

    print("Optimize model...")
    opt = pyo.SolverFactory("gurobi")
    opt.options["threads"] = threads
    opt.options["MIPGap"] = tolerance
    opt.options["Time_limit"] = time_limit
    if log_file is not None:
        opt.options["Logfile"] = log_file
    opt.options["PreQLinearize"] = preqlinearize
    res = opt.solve(m, tee=True, warmstart=True)

    if res.solver.termination_condition == TerminationCondition.infeasible:
        return False, {}

    if print_sol:
        # print the decision variables that are used in the solution and the objective value
        print('Open facilities:')
        for j in m.facs:
            if m.y[j].value > 1e-4:
                print(j, m.y[j].value)
                if j not in m.assignable_facs:
                    print('Facility', j, 'with utilization: 0')
                else:
                    print('Facility', j, 'with utilization:', 1 - m.utilization_cstr[j].uslack())
        print()
        print('Selected travel pairs:')
        for k in m.travel_pairs:
            if m.x[k].value > 1e-4:
                print('Travel pair', k)
        print()
        print('Objective value:', pyo.value(m.obj))
        print()

    # write dictionary with results
    results = {"solution_details":
                   {"assignment": {}, "open_facs": [], "objective_value": pyo.value(m.obj),
                    "lower_bound": None, "solving_time": res.Solver[0]['Time']},
               "model_details":
                   {"users": list(m.users), "facs": list(m.facs), "cap_factor": m.cap_factor.value, "budget_factor": m.budget_factor.value,
                    "access": m.access.value, "discount_factor": m.discount_factor.value, "cutoff": m.cutoff.value,
                    "strict_assign_to_one": m.strict_assign_to_one.value, "tolerance": tolerance,
                    "time_limit": time_limit}
               }

    unassigned_ctr = 0
    for j in m.facs:
        if m.y[j].value > 1e-4:
            results["solution_details"]["open_facs"].append(j)
    for i in m.users:
        if i not in m.assignable_users:
            results["solution_details"]["assignment"][i] = None
            unassigned_ctr += 1
        else:
            assignment_found = False
            for j in m.assignable_facs:
                if (i, j) in m.travel_pairs and m.x[(i, j)].value > 1e-4:
                    results["solution_details"]["assignment"][i] = j
                    assignment_found = True
                    break
            if not assignment_found:
                results["solution_details"]["assignment"][i] = None
                unassigned_ctr += 1
    print(unassigned_ctr, 'users need postprocessing')
    # the lower bound returned by the solver is only a lower bound for the full model when no postprocessing step
    # is necessary
    if unassigned_ctr == 0:
        results["solution_details"]["lower_bound"] = res['Problem'][0]['Lower bound']
    return True, results

def solve_model_with_access_bound(users_and_facs_df, travel_dict, users, facs, budget_factor, access, discount_factor=0.95, strict_assign_to_one=False,
                        cap_factor=1.5, cutoff=0.2, main_threads=1, main_tolerance=5e-3,
                        main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                        post_threads=1, post_tolerance=2e-3, post_print_sol=False, post_log_file=None,
                        post_preqlinearize=-1, different_objective = False):
    """
    solve the optimization model with certain access as a goal
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param access: the access we want this solution to (almost) achieve, i.e. sum_i,jU_iP_ijx_ij
    :param discount_factor: how much deviation to the access we want to achieve we allow
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned
        to exactly 1 facility in the main model
    :param cap_factor: factor by which all capacities are scaled
    :param cutoff: travel combinations with a probability smaller than this value will be removed
    :param main_threads: number of threads used in the main optimization step
    :param main_tolerance: tolerance on the optimality of the solution used in the main optimization step
    :param main_time_limit: time limit in seconds for the optimization used in the main optimization step
    :param main_print_sol: boolean indicating whether the solution should be printed used in the main optimization step
    :param main_log_file: name of the log file for the main optimization step; if "None", no log file will be produced
    :param main_preqlinearize: setting for Gurobi's PreQLinearize parameter in the main optimization step
    :param post_threads: number of threads used in the post processing step
    :param post_tolerance: tolerance on the optimality of the solution used in the post processing step
    :param post_print_sol: boolean indicating whether the solution should be printed used in the post processing step
    :param post_log_file: name of the log file for the post processing step; if "None", no log file will be produced
    :param post_preqlinearize: setting for Gurobi's PreQLinearize parameter used in the post processing step
    :param different_objective: False means minimising facilities in first, with C_j objective in secondary; True means only open facilities count in both
    :return: dictionary containing the results
    """
    model = build_adapted_access_model(users_and_facs_df, travel_dict, users, facs, budget_factor, access, discount_factor, strict_assign_to_one,
                cap_factor, cutoff, different_objective)
    is_feasible, results = optimize_adapted_access_model(model, main_threads, main_tolerance, main_time_limit, main_print_sol,
                                          main_log_file, main_preqlinearize)
    del(model)
    if not is_feasible:
        print('The model is infeasible')
        return is_feasible, {}
    if None in results['solution_details']['assignment'].values():
        print()
        print('Postprocessing...')
        post_model = build_postprocessing_model(users_and_facs_df, travel_dict, users, facs,
                        cap_factor, max_access=False, different_objective=different_objective, results_mainstep=results, continous = False)
        is_feasible, results = optimize_postprocessing_model(post_model, results, post_threads, post_tolerance,
                                                             post_print_sol, post_log_file, post_preqlinearize)
        if not is_feasible:
            print('The post processing model is infeasible')
            return is_feasible, {}
    return is_feasible, results

def solve_main_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor=1.0, strict_assign_to_one=False,
                        cap_factor=1.5, cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3,
                        main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                        post_threads=1, post_tolerance=0.0, post_print_sol=False, post_log_file=None,
                        post_preqlinearize=-1, different_objective = False, proportion_rural = -1):
    """
    solve the optimization model naively
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: ratio of facilities that are allowed to be opened
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned
        to exactly 1 facility in the main model
    :param cap_factor: factor by which all capacities are scaled
    :param cutoff: travel combinations with a probability smaller than this value will be removed
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :param main_threads: number of threads used in the main optimization step
    :param main_tolerance: tolerance on the optimality of the solution used in the main optimization step
    :param main_time_limit: time limit in seconds for the optimization used in the main optimization step
    :param main_print_sol: boolean indicating whether the solution should be printed used in the main optimization step
    :param main_log_file: name of the log file for the main optimization step; if "None", no log file will be produced
    :param main_preqlinearize: setting for Gurobi's PreQLinearize parameter in the main optimization step
    :param post_threads: number of threads used in the post processing step
    :param post_tolerance: tolerance on the optimality of the solution used in the post processing step
    :param post_print_sol: boolean indicating whether the solution should be printed used in the post processing step
    :param post_log_file: name of the log file for the post processing step; if "None", no log file will be produced
    :param post_preqlinearize: setting for Gurobi's PreQLinearize parameter used in the post processing step
    :param different_objective: Malena trying out a different objective with no square but multiplied by y_j
    :param proportion_rural: proportion of the budget that should be rural facilities, -1 indicates no constraint on urban and rural seperately
    :param closed_facilities_list: facilities we want to close
    :return: dictionary containing the results
    """
    model = build_model(users_and_facs_df, travel_dict, users, facs, budget_factor, strict_assign_to_one,
                        cap_factor, cutoff, max_access, different_objective, proportion_rural, closed_facilities_list)
    is_feasible, results = optimize_model(model, main_threads, main_tolerance, main_time_limit, main_print_sol,
                                          main_log_file, main_preqlinearize)
    return is_feasible, results

def build_facilities_open_fixed_model(users_and_facs_df, travel_dict, users, facs,
                        cap_factor, max_access, cutoff, open_facs, different_objective = False, continous = False):
    """
    build the model where we know which facilities are open
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param cap_factor: factor by which all capacities are scaled
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :return: the built model
    """
    print('Build model...')
    start = time()
    m = pyo.ConcreteModel()
    m.users_and_facs_df = users_and_facs_df
    m.travel_dict = travel_dict

    # declare sets, variables and parameters
    m.users = pyo.Set(initialize=users)
    m.facs = pyo.Set(initialize=facs)
    m.assignable_users = pyo.Set(initialize=users)
    m.assignable_facs = pyo.Set(initialize=open_facs)
    m.cutoff = pyo.Param(initialize=cutoff)
    m.travel_pairs = pyo.Set(initialize=[(i, j) for i in m.users for j in m.assignable_facs
                                         if m.travel_dict[i][j] > m.cutoff.value])

    m.cap_factor = pyo.Param(initialize=cap_factor)
    m.cap = pyo.Param(m.facs, initialize=cap_init)
    m.exp_travelers = pyo.Param(m.travel_pairs, initialize=exp_travelers_init)
    m.strict_assign_to_one = pyo.Param(initialize=True)
    m.max_access = pyo.Param(initialize=max_access)
    m.obj_sense = pyo.Param(initialize=obj_sense)
    m.continous = pyo.Param(initialize=continous)
    m.different_objective = pyo.Param(initialize=different_objective)
    if continous:
        m.x = pyo.Var(m.travel_pairs, bounds=(0,1))
    else:
        m.x = pyo.Var(m.travel_pairs, within=pyo.Binary)
    if different_objective:
        m.u = pyo.Var(m.assignable_facs, bounds=(0, 1))

    # constraints and objective
    if m.different_objective:
        m.define_utilization = pyo.Constraint(m.assignable_facs, rule=define_utilization)
    m.utilization_cstr = pyo.Constraint(m.assignable_facs, rule=utilization_cstr)
    m.assign_to_one_cstr = pyo.Constraint(m.assignable_users, rule=assign_to_one_cstr)
    m.obj = pyo.Objective(rule=obj_expression, sense=m.obj_sense.value)
    print('setup complete in ', time() - start, 'seconds')

    return m

def optimize_facilities_open_fixed_model(m, threads=1, tolerance=0.0, print_sol=False, log_file=None,
                                  preqlinearize=-1, time_limit = 10000, output_file = "test.json"):
    """
    solve the postprocessing model that contains only the unassigned zip code areas and the open facilities;
    all constraints related to the facilities are removed;
    the objective value is the objective value of the full model
    :param m: the model to be optimized
    :param threads: number of threads
    :param tolerance: tolerance on the optimality of the solution
    :param print_sol: boolean indicating whether the solution should be printed
    :param log_file: name of the log file. If "None", no log file will be produced
    :param preqlinearize: setting for Gurobi's PreQLinearize parameter
    :param time_limit: how long the model is allowed to run for
    :return: boolean indicating whether the model is feasible; updated results dictionary
    """
    print('Optimize fixed open facilities model...')
    opt = pyo.SolverFactory("gurobi")
    opt.options["threads"] = threads
    opt.options["MIPGap"] = tolerance
    opt.options["NodefileStart"] = 0.5
    opt.options["Time_limit"] = time_limit

    if log_file is not None:
        opt.options["Logfile"] = log_file
    opt.options["PreQLinearize"] = preqlinearize
    # solve with enabling logging of solver
    res = opt.solve(m, tee=True)

    if res.solver.termination_condition == TerminationCondition.infeasible:
        return False, {}

    if print_sol:
        # print the decision variables that are used in the solution and the objective value
        print('Open facilities:')
        for j in m.assignable_facs:
            print('Facility', j, 'with utilization:', 1 - m.utilization_cstr[j].uslack())
        print()
        print('Selected travel pairs:')
        for k in m.travel_pairs:
            if m.x[k].value > 1e-4:
                print('Travel pair', k)
        print()
        print('Objective value:', pyo.value(m.obj))
        print()
        
    dictionary_result = {}
    for (i,j) in m.travel_pairs:
        dictionary_result[str((i,j))] = m.x[(i,j)].value
    outfile = os.getcwd() + "/own_results/" + output_file
    with open(outfile, 'w') as f:
        json.dump(dictionary_result, f)

    # create dictionary for the results after postprocessing
    results = {}
    if m.continous:
        results = {i : {j: m.x[(i,j)].value for (k,j) in m.travel_pairs if k == i}for i in m.users}
    else: 
        assignment = {}
        for i in m.assignable_users:
            for j in m.assignable_facs:
                if (i, j) in m.travel_pairs and m.x[(i, j)].value > 1e-4:
                    assignment[i] = j
        results = {"solution_details":
                    {"assignment": assignment, "open_facs": list(m.assignable_facs), "objective_value": pyo.value(m.obj),
                        "lower_bound": res['Problem'][0]['Lower bound'], "solving_time": res.Solver[0]['Time']},
                "model_details":
                    {"users": list(m.users), "facs": list(m.facs), "cap_factor": m.cap_factor.value,
                        "budget_factor": len(m.assignable_facs)/len(m.facs), "proportion_rural": -1, "cutoff": m.cutoff.value,
                        "strict_assign_to_one": m.strict_assign_to_one.value, "tolerance": tolerance,
                        "time_limit": time_limit}
                }
    return True, results

def solve_facilities_open_fixed_model(users_and_facs_df, travel_dict, users, facs,
                        cap_factor=1.5, max_access=False, threads=1, tolerance=2e-3,cutoff = 0.1,
                        time_limit=20000, print_sol=False, log_file=None,
                        preqlinearize=-1, open_facs = [], different_objective = False, continous = False, output_file = "test.json"):
    """
    solve the optimization model with fixed open facilities
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param cap_factor: factor by which all capacities are scaled
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :param threads: number of threads used in the main optimization step
    :param tolerance: tolerance on the optimality of the solution used in the main optimization step
    :param time_limit: time limit in seconds for the optimization used in the main optimization step
    :param print_sol: boolean indicating whether the solution should be printed used in the main optimization step
    :param log_file: name of the log file for the main optimization step; if "None", no log file will be produced
    :param preqlinearize: setting for Gurobi's PreQLinearize parameter in the main optimization step
    :param different_objective: Malena trying out a different objective with no square but multiplied by y_j
    :return: dictionary containing the results
    """
    model =  build_facilities_open_fixed_model(users_and_facs_df, travel_dict, users, facs,
                        cap_factor, max_access,cutoff, open_facs, different_objective, continous)
    is_feasible, results = optimize_facilities_open_fixed_model(model, threads, tolerance, print_sol, log_file,
                                  preqlinearize, time_limit, output_file)
    if not is_feasible:
        print('The model is infeasible')
        return is_feasible, {}
    return is_feasible, results

def solve_postprocessing_model_naively(users_and_facs_df, travel_dict, users, facs, results, budget_factor=1.0, strict_assign_to_one=False,
                        cap_factor=1.5, cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3,
                        main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                        post_threads=1, post_tolerance=0.0, post_print_sol=False, post_log_file=None,
                        post_preqlinearize=-1, different_objective = False, proportion_rural = -1):
    """
    solve the optimization model naively
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: ratio of facilities that are allowed to be opened
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned
        to exactly 1 facility in the main model
    :param cap_factor: factor by which all capacities are scaled
    :param cutoff: travel combinations with a probability smaller than this value will be removed
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :param main_threads: number of threads used in the main optimization step
    :param main_tolerance: tolerance on the optimality of the solution used in the main optimization step
    :param main_time_limit: time limit in seconds for the optimization used in the main optimization step
    :param main_print_sol: boolean indicating whether the solution should be printed used in the main optimization step
    :param main_log_file: name of the log file for the main optimization step; if "None", no log file will be produced
    :param main_preqlinearize: setting for Gurobi's PreQLinearize parameter in the main optimization step
    :param post_threads: number of threads used in the post processing step
    :param post_tolerance: tolerance on the optimality of the solution used in the post processing step
    :param post_print_sol: boolean indicating whether the solution should be printed used in the post processing step
    :param post_log_file: name of the log file for the post processing step; if "None", no log file will be produced
    :param post_preqlinearize: setting for Gurobi's PreQLinearize parameter used in the post processing step
    :param different_objective: Malena trying out a different objective with no square but multiplied by y_j
    :param proportion_rural: proportion of the budget that should be rural facilities, -1 indicates no constraint on urban and rural seperately
    :return: dictionary containing the results
    """
    if None in results['solution_details']['assignment'].values():
        print()
        print('Postprocessing...')
        post_model = build_postprocessing_model(users_and_facs_df, travel_dict, users, facs,
                        cap_factor, max_access, different_objective, results)
        is_feasible, results = optimize_postprocessing_model(post_model, results, post_threads, post_tolerance,
                                                             post_print_sol, post_log_file, post_preqlinearize)
        if not is_feasible:
            print('The post processing model is infeasible')
            return is_feasible, {}
    return is_feasible, results