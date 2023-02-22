"""
module containing functions for creating excel tables from a given results file
and functions for creating excel tables after creating the corresponding results
"""

from utils import *
from model import *
# from greedy_heuristic import *
import os
import numpy as np
import pandas as pd
import random
import statistics


def write_overall_results(results_list, users_and_facs_df, travel_dict, output_filename='overall_results.xlsx',
                          output_abs_path=None, percentiles=None):
    """
    create an excel file that displays broad results for different budgets
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param percentiles: list of the three considered percentiles. Values must be between 0 and 100
    """

    # init
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]

    # make sure the results are sorted in decreasing order by budget
    # TODO: undo change
    sort_index = 'proportion_rural'
    results_list = sorted(results_list, key=lambda d: -d['model_details'][sort_index])
    nr_of_results = len(results_list)

    # get relevant data
    budget_factor_list = [round(results['model_details'][sort_index] * 100) for results in results_list]
    budget_string_list = [str(budget_factor_list[i // 4]) + "%" if i % 4 == 0 else None for i in
                          range(nr_of_results * 4)]
    regions = ['all', 'rural', 'urban', None] * nr_of_results
    access = []
    for results in results_list:
        access += [get_overall_access(results, users_and_facs_df, travel_dict, user_region='all') * 100,
                   get_overall_access(results, users_and_facs_df, travel_dict, user_region='rural') * 100,
                   get_overall_access(results, users_and_facs_df, travel_dict, user_region='urban') * 100,
                   None]
    distances_all = [list(get_distances_to_assigned(results, users_and_facs_df, user_region='all').values())
                     for results in results_list]
    distances_rural = [list(get_distances_to_assigned(results, users_and_facs_df, user_region='rural').values())
                       for results in results_list]
    distances_urban = [list(get_distances_to_assigned(results, users_and_facs_df, user_region='urban').values())
                       for results in results_list]
    distances_1 = []
    for i in range(nr_of_results):
        distances_1 += [safe_percentile(distances_all[i], percentiles[0]),
                        safe_percentile(distances_rural[i], percentiles[0]),
                        safe_percentile(distances_urban[i], percentiles[0]),
                        None]
    distances_2 = []
    for i in range(nr_of_results):
        distances_2 += [safe_percentile(distances_all[i], percentiles[1]),
                        safe_percentile(distances_rural[i], percentiles[1]),
                        safe_percentile(distances_urban[i], percentiles[1]),
                        None]
    distances_3 = []
    for i in range(nr_of_results):
        distances_3 += [safe_percentile(distances_all[i], percentiles[2]),
                        safe_percentile(distances_rural[i], percentiles[2]),
                        safe_percentile(distances_urban[i], percentiles[2]),
                        None]
    utilization_all = [list(get_utilization(results, users_and_facs_df, travel_dict, facility_region='all').values())
                       for results in results_list]
    utilization_rural = [
        list(get_utilization(results, users_and_facs_df, travel_dict, facility_region='rural').values())
        for results in results_list]
    utilization_urban = [
        list(get_utilization(results, users_and_facs_df, travel_dict, facility_region='urban').values())
        for results in results_list]
    utilization_1 = []
    for i in range(nr_of_results):
        utilization_1 += [safe_percentile(utilization_all[i], percentiles[0]) * 100,
                          safe_percentile(utilization_rural[i], percentiles[0]) * 100,
                          safe_percentile(utilization_urban[i], percentiles[0]) * 100,
                          None]
    utilization_2 = []
    for i in range(nr_of_results):
        utilization_2 += [safe_percentile(utilization_all[i], percentiles[1]) * 100,
                          safe_percentile(utilization_rural[i], percentiles[1]) * 100,
                          safe_percentile(utilization_urban[i], percentiles[1]) * 100,
                          None]
    utilization_3 = []
    for i in range(nr_of_results):
        utilization_3 += [safe_percentile(utilization_all[i], percentiles[2]) * 100,
                          safe_percentile(utilization_rural[i], percentiles[2]) * 100,
                          safe_percentile(utilization_urban[i], percentiles[2]) * 100,
                          None]

    # write the excel file
    data = {'Budget': budget_string_list, 'Region': regions, 'Overall access [%]': access,
            'Travel distance [km] - p' + str(percentiles[0]): distances_1,
            'Travel distance [km] - p' + str(percentiles[1]): distances_2,
            'Travel distance [km] - p' + str(percentiles[2]): distances_3,
            'Utilization [%] - p' + str(percentiles[0]): utilization_1,
            'Utilization [%] - p' + str(percentiles[1]): utilization_2,
            'Utilization [%] - p' + str(percentiles[2]): utilization_3}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "/" + output_filename) as writer:
        df.to_excel(writer, index=False)


def save_overall_results(users_and_facs_df, travel_dict, users, facs, output_filename='overall_results.xlsx',
                         output_abs_path=None, percentiles=None, budget_factor_list=None, strict_assign_to_one=False,
                         cap_factor=1.5, cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3,
                         main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                         post_threads=1, post_tolerance=0.0, post_print_sol=False, post_log_file=None,
                         post_preqlinearize=-1, different_objective = False, closed_facilities_list = [], continous = False):
    """
    create an excel file that displays broad results for different budgets after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param percentiles: list of the three considered percentiles. Values must be between 0 and 100
    :param budget_factor_list: list of different ratios of facilities that are allowed to be opened
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
    :param different_objective: define u seperately as utilisation
    :param continous: indicates if secondary model should be solved continously
    """

    # init
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]
    if budget_factor_list is None:
        budget_factor_list = [0.1 * b for b in range(3, 11)]
    results_list = []

    # build and optimize the relevant models
    for budget_factor in budget_factor_list:
        is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                   strict_assign_to_one, cap_factor, cutoff, max_access,
                                                   main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                   main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                   post_print_sol, post_log_file, post_preqlinearize, different_objective, -1, closed_facilities_list, continous)
        if not is_feasible:
            print('Infeasible model')
            return None
        results_list.append(results)
        #write_results_list(results, "naive_model_results" + str(budget_factor)+".json")


    # write the results
    write_results_list(results_list, "diff_objective.json")
    write_overall_results(results_list, users_and_facs_df, travel_dict, output_filename, output_abs_path, percentiles)
    #write_fairness_results(results_list, users_and_facs_df, travel_dict, "adapted_MIP_fairness.xlsx", output_abs_path)



def write_fairness_results(results_list, users_and_facs_df, travel_dict, output_filename='fairness_results.xlsx',
                           output_abs_path=None):
    """
    create an excel file that displays fairness results for different budgets
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the results are sorted in decreasing order by budget
    results_list = sorted(results_list, key=lambda d: -d['model_details']['budget_factor'])
    nr_of_results = len(results_list)

    # get relevant data
    budget_factor_list = [round(results['model_details']['budget_factor'] * 100) for results in results_list]
    budget_string_list = [str(budget_factor_list[i // 4]) + "%" if i % 4 == 0 else None for i in
                          range(nr_of_results * 4)]
    regions = ['all', 'rural', 'urban', None] * nr_of_results
    wm_list = []
    wstd_list = []
    wcv_list = []
    dof_list = []
    dof_prime_list = []
    prop_open_list = []
    prop_cacpacity_open_list = []
    for results in results_list:
        wm_all, wstd_all, wcv_all = get_weighted_utilization_figures(results, users_and_facs_df, travel_dict, 'all')
        wm_rural, wstd_rural, wcv_rural = \
            get_weighted_utilization_figures(results, users_and_facs_df, travel_dict, 'rural')
        wm_urban, wstd_urban, wcv_urban = \
            get_weighted_utilization_figures(results, users_and_facs_df, travel_dict, 'urban')
        prop_all, prop_rural, prop_urban = get_proportion_open(results, users_and_facs_df)
        prop_all_cap, prop_rural_cap, prop_urban_cap = get_proportion_open_by_capacity(results, users_and_facs_df)
        wm_list += [wm_all * 100, wm_rural * 100, wm_urban * 100, None]
        wstd_list += [wstd_all * 100, wstd_rural * 100, wstd_urban * 100, None]
        wcv_list += [wcv_all * 100, wcv_rural * 100, wcv_urban * 100, None]
        dof, dof_prime = get_dof(results, users_and_facs_df, travel_dict)
        dof_list += [dof * 100, None, None, None]
        dof_prime_list += [dof_prime * 100, None, None, None]
        prop_open_list += [prop_all, prop_rural, prop_urban, None]
        prop_cacpacity_open_list += [prop_all_cap, prop_rural_cap, prop_urban_cap, None]

    # write the excel file
    data = {'Budget': budget_string_list, 'Region': regions, 'Weighted mean [%]': wm_list,
            'Weighted standard deviation [%]': wstd_list, 'Weighted CV [%]': wcv_list, 'DoF [%]': dof_list,
            "DoF\' [%]": dof_prime_list, 'Proportion open': prop_open_list, 'Proportion by capacity open': prop_cacpacity_open_list}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "/" + output_filename) as writer:
        df.to_excel(writer, index=False)


def save_fairness_results(users_and_facs_df, travel_dict, users, facs, output_filename='fairness_results.xlsx',
                          output_abs_path=None,  budget_factor_list=None, strict_assign_to_one=False, cap_factor=1.5,
                          cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3, main_time_limit=20000,
                          main_print_sol=False, main_log_file=None, main_preqlinearize=-1, post_threads=1,
                          post_tolerance=0.0, post_print_sol=False, post_log_file=None, post_preqlinearize=-1):
    """
    create an excel file that displays fairness results for different budgets after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param budget_factor_list: list of different ratios of facilities that are allowed to be opened
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
    """
    # init
    if budget_factor_list is None:
        budget_factor_list = [0.1 * b for b in range(3, 11)]
    results_list = []

    # build and optimize the relevant models
    for budget_factor in budget_factor_list:
        is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                   strict_assign_to_one, cap_factor, cutoff, max_access,
                                                   main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                   main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                   post_print_sol, post_log_file, post_preqlinearize)
        if not is_feasible:
            print('Infeasible model')
            return None
        results_list.append(results)

    # write the results
    write_fairness_results(results_list, users_and_facs_df, travel_dict, output_filename, output_abs_path)


def write_greedy_results(greedy_results_list, naive_results_list, users_and_facs_df, travel_dict,
                         output_filename='greedy_results.csv', output_abs_path=None):
    """
    create an excel file that compares results obtained by the greedy heuristic
     with results obtained by the naive solution method for different budgets
    :param greedy_results_list: list of dictionaries of the results obtained by the greedy heuristic
    :param naive_results_list: list of dictionaries of the results obtained by the naive solution method
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the greedy results are sorted in increasing order by number of users and then number of facilities
    greedy_results_list = sorted(greedy_results_list, key=lambda d: len(d['model_details']['users']) +
                                                                    1e-6 * len(d['model_details']['facs']))

    # make dictionaries where the keys are the users and facilities of an instance and the values are the results
    # to allow comparing the same instances with each other
    greedy_results_dict = {(tuple(results['model_details']['users']), tuple(results['model_details']['facs'])):
                               results for results in greedy_results_list}
    naive_results_dict = {(tuple(results['model_details']['users']), tuple(results['model_details']['facs'])):
                              results for results in naive_results_list}

    # if the naive method needed post-processing, it did not return a lower bound
    # in that case the objective values are compared, instead of the optimality gaps
    is_lb_available = True
    for results in naive_results_dict.values():
        if results['solution_details']['lower_bound'] is None:
            is_lb_available = False
            break

    # init
    nr_of_users_list = []
    nr_of_facs_list = []
    if is_lb_available:
        naive_gap_list = []
        greedy_gap_list = []
        guarantee_list = []
    else:
        naive_obj_list = []
        greedy_obj_list = []
        delta_obj_list = []
    naive_solving_time_list = []
    greedy_solving_time_list = []
    delta_solving_time_list = []
    naive_access_list = []
    greedy_access_list = []
    delta_access_list = []
    naive_wcv_list = []
    greedy_wcv_list = []
    delta_wcv_list = []

    # get relevant data
    for key in greedy_results_dict:
        greedy_obj = greedy_results_dict[key]['solution_details']['objective_value']
        if is_lb_available:
            greedy_lb = greedy_results_dict[key]['solution_details']['lower_bound']
            greedy_gap = 100 * (greedy_obj - greedy_lb) / greedy_obj
            if greedy_gap <= greedy_results_dict[key]['model_details']['tolerance']:
                greedy_gap = None
        greedy_solving_time = greedy_results_dict[key]['solution_details']['solving_time']
        greedy_access = 100 * get_overall_access(greedy_results_dict[key], users_and_facs_df, travel_dict)
        _, _, greedy_wcv = get_weighted_utilization_figures(greedy_results_dict[key], users_and_facs_df, travel_dict)
        greedy_wcv = 100 * greedy_wcv
        if key in naive_results_dict:
            naive_obj = naive_results_dict[key]['solution_details']['objective_value']
            if is_lb_available:
                naive_lb = naive_results_dict[key]['solution_details']['lower_bound']
                naive_gap = (naive_obj - naive_lb) / naive_obj
                if naive_gap <= naive_results_dict[key]['model_details']['tolerance']:
                    naive_gap = None
                else:
                    naive_gap = 100 * naive_gap
                guarantee = 100 * (greedy_obj - naive_lb) / greedy_obj
                if guarantee <= greedy_results_dict[key]['model_details']['tolerance']:
                    guarantee = None
            else:
                delta_obj = 100 * (naive_obj - greedy_obj) / naive_obj
            naive_solving_time = naive_results_dict[key]['solution_details']['solving_time']
            delta_solving_time = 100 * (naive_solving_time - greedy_solving_time) / naive_solving_time
            # if the time limit has been exceeded, a "T" is reported instead of the actual time
            if greedy_solving_time > greedy_results_dict[key]['model_details']['time_limit']:
                greedy_solving_time = "T"
            if naive_solving_time > naive_results_dict[key]['model_details']['time_limit']:
                naive_solving_time = "T"
            naive_access = 100 * get_overall_access(naive_results_dict[key], users_and_facs_df, travel_dict)
            delta_access = 100 * (greedy_access - naive_access) / naive_access
            _, _, naive_wcv = get_weighted_utilization_figures(naive_results_dict[key], users_and_facs_df,
                                                               travel_dict)
            naive_wcv = 100 * naive_wcv
            delta_wcv = 100 * (naive_wcv - greedy_wcv) / naive_wcv
        else:
            if is_lb_available:
                naive_gap = "-"
                guarantee = "-"
            else:
                naive_obj = "-"
                delta_obj = "-"
            naive_solving_time = "-"
            delta_solving_time = "-"
            naive_access = "-"
            delta_access = "-"
            naive_wcv = "-"
            delta_wcv = "-"
        nr_of_users_list.append(len(key[0]))
        nr_of_facs_list.append(len(key[1]))
        if is_lb_available:
            naive_gap_list.append(naive_gap)
            greedy_gap_list.append(greedy_gap)
            guarantee_list.append(guarantee)
        else:
            naive_obj_list.append(naive_obj)
            greedy_obj_list.append(greedy_obj)
            delta_obj_list.append(delta_obj)
        naive_solving_time_list.append(naive_solving_time)
        greedy_solving_time_list.append(greedy_solving_time)
        delta_solving_time_list.append(delta_solving_time)
        naive_access_list.append(naive_access)
        greedy_access_list.append(greedy_access)
        delta_access_list.append(delta_access)
        naive_wcv_list.append(naive_wcv)
        greedy_wcv_list.append(greedy_wcv)
        delta_wcv_list.append(delta_wcv)

    # write the excel file
    if is_lb_available:
        data = {'|I|': nr_of_users_list, '|J|': nr_of_facs_list,
                'Gap [%] - Naive ': naive_gap_list, 'Gap [%] - Algorithm': greedy_gap_list,
                'Gap [%] - Guarantee': guarantee_list,
                'Time [s] - Naive': naive_solving_time_list, 'Time [s] - Algorithm': greedy_solving_time_list,
                'Time [%] - Delta': delta_solving_time_list,
                'Overall access [%] - Naive': naive_access_list, 'Overall access [%] - Algorithm': greedy_access_list,
                'Overall access [%] - Delta': delta_access_list,
                'Weighted CV [%] - Naive': naive_wcv_list, 'Weighted CV [%] - Algorithm': greedy_wcv_list,
                'Weighted CV [%] - Delta': delta_wcv_list
                }
    else:
        data = {'|I|': nr_of_users_list, '|J|': nr_of_facs_list,
                'Objective value - Naive ': naive_obj_list, 'Objective value - Algorithm': greedy_obj_list,
                'Objective value [%] - Delta': delta_obj_list,
                'Time [s] - Naive': naive_solving_time_list, 'Time [s] - Algorithm': greedy_solving_time_list,
                'Time [%] - Delta': delta_solving_time_list,
                'Overall access [%] - Naive': naive_access_list, 'Overall access [%] - Algorithm': greedy_access_list,
                'Overall access [%] - Delta': delta_access_list,
                'Weighted CV [%] - Naive': naive_wcv_list, 'Weighted CV [%] - Algorithm': greedy_wcv_list,
                'Weighted CV [%] - Delta': delta_wcv_list
                }
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    df.to_csv(output_abs_path + "/" + output_filename, index=False)


def save_greedy_results(users_and_facs_df, travel_dict, instance_dict=None, output_filename='greedy_results.csv',
                        output_abs_path=None, budget_factor=1.0, strict_assign_to_one=False, cap_factor=1.5, cutoff=0.2,
                        max_access=False, main_threads=1, main_tolerance=5e-3, main_time_limit=20000,
                        main_print_sol=False, main_log_file=None, main_preqlinearize=-1, post_threads=1,
                        post_tolerance=0.0, post_print_sol=False, post_log_file=None, post_preqlinearize=-1,
                        greedy_turnover_factor=0.02, greedy_tolerance=5e-3, greedy_time_limit=20000,
                        greedy_iteration_limit=20):
    """
    create an excel file that compares results obtained by the greedy heuristic
     with results obtained by the naive solution method for different budgets after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param instance_dict: dictionary of instances for different instance sizes
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
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
    :param greedy_turnover_factor: maximal fraction of facilities that are swapped in each iteration
    :param greedy_tolerance: tolerance on the optimality of the solution
    :param greedy_time_limit: time limit in seconds
    :param greedy_iteration_limit: limit for the number of consecutive iterations without improvement
    """
    # init
    if instance_dict is None:
        users = [int(i) for i in users_and_facs_df.index]
        facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
        sizes = [(0.1 * i, 0.1 * i) for i in range(1, 11)]
        instance_dict = {
            size: {0: {'users': users[:round(len(users) * size[0])], 'facs': facs[:round(len(facs) * size[1])]}}
            for size in sizes}
    naive_results_list = []
    greedy_results_list = []

    # build and optimize the relevant models
    for instance_size in instance_dict:
        for instance_nr in instance_dict[instance_size]:
            users = instance_dict[instance_size][instance_nr]['users']
            facs = instance_dict[instance_size][instance_nr]['facs']
            if cutoff > 0.0 or max(instance_size) <= 0.5:
                # get the naive results
                is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                           strict_assign_to_one, cap_factor, cutoff, max_access,
                                                           main_threads, main_tolerance, main_time_limit,
                                                           main_print_sol,
                                                           main_log_file, main_preqlinearize, post_threads,
                                                           post_tolerance,
                                                           post_print_sol, post_log_file, post_preqlinearize)
                if not is_feasible:
                    print('Infeasible model')
                    return None
                naive_results_list.append(results)
            else:
                print('An instance with', len(instance_dict[instance_size][0]['users']), 'users and ',
                      len(instance_dict[instance_size][0]['facs']), 'facilities cannot be generated without a cutoff.')
                print('Skipping this instance.')

            # get the greedy results
            lb = get_lower_bound(users_and_facs_df, travel_dict, users, facs, budget_factor, cap_factor)
            is_feasible, results = solve_greedily(users_and_facs_df, travel_dict, users, facs, lb, budget_factor,
                                                  cap_factor, greedy_turnover_factor, greedy_tolerance,
                                                  greedy_time_limit, greedy_iteration_limit)
            if not is_feasible:
                print('The greedy heuristic could not construct a solution')
                return None
            greedy_results_list.append(results)

    # write the results
    write_greedy_results(greedy_results_list, naive_results_list, users_and_facs_df, travel_dict, output_filename,
                         output_abs_path)


def write_cutoff_results(cutoff_results_list, nocutoff_results_list, output_filename='cutoff_results.xlsx',
                         output_abs_path=None):
    """
    create an excel file that compares results obtained by the reduced model with results obtained by the true model
    :param cutoff_results_list: list of dictionaries of the results obtained by the reduced model
    :param nocutoff_results_list: list of dictionaries of the results obtained by the true model
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the results that use a cutoff are sorted in increasing order by number of users and then number of
    # facilities
    cutoff_results_list = sorted(cutoff_results_list, key=lambda d: len(d['model_details']['users']) +
                                                                    1e-6 * len(d['model_details']['facs']))

    # make dictionaries where the keys are the users and facilities of an instance and the values are the results
    # to allow comparing the same instances with each other
    cutoff_results_dict = {(tuple(results['model_details']['users']), tuple(results['model_details']['facs'])):
                               results for results in cutoff_results_list}
    nocutoff_results_dict = {(tuple(results['model_details']['users']), tuple(results['model_details']['facs'])):
                                 results for results in nocutoff_results_list}

    # if the method without cutoff needed post-processing, it did not return a lower bound
    # in that case the objective values are compared, instead of the optimality gaps
    is_lb_available = True
    for results in nocutoff_results_dict.values():
        if results['solution_details']['lower_bound'] is None:
            is_lb_available = False
            break

    # init
    nr_of_users_list = []
    nr_of_facs_list = []
    nocutoff_obj_list = []
    cutoff_obj_list = []
    delta_obj_list = []
    guarantee_list = []
    nocutoff_solving_time_list = []
    cutoff_solving_time_list = []
    delta_solving_time_list = []

    # get relevant data
    for key in cutoff_results_dict:
        cutoff_obj = cutoff_results_dict[key]['solution_details']['objective_value']
        cutoff_solving_time = cutoff_results_dict[key]['solution_details']['solving_time']
        if key in nocutoff_results_dict:
            nocutoff_obj = nocutoff_results_dict[key]['solution_details']['objective_value']
            delta_obj = 100 * (nocutoff_obj - cutoff_obj) / nocutoff_obj
            if is_lb_available:
                nocutoff_lb = nocutoff_results_dict[key]['solution_details']['lower_bound']
                guarantee = 100 * (cutoff_obj - nocutoff_lb) / cutoff_obj
            else:
                guarantee = "-"
            nocutoff_solving_time = nocutoff_results_dict[key]['solution_details']['solving_time']
            delta_solving_time = 100 * (nocutoff_solving_time - cutoff_solving_time) / nocutoff_solving_time
        else:
            nocutoff_obj = "-"
            delta_obj = "-"
            guarantee = "-"
            nocutoff_solving_time = "-"
            delta_solving_time = "-"
        # if the time limit has been exceeded, a "T" is reported instead of the actual time
        if cutoff_solving_time > cutoff_results_dict[key]['model_details']['time_limit']:
            cutoff_solving_time = "T"
        if nocutoff_solving_time > nocutoff_results_dict[key]['model_details']['time_limit']:
            nocutoff_solving_time = "T"
        nr_of_users_list.append(len(key[0]))
        nr_of_facs_list.append(len(key[1]))
        nocutoff_obj_list.append(nocutoff_obj)
        cutoff_obj_list.append(cutoff_obj)
        delta_obj_list.append(delta_obj)
        guarantee_list.append(guarantee)
        nocutoff_solving_time_list.append(nocutoff_solving_time)
        cutoff_solving_time_list.append(cutoff_solving_time)
        delta_solving_time_list.append(delta_solving_time)

    # write the excel file
    cutoff = round(100 * cutoff_results_list[0]['model_details']['cutoff'])
    data = {'|I|': nr_of_users_list, '|J|': nr_of_facs_list,
            'Objective value - 0% cutoff ': nocutoff_obj_list,
            'Objective value - ' + str(cutoff) + '% cutoff': cutoff_obj_list,
            'Objective value [%] - Delta': delta_obj_list, 'Guarantee [%]': guarantee_list,
            'Time [s] - 0% cutoff': nocutoff_solving_time_list,
            'Time [s] - ' + str(cutoff) + '% cutoff': cutoff_solving_time_list,
            'Time [%] - Delta': delta_solving_time_list
            }
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "\\" + output_filename) as writer:
        df.to_excel(writer, index=False)


def save_cutoff_results(users_and_facs_df, travel_dict, instance_dict=None, output_filename='cutoff_results.xlsx',
                        output_abs_path=None,  budget_factor=1.0, strict_assign_to_one=False, cap_factor=1.5,
                        compared_cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3,
                        main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize_cutoff=1,
                        main_preqlinearize_nocutoff=-1, post_threads=1, post_tolerance=0.0, post_print_sol=False,
                        post_log_file=None, post_preqlinearize=-1):
    """
    create an excel file that compares results obtained by the reduced model with results obtained by the true model
     after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param instance_dict: dictionary of instances for different instance sizes
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param budget_factor: ratio of facilities that are allowed to be opened
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned
        to exactly 1 facility in the main model
    :param cap_factor: factor by which all capacities are scaled
    :param compared_cutoff: travel combinations with a probability smaller than this value will be removed
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :param main_threads: number of threads used in the main optimization step
    :param main_tolerance: tolerance on the optimality of the solution used in the main optimization step
    :param main_time_limit: time limit in seconds for the optimization used in the main optimization step
    :param main_print_sol: boolean indicating whether the solution should be printed used in the main optimization step
    :param main_log_file: name of the log file for the main optimization step; if "None", no log file will be produced
    :param main_preqlinearize_cutoff: setting for Gurobi's PreQLinearize parameter
        in the main optimization step of the reduced model
    :param main_preqlinearize_nocutoff: setting for Gurobi's PreQLinearize parameter
        in the main optimization step of the true model
    :param post_threads: number of threads used in the post processing step
    :param post_tolerance: tolerance on the optimality of the solution used in the post processing step
    :param post_print_sol: boolean indicating whether the solution should be printed used in the post processing step
    :param post_log_file: name of the log file for the post processing step; if "None", no log file will be produced
    :param post_preqlinearize: setting for Gurobi's PreQLinearize parameter used in the post processing step
    """
    # init
    if instance_dict is None:
        users = [int(i) for i in users_and_facs_df.index]
        facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
        sizes = [(0.1 * i, 0.1 * j) for i in range(1, 6) for j in range(i - 1, i + 2) if 0 < j < 6]
        instance_dict = {
            size: {0: {'users': users[:round(len(users) * size[0])], 'facs': facs[:round(len(facs) * size[1])]}}
            for size in sizes}
    cutoff_results_list = []
    nocutoff_results_list = []

    # build and optimize the relevant models
    for instance_size in instance_dict:
        for instance_nr in instance_dict[instance_size]:
            users = instance_dict[instance_size][instance_nr]['users']
            facs = instance_dict[instance_size][instance_nr]['facs']
            for cutoff in [0.0, compared_cutoff]:
                if cutoff > 0.0:
                    main_preqlinearize = main_preqlinearize_cutoff
                else:
                    main_preqlinearize = main_preqlinearize_nocutoff
                is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                           strict_assign_to_one, cap_factor, cutoff, max_access,
                                                           main_threads, main_tolerance, main_time_limit,
                                                           main_print_sol, main_log_file, main_preqlinearize,
                                                           post_threads, post_tolerance, post_print_sol,
                                                           post_log_file, post_preqlinearize)
                if not is_feasible:
                    print('Infeasible model')
                    return None
                if cutoff > 0.0:
                    cutoff_results_list.append(results)
                else:
                    nocutoff_results_list.append(results)

    # write the results
    write_cutoff_results(cutoff_results_list, nocutoff_results_list, output_filename, output_abs_path)


def write_pof_results(optimal_results_list, maximum_results_list, users_and_facs_df, travel_dict,
                      output_filename='pof_results.xlsx', output_abs_path=None):
    """
    create an excel file that displaying the price of fairness of the model
    :param optimal_results_list: list of dictionaries of the results obtained by the model
    :param maximum_results_list: list of dictionaries of the results obtained by a reference model
        that seeks to maximize access
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the results are sorted in increasing order by budget
    optimal_results_list = sorted(optimal_results_list, key=lambda d: d['model_details']['budget_factor'])
    maximum_results_list = sorted(maximum_results_list, key=lambda d: d['model_details']['budget_factor'])
    nr_of_results = len(optimal_results_list)

    # get relevant data
    budget_string_list = [str(round(results['model_details']['budget_factor'] * 100)) + "%"
                          for results in optimal_results_list]
    optimal_access_list = []
    maximum_access_list = []
    pof_list = []
    for i in range(nr_of_results):
        optimal_access = 100 * get_overall_access(optimal_results_list[i], users_and_facs_df, travel_dict)
        maximum_access = 100 * get_overall_access(maximum_results_list[i], users_and_facs_df, travel_dict)
        optimal_access_list.append(optimal_access)
        maximum_access_list.append(maximum_access)
        pof_list.append(100 * (maximum_access - optimal_access) / maximum_access)

    # write the excel file
    df = pd.DataFrame(columns=['Budget'] + budget_string_list, dtype=object)
    df.loc[1] = ['Optimal overall access [%]'] + optimal_access_list
    df.loc[2] = ['Maximum overall access [%]'] + maximum_access_list
    df.loc[3] = ['Price of fairness [%]'] + pof_list

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "\\" + output_filename) as writer:
        df.to_excel(writer, index=False)


def save_pof_results(users_and_facs_df, travel_dict, users, facs, output_filename='pof_results.xlsx',
                     output_abs_path=None, budget_factor_list=None, strict_assign_to_one=False, cap_factor=1.5,
                     cutoff=0.2, main_threads=1, main_tolerance=5e-3, main_time_limit=20000, main_print_sol=False,
                     main_log_file=None, main_preqlinearize=-1, post_threads=1, post_tolerance=0.0,
                     post_print_sol=False, post_log_file=None, post_preqlinearize=-1):
    """
    create an excel file that displaying the price of fairness of the model after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param budget_factor_list: list of different ratios of facilities that are allowed to be opened
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
    """
    # init
    if budget_factor_list is None:
        budget_factor_list = [0.1 * b for b in range(3, 11)]
    optimal_results_list = []
    maximum_results_list = []

    # build and optimize the relevant models
    for budget_factor in budget_factor_list:
        for max_access in [True, False]:
            is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                       strict_assign_to_one, cap_factor, cutoff, max_access,
                                                       main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                       main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                       post_print_sol, post_log_file, post_preqlinearize)
            if not is_feasible:
                print('Infeasible model')
                return None
            if max_access:
                maximum_results_list.append(results)
            else:
                optimal_results_list.append(results)

    # write the results
    write_pof_results(optimal_results_list, maximum_results_list, users_and_facs_df, travel_dict, output_filename,
                      output_abs_path)


def write_strict_vs_loose_results(strict_results_list, loose_results_list, users_and_facs_df, travel_dict,
                                  output_filename='strict_vs_loose_results.xlsx', output_abs_path=None):
    """
    create an excel file that compares results obtained using the two different implementations of the
     assign-to-one constraint
    :param strict_results_list: list of dictionaries of the results obtained when modeling the assign-to-one constraint
        as an equality
    :param strict_results_list: list of dictionaries of the results obtained when modeling the assign-to-one constraint
        as an inequality
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the results before the post-processing step are listed before the results after the post-processing step
    strict_results_list = sorted(strict_results_list, key=lambda d: d['solution_details']['solving_time'])
    loose_results_list = sorted(loose_results_list, key=lambda d: d['solution_details']['solving_time'])

    # get relevant data
    frac_assigned_list = [100 * sum(1 for fac in results['solution_details']['assignment'].values() if fac) /
                          len(results['model_details']['users']) for results in
                          strict_results_list + loose_results_list]
    access_list = [100 * get_overall_access(results, users_and_facs_df, travel_dict)
                   for results in strict_results_list + loose_results_list]
    obj_list = [results['solution_details']['objective_value'] for results in strict_results_list + loose_results_list]
    solving_time_list = [results['solution_details']['solving_time']
                         for results in strict_results_list + loose_results_list]
    for lst in [frac_assigned_list, access_list, obj_list, solving_time_list]:
        lst.append(100 * (lst[1] - lst[3]) / lst[1])

    # write the excel file
    df = pd.DataFrame(columns=['Strict - Before', 'Strict - After', 'Loose - Before', 'Loose - After',
                               'Improvement [%]'],
                      index=['Assigned zip codes [%]', 'Overall access [%]', 'Objective value', 'Solving time [s]'],
                      dtype=object)
    df.loc['Assigned zip codes [%]'] = frac_assigned_list
    df.loc['Overall access [%]'] = access_list
    df.loc['Objective value'] = obj_list
    df.loc['Solving time [s]'] = solving_time_list

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "\\" + output_filename) as writer:
        df.to_excel(writer, index=False)


def save_strict_vs_loose_results(users_and_facs_df, travel_dict, users, facs,
                                 output_filename='strict_vs_loose_results.xlsx', output_abs_path=None,
                                 budget_factor=1.0, cap_factor=1.5, cutoff=0.2, max_access=False, main_threads=1,
                                 main_tolerance=5e-3, main_time_limit=20000, main_print_sol=False, main_log_file=None,
                                 main_preqlinearize=-1, post_threads=1, post_tolerance=0.0, post_print_sol=False,
                                 post_log_file=None, post_preqlinearize=-1):
    """
    create an excel file that displaying the price of fairness of the model after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param budget_factor: ratio of facilities that are allowed to be opened
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
    """
    # init
    strict_results_list = []
    loose_results_list = []

    # build and optimize the relevant models
    for strict_assign_to_one in [True, False]:
        model = build_model(users_and_facs_df, travel_dict, users, facs, budget_factor, strict_assign_to_one,
                            cap_factor, cutoff, max_access)
        is_feasible, results = optimize_model(model, main_threads, main_tolerance, main_time_limit, main_print_sol,
                                              main_log_file, main_preqlinearize)
        if not is_feasible:
            print('Infeasible model')
            return None
        if strict_assign_to_one:
            strict_results_list.append(results)
        else:
            loose_results_list.append(results)
        if None in results['solution_details']['assignment'].values():
            print()
            print('Postprocessing...')
            post_model = build_postprocessing_model(model, results)
            is_feasible, results = optimize_postprocessing_model(post_model, results, post_threads, post_tolerance,
                                                                 post_print_sol, post_log_file, post_preqlinearize)
            if not is_feasible:
                print('The post processing model is infeasible')
                return None
        if strict_assign_to_one:
            strict_results_list.append(results)
        else:
            loose_results_list.append(results)

    # write the results
    write_strict_vs_loose_results(strict_results_list, loose_results_list, users_and_facs_df, travel_dict,
                                  output_filename, output_abs_path)

def write_results_list(results, output_filename, output_abs_path = None):
    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with open(output_abs_path + "/" + output_filename, 'w') as f:
        json.dump(results, f)


def save_greedy_results_urban_rural(users_and_facs_df, travel_dict, budget_factor_dict=None, output_filename='greedy_results_compare_prop.xlsx',
                        output_abs_path=None, cap_factor=1.5, greedy_turnover_factor=0.02, greedy_tolerance=5e-3, greedy_time_limit=20000,
                        greedy_iteration_limit=20):
    """
    create an excel file that compares results obtained by the greedy heuristic
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param budget_factor_dict: dictionary of ratios of facilities that are allowed to be opened
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param cap_factor: factor by which all capacities are scaled
    :param greedy_turnover_factor: maximal fraction of facilities that are swapped in each iteration
    :param greedy_tolerance: tolerance on the optimality of the solution
    :param greedy_time_limit: time limit in seconds
    :param greedy_iteration_limit: limit for the number of consecutive iterations without improvement
    """
    # init
    
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    if budget_factor_dict == None:
        budget_factor_dict = [0.1*i for i in range(3,11)]
    greedy_results_list = []

    # run greedy algorithm for the different budgets
    for budget_factor in budget_factor_dict:
        # get the greedy results
        lb = get_lower_bound(users_and_facs_df, travel_dict, users, facs, budget_factor, cap_factor)
        is_feasible, results = solve_greedily(users_and_facs_df, travel_dict, users, facs, lb, budget_factor,
                                                cap_factor, greedy_turnover_factor, greedy_tolerance,
                                                greedy_time_limit, greedy_iteration_limit)
        if not is_feasible:
            print('The greedy heuristic could not construct a solution')
            return None
        greedy_results_list.append(results)
    # write the results
    write_fairness_results(greedy_results_list, users_and_facs_df, travel_dict, output_filename,
                           output_abs_path)

def save_greedy_results_urban_rural_split(users_and_facs_df, travel_dict, budget_factor_dict=None, output_filename='greedy_results_split.xlsx',
                        output_abs_path=None, cap_factor=1.5, greedy_turnover_factor=0.02, greedy_tolerance=5e-3, greedy_time_limit=20000,
                        greedy_iteration_limit=20):
    """
    create an excel file that compares results obtained by the greedy heuristic when run seperately on urban and rural
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param budget_factor_dict: dictionary of ratios of facilities that are allowed to be opened
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param cap_factor: factor by which all capacities are scaled
    :param greedy_turnover_factor: maximal fraction of facilities that are swapped in each iteration
    :param greedy_tolerance: tolerance on the optimality of the solution
    :param greedy_time_limit: time limit in seconds
    :param greedy_iteration_limit: limit for the number of consecutive iterations without improvement
    """
    # init
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    users_rural, facs_rural, users_urban, facs_urban = split_rural_urban(users_and_facs_df, users, facs)
    if budget_factor_dict == None:
        budget_factor_dict = [0.1*i for i in range(3,11)]
    greedy_results_list = []

    # run greedy algorithm seperate and combine for the different budgets
    for budget_factor in budget_factor_dict:
        lb_rural = get_lower_bound(users_and_facs_df, travel_dict, users_rural, facs_rural, budget_factor, cap_factor)
        lb_urban = get_lower_bound(users_and_facs_df, travel_dict, users_urban, facs_urban, budget_factor, cap_factor)
        is_feasible_r, results_r = solve_greedily(users_and_facs_df, travel_dict, users_rural, facs_rural, lb_rural, budget_factor,
                                                cap_factor, greedy_turnover_factor, greedy_tolerance,
                                                greedy_time_limit, greedy_iteration_limit)
        is_feasible_u, results_u = solve_greedily(users_and_facs_df, travel_dict, users_urban, facs_urban, lb_urban, budget_factor,
                                                cap_factor, greedy_turnover_factor, greedy_tolerance,
                                                greedy_time_limit, greedy_iteration_limit)
        if(not(is_feasible_r and is_feasible_u)):
            print('Infeasible model at' + str(budget_factor))
            continue
        result_combined = combine_results(results_r, results_u)
        greedy_results_list.append(result_combined)
    # write the results
    write_fairness_results(greedy_results_list, users_and_facs_df, travel_dict, output_filename,
                           output_abs_path)

def save_travel_distance_greedy_results_split(users_and_facs_df, travel_dict, budget_factor_dict=None, users = None, facs = None, output_filename='greedy_travel_split.xlsx',
                         output_abs_path=None, percentiles=None, cap_factor=1.5, greedy_turnover_factor=0.02, greedy_tolerance=5e-3,
                         greedy_time_limit=20000,greedy_iteration_limit=20):
    """
    create an excel file that displays broad results for different budgets after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param buget_factor_dict: dictionary of different budgets to be considered
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param percentiles: list of the three considered percentiles. Values must be between 0 and 100
    :param budget_factor_list: list of different ratios of facilities that are allowed to be opened
    :param cap_factor: factor by which all capacities are scaled
    :param greedy_turnover_factor: maximal fraction of facilities that are swapped in each iteration
    :param greedy_tolerance: tolerance on the optimality of the solution
    :param greedy_time_limit: time limit in seconds
    :param greedy_iteration_limit: limit for the number of consecutive iterations without improvement
    """

    # init
    if users == None:
        users = [int(i) for i in users_and_facs_df.index]
    if facs == None:
        facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    users_rural, facs_rural, users_urban, facs_urban = split_rural_urban(users_and_facs_df, users, facs)

    if budget_factor_dict == None:
        budget_factor_dict = [0.1*i for i in range(3,11)]
    greedy_results_list = []
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]
    greedy_results_list = []

    # run greedy algorithm seperate and combine for the different budgets
    for budget_factor in budget_factor_dict:
        lb_rural = get_lower_bound(users_and_facs_df, travel_dict, users_rural, facs_rural, budget_factor, cap_factor)
        lb_urban = get_lower_bound(users_and_facs_df, travel_dict, users_urban, facs_urban, budget_factor, cap_factor)
        is_feasible_r, results_r = solve_greedily(users_and_facs_df, travel_dict, users_rural, facs_rural, lb_rural, budget_factor,
                                                cap_factor, greedy_turnover_factor, greedy_tolerance,
                                                greedy_time_limit, greedy_iteration_limit)
        is_feasible_u, results_u = solve_greedily(users_and_facs_df, travel_dict, users_urban, facs_urban, lb_urban, budget_factor,
                                                cap_factor, greedy_turnover_factor, greedy_tolerance,
                                                greedy_time_limit, greedy_iteration_limit)
        if(not(is_feasible_r and is_feasible_u)):
            print('Infeasible model at' + str(budget_factor))
            continue
        result_combined = combine_results(results_r, results_u)
        greedy_results_list.append(result_combined)

    # write the results
    write_overall_results(greedy_results_list, users_and_facs_df, travel_dict, output_filename, output_abs_path, percentiles)
    
def save_travel_distance_greedy_results(users_and_facs_df, travel_dict, budget_factor_dict=None, users = None, facs = None, output_filename='greedy_travel.xlsx',
                         output_abs_path=None, percentiles=None, cap_factor=1.5, greedy_turnover_factor=0.02, greedy_tolerance=5e-3,
                         greedy_time_limit=20000,greedy_iteration_limit=20):
    """
    create an excel file that displays broad results for different budgets after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param buget_factor_dict: dictionary of different budgets to be considered
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param percentiles: list of the three considered percentiles. Values must be between 0 and 100
    :param budget_factor_list: list of different ratios of facilities that are allowed to be opened
    :param cap_factor: factor by which all capacities are scaled
    :param greedy_turnover_factor: maximal fraction of facilities that are swapped in each iteration
    :param greedy_tolerance: tolerance on the optimality of the solution
    :param greedy_time_limit: time limit in seconds
    :param greedy_iteration_limit: limit for the number of consecutive iterations without improvement
    """

    # init
    if users == None:
        users = [int(i) for i in users_and_facs_df.index]
    if facs == None:
        facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]

    if budget_factor_dict == None:
        budget_factor_dict = [0.1*i for i in range(3,11)]
    greedy_results_list = []
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]
    greedy_results_list = []

    # run greedy algorithm seperate and combine for the different budgets
    for budget_factor in budget_factor_dict:
        # get the greedy results
        lb = get_lower_bound(users_and_facs_df, travel_dict, users, facs, budget_factor, cap_factor)
        is_feasible, results = solve_greedily(users_and_facs_df, travel_dict, users, facs, lb, budget_factor,
                                                cap_factor, greedy_turnover_factor, greedy_tolerance,
                                                greedy_time_limit, greedy_iteration_limit)
        if not is_feasible:
            print('The greedy heuristic could not construct a solution')
            return None
        greedy_results_list.append(results)

    # write the results
    write_overall_results(greedy_results_list, users_and_facs_df, travel_dict, output_filename, output_abs_path, percentiles)
    
def save_seperate_budgets_results(users_and_facs_df, travel_dict, users, facs, output_filename='rural_urban_budget_results.xlsx',
                          output_abs_path=None, budget_factor = 0.7, budget_factor_list_rural=None, strict_assign_to_one=False, cap_factor=1.5,
                          cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3, main_time_limit=20000,
                          main_print_sol=False, main_log_file=None, main_preqlinearize=-1, post_threads=1,
                          post_tolerance=0.0, post_print_sol=False, post_log_file=None, post_preqlinearize=-1, different_objective = False, continous = False):
    """
    create an excel file that displays fairness and acess results for different budgets, split by rural-urban, after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param budget_factor: overall budget factor
    :param budget_factor_list_rural: list of different ratios of facilities that are allowed to be opened in rural areas
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
    :param different_objective: true if objective changed to subtracting C for the closed facilities
    """
    # init
    if budget_factor_list_rural is None:
        budget_factor_list_rural = [0.1 * b for b in range(3, 10)]
    results_list = []
    print(budget_factor_list_rural)
    # build and optimize the relevant models
    for budget_factor_rural in budget_factor_list_rural:
        is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                   strict_assign_to_one, cap_factor, cutoff, max_access,
                                                   main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                   main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                   post_print_sol, post_log_file, post_preqlinearize, different_objective, budget_factor_rural, [], continous)
        if not is_feasible:
            print('Infeasible model')
            return None
        results_list.append(results)
        write_results_list([results], "separate_budget_"+str(budget_factor_rural*100)+"_" + str(budget_factor*100) + ".json")


    # write the results
    write_results_list(results_list, output_filename +  ".json")
    #write_seperate_budgets_results(results_list, users_and_facs_df, travel_dict, output_filename + ".xlsx", output_abs_path)

def write_seperate_budgets_results(results_list, users_and_facs_df, travel_dict, output_filename='rural_urban_budget_results.xlsx',
                           output_abs_path=None):
    """
    create an excel file that displays fairness and access results for different budgets for rual and urban
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the results are sorted in decreasing order by budget
    results_list = sorted(results_list, key=lambda d: -d['model_details']['proportion_rural'])
    nr_of_results = len(results_list)

    users_rural, facs_rural, users_urban, facs_urban = split_rural_urban(users_and_facs_df)

    # get relevant data
    budget_rural_list = [round(results['model_details']['proportion_rural'] * 100) for results in results_list]
    budget_string_list = [str(budget_rural_list[i // 4]) + "%" if i % 4 == 0 else None for i in
                          range(nr_of_results * 4)]
    regions = ['all', 'rural', 'urban', None] * nr_of_results
    access_list_users = []
    access_list_facs = []
    # how much do rural facilities get used by each type of user
    access_list_rural_facs = []
    access_list_urban_facs = []
    # how much do rural users go to each type of facility
    access_list_rural_users = []
    access_list_urban_users = []
    dof_list = []
    for results in results_list:
        access_list_users += [get_overall_access(results, users_and_facs_df, travel_dict, user_region='all') * 100,
                    get_overall_access(results, users_and_facs_df, travel_dict, user_region='rural')* 100,
                    get_overall_access(results, users_and_facs_df, travel_dict, user_region='urban') * 100,
                    None]

        access_list_facs += [get_overall_access(results, users_and_facs_df, travel_dict, facility_region='all') * 100,
                   get_overall_access(results, users_and_facs_df, travel_dict, facility_region='rural')*100,
                   get_overall_access(results, users_and_facs_df, travel_dict, facility_region='urban')*100,
                   None]
        access_list_rural_facs += [100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='rural', facility_region='rural', perspective='facs')*100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='urban', facility_region='rural', perspective='facs')*100,
                    None]
        access_list_urban_facs += [100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='rural', facility_region='urban', perspective='facs')*100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='urban', facility_region='urban', perspective='facs')*100,
                    None]
        access_list_rural_users += [100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='rural', facility_region='rural', perspective='users')*100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='rural', facility_region='urban', perspective='users')*100,
                    None]
        access_list_urban_users += [100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='urban', facility_region='rural', perspective='users')*100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='urban', facility_region='urban', perspective='users')*100,
                    None]
        dof_list += [get_dof_of_subset(results, users_and_facs_df, travel_dict) * 100,
                   get_dof_of_subset(results, users_and_facs_df, travel_dict, facs_rural) * 100,
                   get_dof_of_subset(results, users_and_facs_df, travel_dict, facs_urban) * 100,
                   None]
        

    # write the excel file
    data = {'Rural budget factor': budget_string_list, 'Region': regions, 'Access of users [%]': access_list_users,
    'Access of facilities [%]': access_list_facs,'Percentage of ... users in rural facilities': access_list_rural_facs
    ,'Percentage of ... users in urban facilities': access_list_urban_facs,'Percentage of ... facilities in rural users': access_list_rural_users
    ,'Percentage of ... facilities in urban users': access_list_urban_users,'DoF [%]': dof_list}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "/" + output_filename) as writer:
        df.to_excel(writer, index=False)


def write_seperate_budgets_results_compact(results_list, users_and_facs_df, travel_dict, output_filename='rural_urban_budget_results.xlsx',
                           output_abs_path=None):
    """
    create an excel file that displays fairness and access results for different budgets for rual and urban
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the results are sorted in decreasing order by budget
    results_list = sorted(results_list, key=lambda d: -d['model_details']['proportion_rural'])
    nr_of_results = len(results_list)

    users_rural, facs_rural, users_urban, facs_urban = split_rural_urban(users_and_facs_df)

    # get relevant data
    budget_rural_list = [round(results['model_details']['proportion_rural'] * 100) for results in results_list]
    budget_string_list = [str(budget_rural_list[i // 4]) + "%" if i % 4 == 0 else None for i in
                          range(nr_of_results * 4)]
    regions = ['all', 'rural', 'urban', None] * nr_of_results
    access_list_users = []

    # how much do users go to rural facilities 
    access_list_rural_users = []
    # how much do users go to urban facilities 
    access_list_urban_users = []
    dof_list = []
    for results in results_list:
        access_list_users += [get_overall_access(results, users_and_facs_df, travel_dict, user_region='all') * 100,
                    get_overall_access(results, users_and_facs_df, travel_dict, user_region='rural')* 100,
                    get_overall_access(results, users_and_facs_df, travel_dict, user_region='urban') * 100,
                    None]
        access_list_rural_users += [get_access_split(results, users_and_facs_df, travel_dict, user_region='all', facility_region='rural', perspective='users')*100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='rural', facility_region='rural', perspective='users')*100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='urban', facility_region='rural', perspective='users')*100,
                    None]
        access_list_urban_users += [get_access_split(results, users_and_facs_df, travel_dict, user_region='all', facility_region='urban', perspective='users')*100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='rural', facility_region='urban', perspective='users')*100,
                    get_access_split(results, users_and_facs_df, travel_dict, user_region='urban', facility_region='urban', perspective='users')*100,
                    None]
        dof_list += [get_dof_of_subset(results, users_and_facs_df, travel_dict) * 100,
                   get_dof_of_subset(results, users_and_facs_df, travel_dict, facs_rural) * 100,
                   get_dof_of_subset(results, users_and_facs_df, travel_dict, facs_urban) * 100,
                   None]
        

    # write the excel file
    data = {'Rural budget factor': budget_string_list, 'Region': regions, 'Access of users [%]': access_list_users
    ,'Percentage of rural facilities in ... users': access_list_rural_users
    ,'Percentage of urban facilities in ... users': access_list_urban_users,'DoF [%]': dof_list}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "/" + output_filename) as writer:
        df.to_excel(writer, index=False)



def save_randomly_closing_results(users_and_facs_df, travel_dict, users, facs,
                        cap_factor=1.5, budget_factor = 0.7, max_access=False, threads=1, tolerance=2e-3,
                        time_limit=20000, print_sol=False, log_file=None,
                        preqlinearize=-1, different_objective = False, output_filename='random_70.xlsx',
                        output_abs_path=None, num_runs = 10, open_facs_list = None):
    """
    solve the optimization model with fixed open facilities which are randomly chosen
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param cap_factor: factor by which all capacities are scaled
    :param budget: what percentage of facilities we still want to be open
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :param threads: number of threads used in the main optimization step
    :param tolerance: tolerance on the optimality of the solution used in the main optimization step
    :param time_limit: time limit in seconds for the optimization used in the main optimization step
    :param print_sol: boolean indicating whether the solution should be printed used in the main optimization step
    :param log_file: name of the log file for the main optimization step; if "None", no log file will be produced
    :param preqlinearize: setting for Gurobi's PreQLinearize parameter in the main optimization step
    :param different_objective: true if objective changed to subtracting C for the closed facilities
    :param num_runs: number of times we want to run this
    :param open_facs_list: list of lists of open facilities
    """
    # init
    budget = round(budget_factor * len(facs))
    if open_facs_list is None:
        open_facs_list = [random.sample(facs, budget)for i in range(num_runs)]
    results_list = []

    # build and optimize the relevant models
    for i in range(len(open_facs_list)):
        open_facs = open_facs_list[i]
        is_feasible, results = solve_facilities_open_fixed_model(users_and_facs_df, travel_dict, users, facs,
                        cap_factor, max_access, threads, tolerance,
                        time_limit, print_sol, log_file,
                        preqlinearize, open_facs, different_objective)
        if not is_feasible:
            print('Infeasible model')
            return None
        results_list.append(results)
        write_results_list([results], "random_"+str(budget_factor*100)+"_" + str(i) + ".json")


    # write the results
    write_results_list(results_list, "random_" + str(budget_factor*100) + ".json")
    # write_seperate_budgets_results(results_list, users_and_facs_df, travel_dict, output_filename, output_abs_path)


def save_randomly_closing_results_with_cutoff(users_and_facs_df, travel_dict, users, facs, output_filename='random_70.xlsx',
                          output_abs_path=None, budget_factor = 0.7, strict_assign_to_one=False, cap_factor=1.5,
                          cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3, main_time_limit=20000,
                          main_print_sol=False, main_log_file=None, main_preqlinearize=-1, post_threads=1,
                          post_tolerance=0.0, post_print_sol=False, post_log_file=None, post_preqlinearize=-1,
                        different_objective = False, num_runs = 10, closed_facs_list = None, continous = False):
    """
    solve the optimization model with fixed open facilities which are randomly chosen
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param cap_factor: factor by which all capacities are scaled
    :param budget: what percentage of facilities we still want to be open
    :param max_access: boolean indicating whether the objective should changed to maximizing access
    :param threads: number of threads used in the main optimization step
    :param tolerance: tolerance on the optimality of the solution used in the main optimization step
    :param time_limit: time limit in seconds for the optimization used in the main optimization step
    :param print_sol: boolean indicating whether the solution should be printed used in the main optimization step
    :param log_file: name of the log file for the main optimization step; if "None", no log file will be produced
    :param preqlinearize: setting for Gurobi's PreQLinearize parameter in the main optimization step
    :param different_objective: true if objective changed to subtracting C for the closed facilities
    :param num_runs: number of times we want to run this
    :param open_facs_list: list of lists of open facilities
    """
    # init
    budget = round(budget_factor * len(facs))
    closed = len(facs) - budget
    if closed_facs_list is None:
        closed_facs_list = [random.sample(facs, closed)for i in range(num_runs)]
    results_list = []

    # build and optimize the relevant models
    for i in range(len(closed_facs_list)):
        closed_facs = closed_facs_list[i]
        is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                   strict_assign_to_one, cap_factor, cutoff, max_access,
                                                   main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                   main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                   post_print_sol, post_log_file, post_preqlinearize, different_objective, -1, closed_facs, continous)
        if not is_feasible:
            print('Infeasible model')
            return None
        results_list.append(results)
        #write_results_list([results], "random_"+str(budget_factor*100)+"_" + str(i) + ".json")


    # write the results
    write_results_list(results_list, "largest_facilities_closed_number.json")
    write_overall_results(results_list, users_and_facs_df, travel_dict, output_filename=output_filename)
    # write_seperate_budgets_results(results_list, users_and_facs_df, travel_dict, output_filename, output_abs_path)

def write_stdev_results(results_list, users_and_facs_df, travel_dict, budget_name = 'proportion_rural', output_filename='stdev.xlsx',output_abs_path=None):
    """
    create an excel file that displays standard deviation for different budgets
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param budget_name: name of what we index into (e.g. budget_factor or proportion_rural)
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    regions_list = ["all", "rural", "urban"]
    util_stdev = []
    results_list = sorted(results_list, key=lambda d: -d['model_details'][budget_name])
    nr_of_results = len(results_list)
    budget_factor_list = [round(results['model_details'][budget_name] * 100) for results in results_list if results['model_details'][budget_name] != "None"]
    budget_string_list = [str(budget_factor_list[i // 4]) + "%" if i % 4 == 0 else None for i in
                            range(nr_of_results * 4)]
    regions = ['all', 'rural', 'urban', None] * nr_of_results
    for result in results_list:
        temp_array = []
        for region in regions_list:
            utilisation = get_utilization(result, users_and_facs_df, travel_dict, facility_region=region)
            if len(utilisation) < 2:
                temp_array.append(None)
                continue
            temp_array.append(statistics.stdev([x for x in utilisation.values()]))
        temp_array.append(None)
        util_stdev += temp_array
        

    print(util_stdev)
    print([result["model_details"]["proportion_rural"] for result in results_list])
    # write the excel file
    print(len(budget_string_list))
    print(len(regions))
    print(len(util_stdev))
    data = {'Rural budget [%]': budget_string_list, 'Region': regions, 'Standard deviation': util_stdev}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "/" + output_filename) as writer:
        df.to_excel(writer, index=False)

def save_overall_results_adapted_model(users_and_facs_df, travel_dict, users, facs, budget_factor_list, access_list, discount_factor_list, output_filename='overall_results.xlsx',
                         output_abs_path=None, percentiles=None, strict_assign_to_one=False,
                         cap_factor=1.5, cutoff=0.2, main_threads=1, main_tolerance=5e-3,
                         main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                         post_threads=1, post_tolerance=2e-3, post_print_sol=False, post_log_file=None,
                         post_preqlinearize=-1, different_objective = False):
    """
    create an excel file that displays broad results for different budgets after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor_list: list of budget factors
    :param access_list: list of accesses we want to (almost) achieve, as a percentage, so in [0,100]
    :param discount_factor_list: list of discount factors corresponding to accesses in access_list
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param percentiles: list of the three considered percentiles. Values must be between 0 and 100
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
    :param different_objective: True if closed facilities not taken into account
    """

    # init
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]

    results_list = []

    if(len(access_list) != len(discount_factor_list) or len(access_list) != len(budget_factor_list)) :
        print("Need same number of accesses and budget factors. ABORT")
        return

    total_population = sum(users_and_facs_df.at[i, 'population'] for i in users)

    # build and optimize the relevant models
    for i in range(len(access_list)):
        input_access = access_list[i]/ 100.0 * total_population
        discount_factor = discount_factor_list[i]
        budget_factor = budget_factor_list[i]
        is_feasible, results  = solve_model_with_access_bound(users_and_facs_df, travel_dict, users, facs, budget_factor, input_access, discount_factor, strict_assign_to_one,
                        cap_factor, cutoff, main_threads, main_tolerance,
                        main_time_limit, main_print_sol, main_log_file, main_preqlinearize,
                        post_threads, post_tolerance, post_print_sol, post_log_file,
                        post_preqlinearize, different_objective)
        if not is_feasible:
            print('Infeasible model')
            return None
        results_list.append(results)
        #write_results_list(results, "naive_model_results" + str(budget_factor)+".json")


    # write the results
    write_results_list(results_list, output_filename+".json")
    write_overall_results_adapted_model(results_list, users_and_facs_df, travel_dict, users, facs, output_filename + '.xlsx', output_abs_path, percentiles)
    #write_fairness_results(results_list, users_and_facs_df, travel_dict, "adapted_MIP_fairness.xlsx", output_abs_path)

def write_overall_results_adapted_model(results_list, users_and_facs_df, travel_dict, users, facs, output_filename='overall_results.xlsx',
                          output_abs_path=None, percentiles=None):
    """
    create an excel file that displays broad results for different budgets
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param percentiles: list of the three considered percentiles. Values must be between 0 and 100
    """

    # init
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]

    total_population = sum(users_and_facs_df.at[i, 'population'] for i in users)

    # make sure the results are sorted in decreasing order by budget
    sort_index = 'budget_factor'
    results_list = sorted(results_list, key=lambda d: (-d['model_details']['access'], -d['model_details']['discount_factor']))
    nr_of_results = len(results_list)

    # get relevant data
    access_input_list = [results['model_details']['access']/total_population * 100 for results in results_list]
    access_input_string_list = [str(access_input_list[i // 4]) + "%" if i % 4 == 0 else None for i in
                          range(nr_of_results * 4)]
    discount_factor_list = [results['model_details']['discount_factor'] for results in results_list]
    discount_factor_string_list = [discount_factor_list[i//4] if i % 4 == 0 else None for i in range(nr_of_results * 4)]

    budget_list = [len(results['solution_details']['open_facs'])/len(facs) for results in results_list]
    budget_factor_string_list = [budget_list[i//4]*100 if i % 4 == 0 else None for i in range(nr_of_results * 4)]
    
    regions = ['all', 'rural', 'urban', None] * nr_of_results
    access_list = []
    for results in results_list:
        access_list += [get_overall_access(results, users_and_facs_df, travel_dict, user_region='all') * 100,
                   get_overall_access(results, users_and_facs_df, travel_dict, user_region='rural') * 100,
                   get_overall_access(results, users_and_facs_df, travel_dict, user_region='urban') * 100,
                   None]
    distances_all = [list(get_distances_to_assigned(results, users_and_facs_df, user_region='all').values())
                     for results in results_list]
    distances_rural = [list(get_distances_to_assigned(results, users_and_facs_df, user_region='rural').values())
                       for results in results_list]
    distances_urban = [list(get_distances_to_assigned(results, users_and_facs_df, user_region='urban').values())
                       for results in results_list]
    distances_1 = []
    for i in range(nr_of_results):
        distances_1 += [safe_percentile(distances_all[i], percentiles[0]),
                        safe_percentile(distances_rural[i], percentiles[0]),
                        safe_percentile(distances_urban[i], percentiles[0]),
                        None]
    distances_2 = []
    for i in range(nr_of_results):
        distances_2 += [safe_percentile(distances_all[i], percentiles[1]),
                        safe_percentile(distances_rural[i], percentiles[1]),
                        safe_percentile(distances_urban[i], percentiles[1]),
                        None]
    distances_3 = []
    for i in range(nr_of_results):
        distances_3 += [safe_percentile(distances_all[i], percentiles[2]),
                        safe_percentile(distances_rural[i], percentiles[2]),
                        safe_percentile(distances_urban[i], percentiles[2]),
                        None]
    utilization_all = [list(get_utilization(results, users_and_facs_df, travel_dict, facility_region='all').values())
                       for results in results_list]
    utilization_rural = [
        list(get_utilization(results, users_and_facs_df, travel_dict, facility_region='rural').values())
        for results in results_list]
    utilization_urban = [
        list(get_utilization(results, users_and_facs_df, travel_dict, facility_region='urban').values())
        for results in results_list]
    utilization_1 = []
    for i in range(nr_of_results):
        utilization_1 += [safe_percentile(utilization_all[i], percentiles[0]) * 100,
                          safe_percentile(utilization_rural[i], percentiles[0]) * 100,
                          safe_percentile(utilization_urban[i], percentiles[0]) * 100,
                          None]
    utilization_2 = []
    for i in range(nr_of_results):
        utilization_2 += [safe_percentile(utilization_all[i], percentiles[1]) * 100,
                          safe_percentile(utilization_rural[i], percentiles[1]) * 100,
                          safe_percentile(utilization_urban[i], percentiles[1]) * 100,
                          None]
    utilization_3 = []
    for i in range(nr_of_results):
        utilization_3 += [safe_percentile(utilization_all[i], percentiles[2]) * 100,
                          safe_percentile(utilization_rural[i], percentiles[2]) * 100,
                          safe_percentile(utilization_urban[i], percentiles[2]) * 100,
                          None]

    # write the excel file
    data = {'Input access': access_input_string_list, 'Discount factor': discount_factor_string_list ,'Budget [%]': budget_factor_string_list, 'Region': regions, 'Overall access [%]': access_list,
            'Travel distance [km] - p' + str(percentiles[0]): distances_1,
            'Travel distance [km] - p' + str(percentiles[1]): distances_2,
            'Travel distance [km] - p' + str(percentiles[2]): distances_3,
            'Utilization [%] - p' + str(percentiles[0]): utilization_1,
            'Utilization [%] - p' + str(percentiles[1]): utilization_2,
            'Utilization [%] - p' + str(percentiles[2]): utilization_3}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "/" + output_filename) as writer:
        df.to_excel(writer, index=False)
