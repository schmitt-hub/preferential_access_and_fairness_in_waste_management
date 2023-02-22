"""
module containing functions for plotting results from a given results file
and functions for plotting results after creating the corresponding results
"""

from utils import *
from model import *
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import geopandas as gpd
import pandas as pd
import folium
from folium import plugins
from branca.element import Template, MacroElement
from matplotlib.patches import Patch
from typing import List, Tuple
import math


def plot_cap_vs_access(results, users_and_facs_df, travel_dict, output_filename='cap_vs_access.pdf',
                       output_abs_path=None, facility_region='all'):
    """
    make a capacity vs access scatter plot for the open facilities
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
    """
    # get relevant data
    open_facs = results['solution_details']['open_facs']
    cap_factor = results['model_details']['cap_factor']
    _, facility_region_list = get_region_list(facility_region=facility_region)
    capacity = {j: users_and_facs_df.at[j, 'capacity'] * cap_factor
                for j in open_facs if users_and_facs_df.at[j, 'regional spatial type'] in facility_region_list}
    access = get_access(results, users_and_facs_df, travel_dict, facility_region=facility_region)
    capacity_array = np.array(list(capacity.values())).reshape((-1, 1))
    access_array = np.array(list(access.values()))

    # fit a linear function
    model = LinearRegression(fit_intercept=False).fit(capacity_array, access_array)

    # plot the figure
    tnr_font = {'fontname': 'Times New Roman'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    plt.scatter(capacity_array, access_array)
    plt.plot(capacity_array, model.coef_ * capacity_array, "r")
    plt.xlabel('Capacity', fontsize="xx-large", **tnr_font)
    plt.ylabel('Access', fontsize="xx-large", **tnr_font)
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.xlim([0, None])
    plt.ylim([0, None])

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "\\" + output_filename, dpi=1200)
    plt.clf()


def save_cap_vs_access_plot(users_and_facs_df, travel_dict, users, facs, output_filename='cap_vs_access.pdf',
                            output_abs_path=None, facility_region='all', budget_factor=1.0, strict_assign_to_one=False,
                            cap_factor=1.5, cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3,
                            main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                            post_threads=1, post_tolerance=0.0, post_print_sol=False, post_log_file=None,
                            post_preqlinearize=-1):
    """
    make a capacity vs access scatter plot for the open facilities after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
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
    """
    # build and optimize the model
    is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                               strict_assign_to_one, cap_factor, cutoff, max_access,
                                               main_threads, main_tolerance, main_time_limit, main_print_sol,
                                               main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                               post_print_sol, post_log_file, post_preqlinearize)
    if not is_feasible:
        print('Infeasible model')
        return None

    # plot the figure
    plot_cap_vs_access(results, users_and_facs_df, travel_dict, output_filename, output_abs_path, facility_region)


def plot_overall_access(results_list, users_and_facs_df, travel_dict, output_filename='overall_access.pdf',
                        output_abs_path=None):
    """
    plot the overall access for different budgets
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the results are sorted in increasing order by budget
    results_list = sorted(results_list, key=lambda d: d['model_details']['budget_factor'])

    # get relevant data
    overall_access_dict = {
        region: [get_overall_access(results, users_and_facs_df, travel_dict, user_region=region) * 100 for results in
                 results_list] for region in ['all', 'rural', 'urban']}
    budget_factor_list = [round(results['model_details']['budget_factor'] * 100) for results in results_list]

    # plot the figure
    tnr_font = {'fontname': 'Times New Roman'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    plt.plot(budget_factor_list, overall_access_dict['all'], label='all', color='blue')
    plt.plot(budget_factor_list, overall_access_dict['rural'], label='rural', color='green')
    plt.plot(budget_factor_list, overall_access_dict['urban'], label='urban', color='red')
    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 'xx-large'})
    plt.xlabel('Budget [%]', fontsize="xx-large", **tnr_font)
    plt.ylabel('Overall access [%]', fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.xlim([None, budget_factor_list[-1]])

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "\\" + output_filename, dpi=1200)
    plt.clf()


def save_overall_access_plot(users_and_facs_df, travel_dict, users, facs, output_filename='overall_access.pdf',
                             output_abs_path=None, budget_factor_list=None, strict_assign_to_one=False, cap_factor=1.5,
                             cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3, main_time_limit=20000,
                             main_print_sol=False, main_log_file=None, main_preqlinearize=-1, post_threads=1,
                             post_tolerance=0.0, post_print_sol=False, post_log_file=None, post_preqlinearize=-1):
    """
    plot the overall access for different budgets after creating the corresponding results
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

    # build and optimize the model
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

    # plot the figure
    plot_overall_access(results_list, users_and_facs_df, travel_dict, output_filename, output_abs_path)


def plot_utilization_percentiles(results_list, users_and_facs_df, travel_dict,
                                 output_filename='utilization_percentiles.pdf', output_abs_path=None, percentiles=None):
    """
    plot the utilization of open facilities for three different percentiles for different budgets
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param percentiles: list of the three considered percentiles. Values must be between 0 and 100
    """
    # init
    regions = ['all', 'rural', 'urban']
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]

    # make sure the results are sorted in increasing order by budget
    results_list = sorted(results_list, key=lambda d: d['model_details']['budget_factor'])
    nr_of_results = len(results_list)

    # get relevant data
    utilization_dict = \
        {region: [[u * 100
                   for u in get_utilization(results, users_and_facs_df, travel_dict, facility_region=region).values()]
                  for results in results_list] for region in regions}
    budget_factor_list = [round(results['model_details']['budget_factor'] * 100) for results in results_list]

    # compute the percentiles of the utilization of open facilities for each instance and each region
    percentiles_dict = {region: {p: [safe_percentile(utilization_dict[region][i], p) for i in range(nr_of_results)]
                                 for p in percentiles}
                        for region in regions}

    # plot the figure
    tnr_font = {'fontname': 'Times New Roman'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    plt.plot(budget_factor_list, percentiles_dict['all'][percentiles[0]], ls="dashed", color='blue')
    plt.plot(budget_factor_list, percentiles_dict['rural'][percentiles[0]], ls="dashed", color='green')
    plt.plot(budget_factor_list, percentiles_dict['urban'][percentiles[0]], ls="dashed", color='red')
    line_all, = plt.plot(budget_factor_list, percentiles_dict['all'][percentiles[1]], label='all', color='blue')
    line_rural, = plt.plot(budget_factor_list, percentiles_dict['rural'][percentiles[1]], label='rural', color='green')
    line_urban, = plt.plot(budget_factor_list, percentiles_dict['urban'][percentiles[1]], label='urban', color='red')
    plt.plot(budget_factor_list, percentiles_dict['all'][percentiles[2]], ls="dotted", color='blue', linewidth=2)
    plt.plot(budget_factor_list, percentiles_dict['rural'][percentiles[2]], ls="dotted", color='green', linewidth=2)
    plt.plot(budget_factor_list, percentiles_dict['urban'][percentiles[2]], ls="dotted", color='red', linewidth=2)
    # plt.legend(handles=[line_all, line_rural, line_urban], fontsize="xx-large",
    #            prop={'family': 'Times New Roman', 'size': 'xx-large'})
    legend = plt.legend(handles=[line_all, line_rural, line_urban], fontsize="xx-large",
                        prop={'family': 'Times New Roman', 'size': 'xx-large'}, loc='upper right',
                        bbox_to_anchor=(0.7, 1))
    # dummy lines for legend
    line_10, = plt.plot([], [], label=str(percentiles[0])+'th percentile', ls="dashed", color='black')
    line_50, = plt.plot([], [], label=str(percentiles[1])+'th percentile', color='black')
    line_90, = plt.plot([], [], label=str(percentiles[2])+'th percentile', ls="dotted", color='black')
    plt.gca().add_artist(legend)
    plt.legend(handles=[line_10, line_50, line_90], fontsize="xx-large",
               prop={'family': 'Times New Roman', 'size': 'xx-large'}, loc='upper right')
    plt.xlabel('Budget [%]', fontsize="xx-large", **tnr_font)
    plt.ylabel('Utilization [%]', fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.xlim([None, budget_factor_list[-1]])

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "\\" + output_filename, dpi=1200)
    plt.clf()


def save_utilization_percentiles_plot(users_and_facs_df, travel_dict, users, facs,
                                      output_filename='utilization_percentiles.pdf', output_abs_path=None,
                                      percentiles=None, budget_factor_list=None, strict_assign_to_one=False,
                                      cap_factor=1.5, cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3,
                                      main_time_limit=20000, main_print_sol=False, main_log_file=None,
                                      main_preqlinearize=-1, post_threads=1, post_tolerance=0.0, post_print_sol=False,
                                      post_log_file=None, post_preqlinearize=-1):
    """
    plot the utilization of open facilities for three different percentiles for different budgets
     after creating the corresponding results
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
    """
    # init
    if budget_factor_list is None:
        budget_factor_list = [0.1 * b for b in range(3, 11)]
    results_list = []

    for budget_factor in budget_factor_list:
        # build and optimize the model
        is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                   strict_assign_to_one, cap_factor, cutoff, max_access,
                                                   main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                   main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                   post_print_sol, post_log_file, post_preqlinearize)
        if not is_feasible:
            print('Infeasible model')
            return None
        results_list.append(results)

    # plot the figure
    plot_utilization_percentiles(results_list, users_and_facs_df, travel_dict, output_filename, output_abs_path,
                                 percentiles)


def plot_utilization_distribution(results_list, users_and_facs_df, travel_dict,
                                  output_filename='utilization_distribution.pdf', output_abs_path=None,
                                  facility_region='all'):
    """
    plot the the distribution in the utilization of open facilities for different budgets
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
    """
    if len(results_list) > 12:
        print('Number of instances exceeds the feasible amount (12)')
        return None

    # make sure the results are sorted in increasing order by budget
    results_list = sorted(results_list, key=lambda d: d['model_details']['budget_factor'])

    # initialize the figure
    tnr_font = {'fontname': 'Times New Roman'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    markers = {0: 'o', 1: '', 2: 7}
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}
    fig, ax = plt.subplots()
    x = np.linspace(0, 100, 250)

    for i, results in enumerate(results_list):
        # get relevant data
        utilization_list = [100 * u for u in get_utilization(results, users_and_facs_df, travel_dict,
                                                             facility_region=facility_region).values()]
        budget_factor_list = [round(results['model_details']['budget_factor'] * 100) for results in results_list]
        density = stats.gaussian_kde(utilization_list)

        # plot the figure
        plt.plot(x, density(x), label=str(budget_factor_list[i]) + '% ' + 'Budget',
                 marker=markers[i // 4], markevery=10, color=colors[i % 4])
        plt.xlabel('Utilization [%]', fontsize="xx-large", **tnr_font)
        plt.ylabel('Open facilities [%]', fontsize="xx-large", **tnr_font)
        plt.ylim(0, 0.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol=None))
        plt.legend(loc='upper center', ncol=3, prop={'family': 'Times New Roman', 'size': 'xx-large'})
        plt.tick_params(labelsize="x-large")
        plt.xticks(**tnr_font)
        plt.yticks(**tnr_font)
        plt.xlim(0, 100)

        # save the figure
        if not output_abs_path:
            output_abs_path = os.getcwd() + "\\own_results"
        if not os.path.exists(output_abs_path):
            os.makedirs(output_abs_path)
        plt.savefig(output_abs_path + "\\" + output_filename, dpi=1200)
    plt.clf()


def save_utilization_distribution_plot(users_and_facs_df, travel_dict, users, facs,
                                       output_filename='utilization_distribution.pdf', output_abs_path=None,
                                       budget_factor_list=None, facility_region='all', strict_assign_to_one=False,
                                       cap_factor=1.5, cutoff=0.2, max_access=False, main_threads=1,
                                       main_tolerance=5e-3, main_time_limit=20000, main_print_sol=False,
                                       main_log_file=None, main_preqlinearize=-1, post_threads=1, post_tolerance=0.0,
                                       post_print_sol=False, post_log_file=None, post_preqlinearize=-1):
    """
    plot the the distribution in the utilization of open facilities for different budgets
     after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param budget_factor_list: list of different ratios of facilities that are allowed to be opened
    :param facility_region: string indicating the considered region of location for the facilities:
       "urban", "rural" or "all"
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
    elif len(budget_factor_list) > 12:
        print('Number of instances exceeds the feasible amount (12)')
        return None
    results_list = []

    for budget_factor in budget_factor_list:
        # build and optimize the model
        is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                   strict_assign_to_one, cap_factor, cutoff, max_access,
                                                   main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                   main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                   post_print_sol, post_log_file, post_preqlinearize)
        if not is_feasible:
            print('Infeasible model')
            return None
        results_list.append(results)

    # plot the figure
    plot_utilization_distribution(results_list, users_and_facs_df, travel_dict, output_filename, output_abs_path,
                                  facility_region)


def plot_distance_percentiles(results_list, users_and_facs_df, output_filename='distance_percentiles.pdf',
                              output_abs_path=None, percentiles=None, facility_region='all'):
    """
    plot the distance to the assigned facilities for three different percentiles for different budgets
    :param results_list: list of dictionaries of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param percentiles: list of the three considered percentiles. Values must be between 0 and 100
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
    """
    # init
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]
    user_regions = ['all', 'rural', 'urban']

    # make sure the results are sorted in increasing order by budget
    results_list = sorted(results_list, key=lambda d: d['model_details']['budget_factor'])
    nr_of_results = len(results_list)

    # get relevant data
    distance_dict = {user_region: [
        [d for d in get_distances_to_assigned(results, users_and_facs_df, user_region=user_region,
                                              facility_region=facility_region).values()]
        for results in results_list] for user_region in user_regions}
    budget_factor_list = [round(results['model_details']['budget_factor'] * 100) for results in results_list]

    # compute the percentiles of the utilization of open facilities for each instance and each region
    percentiles_dict = {region: {p: [safe_percentile(distance_dict[region][i], p) for i in range(nr_of_results)]
                                 for p in percentiles}
                        for region in user_regions}

    # plot the figure
    tnr_font = {'fontname': 'Times New Roman'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    plt.plot(budget_factor_list, percentiles_dict['all'][percentiles[0]], ls="dashed", color='blue')
    plt.plot(budget_factor_list, percentiles_dict['rural'][percentiles[0]], ls="dashed", color='green')
    plt.plot(budget_factor_list, percentiles_dict['urban'][percentiles[0]], ls="dashed", color='red')
    line_all, = plt.plot(budget_factor_list, percentiles_dict['all'][percentiles[1]], label='all', color='blue')
    line_rural, = plt.plot(budget_factor_list, percentiles_dict['rural'][percentiles[1]], label='rural', color='green')
    line_urban, = plt.plot(budget_factor_list, percentiles_dict['urban'][percentiles[1]], label='urban', color='red')
    plt.plot(budget_factor_list, percentiles_dict['all'][percentiles[2]], ls="dotted", color='blue', linewidth=2)
    plt.plot(budget_factor_list, percentiles_dict['rural'][percentiles[2]], ls="dotted", color='green', linewidth=2)
    plt.plot(budget_factor_list, percentiles_dict['urban'][percentiles[2]], ls="dotted", color='red', linewidth=2)
    legend = plt.legend(handles=[line_all, line_rural, line_urban], fontsize="xx-large",
                        prop={'family': 'Times New Roman', 'size': 'xx-large'}, loc='upper right',
                        bbox_to_anchor=(0.7, 1))
    # dummy lines for legend
    line_10, = plt.plot([], [], label=str(percentiles[0])+'th percentile', ls="dashed", color='black')
    line_50, = plt.plot([], [], label=str(percentiles[1])+'th percentile', color='black')
    line_90, = plt.plot([], [], label=str(percentiles[2])+'th percentile', ls="dotted", color='black')
    plt.gca().add_artist(legend)
    plt.legend(handles=[line_10, line_50, line_90], fontsize="xx-large",
               prop={'family': 'Times New Roman', 'size': 'xx-large'}, loc='upper right')
    plt.xlabel('Budget [%]', fontsize="xx-large", **tnr_font)
    plt.ylabel('Distance to assigned facility [km]', fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.xlim([None, budget_factor_list[-1]])

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "\\" + output_filename, dpi=1200)
    plt.clf()


def save_distance_percentiles_plot(users_and_facs_df, travel_dict, users, facs,
                                   output_filename='distance_percentiles.pdf', output_abs_path=None, percentiles=None,
                                   budget_factor_list=None, facility_region='all', strict_assign_to_one=False,
                                   cap_factor=1.5, cutoff=0.2, max_access=False, main_threads=1, main_tolerance=5e-3,
                                   main_time_limit=20000, main_print_sol=False, main_log_file=None,
                                   main_preqlinearize=-1, post_threads=1, post_tolerance=0.0, post_print_sol=False,
                                   post_log_file=None, post_preqlinearize=-1):
    """
    plot the distance to the assigned facilities for three different percentiles for different budgets
     after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param percentiles: list of the three considered percentiles. Values must be between 0 and 100
    :param budget_factor_list: list of different ratios of facilities that are allowed to be opened
    :param facility_region: string indicating the considered region of location for the facilities:
       "urban", "rural" or "all"
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
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]
    results_list = []

    for budget_factor in budget_factor_list:
        # build and optimize the model
        is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                   strict_assign_to_one, cap_factor, cutoff, max_access,
                                                   main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                   main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                   post_print_sol, post_log_file, post_preqlinearize)
        if not is_feasible:
            print('Infeasible model')
            return None
        results_list.append(results)

    # plot the figure
    plot_distance_percentiles(results_list, users_and_facs_df, output_filename, output_abs_path, percentiles,
                              facility_region)


def plot_strict_vs_loose(strict_results_list, loose_results_list, output_filename='strict_vs_loose.pdf',
                         output_abs_path=None):
    """
    make a plot comparing the computational performance for the two different implementations of the
    assign-to-one constraint
    :param strict_results_list: list of dictionaries of the results obtained when modeling the assign-to-one constraint
        as an equality
    :param loose_results_list: list of dictionaries of the results obtained when modeling the assign-to-one constraint
        as an inequality
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the results are sorted in increasing order by budget
    strict_results_list = sorted(strict_results_list, key=lambda d: d['model_details']['budget_factor'])
    loose_results_list = sorted(loose_results_list, key=lambda d: d['model_details']['budget_factor'])

    # get relevant data
    strict_obj_list = [results['solution_details']['objective_value'] for results in strict_results_list]
    strict_solving_time_list = [results['solution_details']['solving_time'] for results in strict_results_list]
    strict_budget_factor_list = [round(results['model_details']['budget_factor'] * 100)
                                 for results in strict_results_list]
    loose_obj_list = [results['solution_details']['objective_value'] for results in loose_results_list]
    loose_solving_time_list = [results['solution_details']['solving_time'] for results in loose_results_list]
    loose_budget_factor_list = [round(results['model_details']['budget_factor'] * 100)
                                for results in loose_results_list]

    # plot the figure
    tnr_font = {'fontname': 'Times New Roman'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Budget [%]', fontsize="xx-large", **tnr_font)
    ax1.set_ylabel('Solving times [s]', fontsize="xx-large", color="blue", **tnr_font)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize="x-large")
    ax1.tick_params(axis='x', labelsize="x-large")
    for tick in ax1.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax1.get_yticklabels():
        tick.set_fontname("Times New Roman")
    plt.plot(loose_budget_factor_list, loose_solving_time_list, label='Loose', color="blue")
    plt.plot(strict_budget_factor_list, strict_solving_time_list, label='Strict', color="blue", ls="dashed")
    ax1.set_ylim([0, None])
    ax2 = ax1.twinx()
    ax2.set_ylabel('Objective value', fontsize="xx-large", color="red", **tnr_font)
    ax2.tick_params(axis='y', labelcolor="red", labelsize="x-large")
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for tick in ax2.get_yticklabels():
        tick.set_fontname("Times New Roman")
    plt.plot(loose_budget_factor_list, loose_obj_list, color="red")
    plt.plot(strict_budget_factor_list, strict_obj_list, color="red", ls="dashed")
    # dummy lines to put on legend
    line_strict, = plt.plot([], [], label='Strict', ls="dashed", color='black')
    line_loose, = plt.plot([], [], label='Loose', color='black')
    plt.legend(fontsize="xx-large", handles=[line_strict, line_loose],
               prop={'family': 'Times New Roman', 'size': 'xx-large'})
    plt.xlim([None, max(loose_budget_factor_list[-1], strict_budget_factor_list[-1])])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "\\" + output_filename, dpi=1200)
    plt.clf()


def save_strict_vs_loose_plot(users_and_facs_df, travel_dict, users, facs, output_filename='strict_vs_loose.pdf',
                              output_abs_path=None, budget_factor_list=None, cap_factor=1.5, cutoff=0.2,
                              max_access=False, main_threads=1, main_tolerance=5e-3, main_time_limit=20000,
                              main_print_sol=False, main_log_file=None, main_preqlinearize=-1, post_threads=1,
                              post_tolerance=0.0, post_print_sol=False, post_log_file=None, post_preqlinearize=-1):
    """
    make a plot comparing the computational performance for the two different implementations of the
     assign-to-one constraint after creating the corresponding results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param budget_factor_list: list of different ratios of facilities that are allowed to be opened
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
    strict_results_list = []
    loose_results_list = []

    for budget_factor in budget_factor_list:
        for strict_assign_to_one in [True, False]:
            # build and optimize the model
            is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                       strict_assign_to_one, cap_factor, cutoff, max_access,
                                                       main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                       main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                       post_print_sol, post_log_file, post_preqlinearize)
            if not is_feasible:
                print('Infeasible model')
                return None
            if strict_assign_to_one:
                strict_results_list.append(results)
            else:
                loose_results_list.append(results)

    # plot the figure
    plot_strict_vs_loose(strict_results_list, loose_results_list, output_filename, output_abs_path)


def plot_cutoff_vs_nocutoff(cutoff_results_list, nocutoff_results_list, output_filename='cutoff_vs_nocutoff.pdf',
                            output_abs_path=None):
    """
    make a plot comparing the computational performance of the reduced model with the true model
    :param cutoff_results_list: list of dictionaries of the results obtained solving a reduced model
    :param nocutoff_results_list: list of dictionaries of the results obtained solving the true model
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    """
    # make sure the results are sorted in increasing order by budget
    cutoff_results_list = sorted(cutoff_results_list, key=lambda d: d['model_details']['budget_factor'])
    nocutoff_results_list = sorted(nocutoff_results_list, key=lambda d: d['model_details']['budget_factor'])

    # get relevant data
    if not len(set(results['model_details']['cutoff'] for results in cutoff_results_list)) == 1:
        print('Not all instances use the same cutoff')
        return None
    else:
        cutoff = cutoff_results_list[0]['model_details']['cutoff'] * 100
    cutoff_obj_list = [results['solution_details']['objective_value'] for results in cutoff_results_list]
    cutoff_solving_time_list = [results['solution_details']['solving_time'] for results in cutoff_results_list]
    cutoff_budget_factor_list = [round(results['model_details']['budget_factor'] * 100)
                                 for results in cutoff_results_list]
    nocutoff_obj_list = [results['solution_details']['objective_value'] for results in nocutoff_results_list]
    nocutoff_solving_time_list = [results['solution_details']['solving_time'] for results in nocutoff_results_list]
    nocutoff_budget_factor_list = [round(results['model_details']['budget_factor'] * 100)
                                   for results in nocutoff_results_list]

    # plot the figure
    tnr_font = {'fontname': 'Times New Roman'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Budget [%]', fontsize="xx-large", **tnr_font)
    ax1.set_ylabel('Solving times [s]', fontsize="xx-large", color="blue", **tnr_font)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize="x-large")
    ax1.tick_params(axis='x', labelsize="x-large")
    for tick in ax1.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax1.get_yticklabels():
        tick.set_fontname("Times New Roman")
    plt.plot(cutoff_budget_factor_list, cutoff_solving_time_list, label=str(round(cutoff)) + '% cutoff', color="blue")
    plt.plot(nocutoff_budget_factor_list, nocutoff_solving_time_list, label='0% cutoff', color="blue", ls="dashed")
    ax1.set_ylim([0, None])
    ax2 = ax1.twinx()
    ax2.set_ylabel('Objective value', fontsize="xx-large", color="red", **tnr_font)
    ax2.tick_params(axis='y', labelcolor="red", labelsize="x-large")
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for tick in ax2.get_yticklabels():
        tick.set_fontname("Times New Roman")
    plt.plot(cutoff_budget_factor_list, cutoff_obj_list, color="red")
    plt.plot(nocutoff_budget_factor_list, nocutoff_obj_list, color="red", ls="dashed")
    # dummy lines to put on legend
    line_cutoff, = plt.plot([], [], label=str(round(cutoff)) + '% cutoff', color='black')
    line_nocutoff, = plt.plot([], [], label='0% cutoff', ls="dashed", color='black')
    plt.legend(fontsize="xx-large", handles=[line_nocutoff, line_cutoff],
               prop={'family': 'Times New Roman', 'size': 'xx-large'}, loc='upper right', bbox_to_anchor=(1, 0.95))
    plt.xlim([None, max(cutoff_budget_factor_list[-1], nocutoff_budget_factor_list[-1])])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "\\" + output_filename, dpi=1200)
    plt.clf()


def save_cutoff_vs_nocutoff_plot(users_and_facs_df, travel_dict, users, facs, output_filename='cutoff_vs_nocutoff.pdf',
                                 output_abs_path=None, budget_factor_list=None, strict_assign_to_one=False,
                                 cap_factor=1.5, compared_cutoff=0.2, max_access=False, main_threads=1,
                                 main_tolerance=5e-3, main_time_limit=20000, main_print_sol=False, main_log_file=None,
                                 main_preqlinearize=-1, post_threads=1, post_tolerance=0.0, post_print_sol=False,
                                 post_log_file=None, post_preqlinearize=-1):
    """
    make a plot comparing the computational performance of the reduced model with the true model
     after creating the corresponding results
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
    :param compared_cutoff: travel combinations with a probability smaller than this value will be removed
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
    cutoff_results_list = []
    nocutoff_results_list = []

    for budget_factor in budget_factor_list:
        for cutoff in [0.0, compared_cutoff]:
            # build and optimize the model
            is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor,
                                                       strict_assign_to_one, cap_factor, cutoff, max_access,
                                                       main_threads, main_tolerance, main_time_limit, main_print_sol,
                                                       main_log_file, main_preqlinearize, post_threads, post_tolerance,
                                                       post_print_sol, post_log_file, post_preqlinearize)
            if not is_feasible:
                print('Infeasible model')
                return None
            if cutoff > 0.0:
                cutoff_results_list.append(results)
            else:
                nocutoff_results_list.append(results)

    # plot the figure
    plot_cutoff_vs_nocutoff(cutoff_results_list, nocutoff_results_list, output_filename, output_abs_path)


def plot_with_percentiles_from_file(input_file, column_name_x = "Budget", column_name_y = "Travel distance [km]", x_label = "Budget", y_label = "Travel distance [km]", input_abs_path = None, output_abs_path = None, output_filename = "greedy_travel.pdf"):
    """
    make a plot of distances by reading in the values from a file (instead of recomputing)
    :param input_file: the name of the input file
    :param column_name_x: the column name we want the data for the x-axis being read from
    :param column_name_y: the column name we want the data for the y-axis being read from
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param input_abs_path: the absolute path of where the input file is
    :param output_abs_path: the absolute path of where the output file should be written
    :param output_filename: the name the file we write the plot to should have
    """
    if input_abs_path == None:
        input_abs_path = os.getcwd() + "/own_results"
    travel_results = pd.read_excel(input_abs_path + "/" + input_file)

    percentiles = [10, 50, 90]
    user_regions = ['all', 'rural', 'urban']

    # make sure the results are sorted in increasing order by budget
    nr_of_results = len(travel_results)
    if "None" in [budget for budget in travel_results[column_name_x]]:
        nr_of_results = nr_of_results - 3
    budget_factor_list = [float(str(budget)[:-1]) for budget in travel_results[column_name_x] if (not (type(budget) == float and math.isnan(budget))) and budget != "None"]
    budget_factor_list.reverse()
    # compute the percentiles of the utilization of open facilities for each instance and each region
    percentiles_dict = {region: {p: list(reversed([travel_results[column_name_y+ ' - p' + str(p)][i] for i in range(nr_of_results) if travel_results['Region'][i] == region]))
                                    for p in percentiles}
                        for region in user_regions}
    # plot the figure
    tnr_font = {'fontname': 'DejaVu Sans'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    plt.plot(budget_factor_list, percentiles_dict['all'][percentiles[0]],ls="dotted", color='blue')
    plt.plot(budget_factor_list, percentiles_dict['rural'][percentiles[0]],ls="dotted", color='green')
    plt.plot(budget_factor_list, percentiles_dict['urban'][percentiles[0]], ls="dotted", color='red')
    plt.plot(budget_factor_list, percentiles_dict['all'][percentiles[1]], ls="dashed",  color='blue')
    plt.plot(budget_factor_list, percentiles_dict['rural'][percentiles[1]], ls="dashed", color='green')
    plt.plot(budget_factor_list, percentiles_dict['urban'][percentiles[1]], ls="dashed", color='red')
    line_all, = plt.plot(budget_factor_list, percentiles_dict['all'][percentiles[2]], label='all', color='blue', linewidth=2)
    line_rural, =plt.plot(budget_factor_list, percentiles_dict['rural'][percentiles[2]], label='rural', color='green', linewidth=2)
    line_urban, = plt.plot(budget_factor_list, percentiles_dict['urban'][percentiles[2]], label='urban', color='red', linewidth=2)
    blue_patch = mpatches.Patch(color='blue', label='all')
    green_patch = mpatches.Patch(color='green', label='rural')
    red_patch = mpatches.Patch(color='red', label='urban')
    legend = plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize="xx-large",prop={'family': 'DejaVu Sans', 'size': 'xx-large'},
        loc="upper left")
    # dummy lines for legend
    line_10, = plt.plot([], [], label=str(percentiles[0])+'th percentile', ls="dotted", color='black')
    line_50, = plt.plot([], [], label=str(percentiles[1])+'th percentile', ls="dashed", color='black')
    line_90, = plt.plot([], [], label=str(percentiles[2])+'th percentile', color='black')
    plt.gca().add_artist(legend)
    plt.legend(handles=[line_10, line_50, line_90], fontsize="xx-large",
                prop={'family': 'DejaVu Sans', 'size': 'xx-large'}, bbox_to_anchor = (0.8,1.0))
    plt.xlabel(x_label, fontsize="xx-large", **tnr_font)
    plt.ylabel(y_label, fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    #plt.xlim([None, budget_factor_list[-1]])
    plt.xlim()
    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "/" + output_filename, dpi=1200)
    plt.clf()

def plot_column_from_file(input_file, column_name_x = "Budget", column_name_y = "Proportion open [%]", x_label = "Budget", y_label = "Proportion open [%]", input_abs_path = None, output_abs_path = None,
                            output_filename = "greedy_results_proportion_open.pdf"):
    """
    make a plot of a column with all, rural and urban data by reading in the values from a file (instead of recomputing)
    :param input_file: the name of the input file
    :param column_name_x: the column name we want the data for the x-axis being read from
    :param column_name_y: the column name we want the data for the y-axis being read from
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param input_abs_path: the absolute path of where the input file is
    :param ioutput_abs_path: the absolute path of where the output file should be written
    :param output_filename: the name the file we write the plot to should have
    """
    if input_abs_path == None:
        input_abs_path = os.getcwd() + "/own_results"
    file = pd.read_excel(input_abs_path + "/" + input_file)

    user_regions = ['all', 'rural', 'urban']

    # make sure the results are sorted in increasing order by budget
    nr_of_results = len(file)
    plot_none = False
    if "None" in [budget for budget in file[column_name_x]]:
        nr_of_results = nr_of_results - 3
        plot_none = True
    budget_factor_list = [float(str(budget)[:-1]) for budget in file[column_name_x] if (not (type(budget) == float and math.isnan(budget))) and budget != "None"]
    budget_factor_list.reverse()
    print(budget_factor_list)
    # compute the percentiles of the utilization of open facilities for each instance and each region
    data = {region: list(reversed([file[column_name_y][i] for i in range(nr_of_results) if file['Region'][i] == region]))
                        for region in user_regions}
    # plot the figure
    tnr_font = {'fontname': 'DejaVu Sans'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    
    line_all, = plt.plot(budget_factor_list, data['all'], label='all', color='blue')
    line_rural, = plt.plot(budget_factor_list, data['rural'], label='rural', color='green')
    line_urban, = plt.plot(budget_factor_list, data['urban'], label='urban', color='red')
    lines = [line_all, line_rural, line_urban]
    blue_patch = mpatches.Patch(color='blue', label='all')
    green_patch = mpatches.Patch(color='green', label='rural')
    red_patch = mpatches.Patch(color='red', label='urban')
    legend = plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize="xx-large",prop={'family': 'DejaVu Sans', 'size': 'xx-large'}, loc='lower left',
        )
    if plot_none:
        data_none = {region: [file[column_name_y][i] for i in range(nr_of_results, nr_of_results + 3) if file['Region'][i] == region]*len(data['all'])
                        for region in user_regions}
        line_all_none, = plt.plot(budget_factor_list, data_none['all'], label='all', color='blue', ls="dashed")
        line_rural_none, = plt.plot(budget_factor_list, data_none['rural'], label='rural', color='green', ls="dashed")
        line_urban_none, = plt.plot(budget_factor_list, data_none['urban'], label='urban', color='red', ls="dashed")
        
        dummy_separate, = plt.plot([], [], label='Rural budget', color='black')
        dummy_no_budget, = plt.plot([], [], label='No rural budget', color='black', ls ="dashed")
        plt.gca().add_artist(legend)
        plt.legend(handles=[dummy_separate, dummy_no_budget], fontsize="xx-large",
                prop={'family': 'DejaVu Sans', 'size': 'xx-large'}, bbox_to_anchor=(0.7, 0.3))
        

    
    plt.xlabel(x_label, fontsize="xx-large", **tnr_font)
    plt.ylabel(y_label, fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.xlim()

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "/" + output_filename, dpi=1200)
    plt.clf()

def plot_2_columns_from_file(input_file_1, input_file_2, column_name = "DoF' [%]", name_1 = "Naive", name_2 = "Greedy", colours = ["red", "blue"], reverse = False,
                             x_axis_name = "Budget factor [%]", input_abs_path = None, output_abs_path = None, output_filename = "dof_prime_compare.pdf"):
    """
    make a plot of a column in two different files by reading in the values from a file (instead of recomputing)
    :param input_file_1: the name of the first input file (number of rows determined by this first file)
    :param input_file_2: the name of the second input file
    :param column_name: name of the column we want to plot on the x-axis
    :param name_1: the name of the first file as should appear in the legend
    :param name_2: the name of the second file as should appear in the legend
    :param input_abs_path: the absolute path of where the input file is
    :param ioutput_abs_path: the absolute path of where the output file should be written
    :param output_filename: the name the file we write the plot to should have
    :param reverse: true if we want to go from few open to a lot open, false otherwise
    """
    if input_abs_path == None:
        input_abs_path = os.getcwd() + "/own_results"
    results = [pd.read_excel(input_abs_path + "/" + input_file_1), pd.read_excel(input_abs_path + "/" + input_file_2)]

    # make sure the results are sorted in increasing order by budget
    nr_of_results = len(results[0])
    budget_factor_list = [int(str(budget)[:-1]) for budget in results[0]['Budget'] if type(budget) != float]
    if reverse:
        budget_factor_list.reverse()
    else:
        budget_factor_list = [100-x for x in budget_factor_list]
    # compute the percentiles of the utilization of open facilities for each instance and each region
    if reverse:
        data = {j: list(reversed([results[j][column_name][i] for i in range(nr_of_results) if results[j]['Region'][i] == "all"]))
                                    for j in range(len(results))}
    else:
        data = {j: list([results[j][column_name][i] for i in range(nr_of_results) if results[j]['Region'][i] == "all"])
                                    for j in range(len(results))}
    # plot the figure
    print(data)
    print(budget_factor_list)
    tnr_font = {'fontname': 'DejaVu Sans'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    line_1, = plt.plot(budget_factor_list, data[0], label=name_1, color=colours[0])
    line_2, = plt.plot(budget_factor_list, data[1], label=name_2, color=colours[1])
    legend = plt.legend(handles=[line_1, line_2], fontsize="xx-large",
                        prop={'family': 'DejaVu Sans', 'size': 'xx-large'}, loc='upper right',
                        bbox_to_anchor=(0.7, 1))
    plt.xlabel(x_axis_name, fontsize="xx-large", **tnr_font)
    plt.ylabel(column_name, fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    if reverse:
        plt.xlim([None, budget_factor_list[-1]])

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "/" + output_filename, dpi=1200)
    plt.clf()

def plot_2_columns_rural_urban_from_file(input_file_1, input_file_2, column_name = "Overall access [%]", name_1 = "Naive", name_2 = "Greedy",
                             input_abs_path = None, output_abs_path = None, output_filename = "access_compare.pdf"):
    """
    make a plot of a column split by rural, urban, all in two different files by reading in the values from a file (instead of recomputing)
    :param input_file_1: the name of the first input file (number of rows determined by this first file)
    :param input_file_2: the name of the second input file
    :param column_name: name of the column we want to plot
    :param name_1: the name of the first file as should appear in the legend
    :param name_2: the name of the second file as should appear in the legend
    :param input_abs_path: the absolute path of where the input file is
    :param ioutput_abs_path: the absolute path of where the output file should be written
    :param output_filename: the name the file we write the plot to should have
    """
    if input_abs_path == None:
        input_abs_path = os.getcwd() + "/own_results"
    results = [pd.read_excel(input_abs_path + "/" + input_file_1), pd.read_excel(input_abs_path + "/" + input_file_2)]

    user_regions = ['all', 'rural', 'urban']

    # make sure the results are sorted in increasing order by budget
    nr_of_results = len(results[0])
    budget_factor_list = [int(str(budget)[:-1]) for budget in results[0]['Budget'] if type(budget) != float]
    budget_factor_list.reverse()
    # compute the percentiles of the utilization of open facilities for each instance and each region
    percentiles_dict = {region: {j: list(reversed([results[j][column_name][i] for i in range(nr_of_results) if results[j]['Region'][i] == region]))
                                    for j in range(len(results))}
                        for region in user_regions}
    # plot the figure
    tnr_font = {'fontname': 'DejaVu Sans'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    plt.plot(budget_factor_list, percentiles_dict['all'][0], ls="dashed", color='blue')
    plt.plot(budget_factor_list, percentiles_dict['rural'][0], ls="dashed", color='green')
    plt.plot(budget_factor_list, percentiles_dict['urban'][0], ls="dashed", color='red')
    line_all, = plt.plot(budget_factor_list, percentiles_dict['all'][1], label='all', color='blue')
    line_rural, = plt.plot(budget_factor_list, percentiles_dict['rural'][1], label='rural', color='green')
    line_urban, = plt.plot(budget_factor_list, percentiles_dict['urban'][1], label='urban', color='red')
    legend = plt.legend(handles=[line_all, line_rural, line_urban], fontsize="xx-large",
                        prop={'family': 'DejaVu Sans', 'size': 'xx-large'}, loc='lower right',
                        bbox_to_anchor=(0.5, 0))
    # dummy lines for legend
    line_1, = plt.plot([], [], label=name_1, ls="dashed", color='black')
    line_2, = plt.plot([], [], label=name_2, color='black')
    plt.gca().add_artist(legend)
    plt.legend(handles=[line_1, line_2], fontsize="xx-large",
                prop={'family': 'DejaVu Sans', 'size': 'xx-large'}, loc='lower right')
    plt.xlabel('Budget [%]', fontsize="xx-large", **tnr_font)
    plt.ylabel(column_name, fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.xlim([None, budget_factor_list[-1]])

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "/" + output_filename, dpi=1200)
    plt.clf()

def plot_3_lines(input_files, column_name_x = "Budget", y_label = "Utilization [%]", column_names_y = ["Utilization [%] - p10", "Utilization [%] - p50","Utilization [%] - p90"], names = ["Random"], colours = ["black"],
                             input_abs_path = None, output_abs_path = None, output_filename = "boxplot.pdf", add_100 = 3, region = "all", reverse = False):
    """
    make a boxplot of a column
    :param input_files: array of names of input files
    :param column_name_x: name of the column we want to plot in x-axis
    :param y_label: label for the y-axis
    :param column_names_y: list of columns we want to plot
    :param name: array of the names of the methods we are plotting
    :param input_abs_path: the absolute path of where the input file is
    :param ioutput_abs_path: the absolute path of where the output file should be written
    :param output_filename: the name the file we write the plot to should have
    :param add_100: adds the index of where the data is for 100% to be added, if -1 should not be added
    :param region: region for which we wish to plot this
    :param reverse: if true go from 30% to 100% open, if false go from 0% to 70% closed
    """
    if input_abs_path == None:
        input_abs_path = os.getcwd() + "/own_results"
    results = []
    for input_file in input_files:
        results.append(pd.read_excel(input_abs_path + "/" + input_file))

    # make sure the results are sorted in increasing order by budget
    budget_factor_list = [int(str(budget)[:-1]) for budget in results[0][column_name_x] if type(budget) != float]
    budget_factor_list = list(set(budget_factor_list))
    nr_of_results = len(budget_factor_list)
    if add_100 > 0:
        if 100 not in budget_factor_list:
            budget_factor_list.append(100)
            nr_of_results += 1
    budget_factor_list.sort()
    if reverse:
        data_lines = {column_name_y: {j: list(reversed([results[j][column_name_y][i] for i in range(len(results[j])) if results[j]['Region'][i] == region]))  for j in range(len(input_files))}for column_name_y in column_names_y}
    else:
        data_lines = {column_name_y: {j: list([results[j][column_name_y][i] for i in range(len(results[j])) if results[j]['Region'][i] == region])  for j in range(len(input_files))}for column_name_y in column_names_y}
    print(data_lines)
    if add_100 >= 0:
        for column_name in column_names_y:
            for j in range(0, len(input_files)):
                if j != add_100:
                    if reverse:
                        data_lines[column_name][j].append(data_lines[column_name][add_100][-1])
                    else:
                        data_lines[column_name][j].insert(0,data_lines[column_name][add_100][-1])
    print(data_lines)
    # plot the figure
    fig, axes = plt.subplots(figsize=(9, 5), sharex=True)
    tnr_font = {'fontname': 'DejaVu Sans'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    # TODO: fix colour and appearing on legend for boxplot
    #for column_name_y in column_names_y:
    #    bplot = axes.boxplot(data_boxplot[column_name_y].values(), labels=budget_factor_list, showfliers=False)
    #ticks = axes.get_xticks()
    lines_diff_method = []
    #lines.append(bplot[0])
    styles = ["dotted", "dashed", "solid"]
    for i in range(3):
        column_name = column_names_y[i]
        for j in data_lines[column_name].keys():
            line, = axes.plot(budget_factor_list, data_lines[column_name][j], label=names[j], color=colours[j], linestyle = styles[i])
            if i == 2:
                lines_diff_method.append(mpatches.Patch(color=colours[j], label=names[j]))

    
    legend = axes.legend(handles=lines_diff_method, fontsize="xx-large",
                        prop={'family': 'DejaVu Sans', 'size': 'xx-large'}, loc='upper right',  bbox_to_anchor=(1.0, 0.5))
    # dummy lines for legend
    line_10, = plt.plot([], [], label='10th percentile', ls="dotted", color='black')
    line_50, = plt.plot([], [], label='50th percentile', color='black', ls="dashed")
    line_90, = plt.plot([], [], label='90th percentile', color='black')
    plt.gca().add_artist(legend)
    plt.legend(handles=[line_10, line_50, line_90], fontsize="xx-large",
               prop={'family': 'DejaVu Sans', 'size': 'xx-large'}, loc='upper right')
    plt.xlabel(column_name_x, fontsize="xx-large", **tnr_font)
    plt.ylabel(y_label, fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.xlim()

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "/" + output_filename, dpi=1200)
    plt.clf()

def plot_boxplot(input_files, column_name_x = "Budget", x_label = "Budget", column_name_y = "Overall access [%]", y_label = "Recycling access [%]",names = ["Random"], colours = ["black"],
                             input_abs_path = None, output_abs_path = None, output_filename = "boxplot.pdf", add_100 = 3, region = "all", reverse = False, num_facs = 1394):
    """
    make a boxplot of a column
    :param input_files: array of names of input files
    :param column_name_x: name of the column we want to plot in x-axis
    :param column_name_y: name of the column we want to plot in y-axis
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param name: array of the names of the methods we are plotting
    :param input_abs_path: the absolute path of where the input file is
    :param ioutput_abs_path: the absolute path of where the output file should be written
    :param output_filename: the name the file we write the plot to should have
    :param add_100: adds the index of where the data is for 100% to be added, if -1 should not be added
    :param region: region for which we wish to plot this
    :param reverse: if true go from 30% to 100% open, if false go from 0% to 70% closed
    :param num_facs: if -1 use percentage, if positive is the number of total facilities and should be used
    """
    if input_abs_path == None:
        input_abs_path = os.getcwd() + "/own_results"
    results = []
    for input_file in input_files:
        results.append(pd.read_excel(input_abs_path + "/" + input_file))

    # make sure the results are sorted in increasing order by budget
    budget_factor_list = [int(str(budget)[:-1]) for budget in results[0][column_name_x] if type(budget) != float]
    budget_factor_list = list(set(budget_factor_list))
    nr_of_results = len(budget_factor_list)
    if add_100 > 0:
        budget_factor_list.append(100)
        nr_of_results += 1
    budget_factor_list.sort()

    if not reverse:
        budget_factor_list = list(reversed(budget_factor_list))

    print(budget_factor_list)
    region_offset = 0
    if(region == "rural"):
        region_offset = 1
    elif(region == "urban"):
        region_offset = 2
    if reverse:
        data_boxplot = {j: [results[0][column_name_y][i+region_offset] for i in range(len(results[0])) if results[0][column_name_x][i] == str(j)+"%"] for j in budget_factor_list[0:-1]}
    else:
        if num_facs >= 0:
            data_boxplot = {round(j*num_facs/100): [results[0][column_name_y][i+region_offset] for i in range(len(results[0])) if results[0][column_name_x][i] == str(j)+"%"] for j in budget_factor_list[1:len(budget_factor_list)]}
        else:
            data_boxplot = {100 - j: [results[0][column_name_y][i+region_offset] for i in range(len(results[0])) if results[0][column_name_x][i] == str(j)+"%"] for j in budget_factor_list[1:len(budget_factor_list)]}
    
    if num_facs >= 0:
        budget_factor_list = [round((100-x)*num_facs/100) for x in budget_factor_list]
    bxpstats = [{"med": np.percentile(data,50), "q1":  np.percentile(data,10),  "q3": np.percentile(data,90),
     "whislo": np.percentile(data,0) , "whishi": np.percentile(data,100), "label": budget_factor } for budget_factor, data in data_boxplot.items()]

    if reverse:
        data_lines = {j: list(reversed([results[j][column_name_y][i] for i in range(len(results[j])) if results[j]['Region'][i] == region]))  for j in range(1, len(input_files))}
    else:
        data_lines = {j: list([results[j][column_name_y][i] for i in range(len(results[j])) if results[j]['Region'][i] == region])  for j in range(1, len(input_files))}

    if add_100 > 0:
        if reverse:
            bxpstats.append({"med":data_lines[add_100][-1], "q1":  data_lines[add_100][-1],  "q3": data_lines[add_100][-1],
            "whislo":data_lines[add_100][-1] , "whishi": data_lines[add_100][-1], "label": budget_factor_list[-1] })
        else:
            bxpstats.insert(0,{"med":data_lines[add_100][0], "q1":  data_lines[add_100][0],  "q3": data_lines[add_100][0],
            "whislo":data_lines[add_100][0] , "whishi": data_lines[add_100][0], "label": num_facs if num_facs >= 0 else 0 })
        for j in range(1, len(input_files)):
            if j != add_100:
                if reverse:
                    data_lines[j].append(data_lines[add_100][0])
                else:
                     data_lines[j].insert(0,data_lines[add_100][0])

    print(bxpstats)
    print(data_lines)
    # plot the figure
    fig, axes = plt.subplots(figsize=(9, 5), sharex=True)
    tnr_font = {'fontname': 'DejaVu Sans'}
    plt.rcParams["figure.figsize"] = (10, 2 / 3 * 10)
    bplot = axes.bxp(bxpstats, showfliers=False, patch_artist=True)
    for median in bplot['medians']:
        median.set_color('black')

    for box in bplot['boxes']:
        box.set_facecolor('white')
    ticks = axes.get_xticks()
    lines = []
    #lines.append(bplot[0])
    for j in data_lines.keys():
        line, = axes.plot(ticks, data_lines[j], label=names[j], color=colours[j])
        lines.append(line)
    patch_boxplot = Patch(facecolor= colours[0], edgecolor=colours[0], label=names[0])
    lines.append(patch_boxplot)
    axes.legend(handles=lines, fontsize="xx-large",
                        prop={'family': 'DejaVu Sans', 'size': 'xx-large'}, loc='lower left')
    plt.xlabel(x_label, fontsize="xx-large", **tnr_font)
    plt.ylabel(y_label, fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.xlim()

    # save the figure
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    plt.savefig(output_abs_path + "/" + output_filename, dpi=1200)
    plt.clf()

def plotMap(
    df: pd.DataFrame,
    value_column: str,
    output_file_name: str = "map.html",
    bins: List[float] = [0, 0.2, 0.4, 0.6, 0.8, 1],
    open_facility_locations: List[Tuple[float, float]] = [],
    rural_zipcodes: List[str] = None, # todo: default to getting these from the input data, and add parameter to split or not
    urban_zipcodes: List[str] = None,
):
    """
    plot geographic results on a chloropleth map
    :param df: data to plot, with a 'zipcode' column and value column (see value_column)
    :param value_column: name of the numeric value column in df to plot
    :param output_file_name: file to output map to as HTML
    :param bins: colour divisions used on the chloropleth map legend
    :param open_facility_locations: coordinates of the open facilities, in (lat, long) format (todo: check)
    :param rural_zipcodes: list of zipcodes which are classified as rural
    :param urban_zipcodes: list of zipcodes which are classified as urban
    """
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "plz-5stellig.geojson")) as f:
        zip_codes_geojson = json.load(f)

    map = folium.Map(
        location=[49.0, 11.3],
        tiles='https://{s}.tile.thunderforest.com/mobile-atlas/{z}/{x}/{y}.png?apikey=bdd3d8f0c5c34c858dc7b57d1fc6c573',
        attr="Thunderforest",
        zoom_start=8
    )

    border = folium.GeoJson(
        # data from https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/2_hoch.geo.json
        os.path.join(os.path.dirname(__file__), "..", "data", "bayern.geojson"),
        style_function=lambda _: { "color": "#000000", "fillOpacity": "0" }
    )
    border.add_to(map)
    #map.fit_bounds(border.get_bounds())

    # todo: assert df has 'zipcode' column and it is of type string
    # todo: assert df has value_column column and it is a numeric type

    if not df.empty:
        df_zip_codes = list(df['zipcode'])
        df_zip_code_geojsons = [geo for geo in zip_codes_geojson["features"] if geo["properties"]["plz"] in df_zip_codes]
        
        if rural_zipcodes == None and urban_zipcodes == None:
            zip_codes_geojson["features"] = df_zip_code_geojsons
            folium.Choropleth(geo_data=zip_codes_geojson, data=df, columns=['zipcode', value_column],
                    key_on='properties.plz', fill_color="BuPu", fill_opacity=0.5, line_opacity=0.1,legend_name="Rural urban",bins=bins).add_to(map)
        
        # colours YlGn and BuPu for travel
        if rural_zipcodes != None:
            rural_df_zip_code_geojsons = [geo for geo in df_zip_code_geojsons if geo["properties"]["plz"] in rural_zipcodes]
            copy_zip_codes_geojson = zip_codes_geojson.copy()
            copy_zip_codes_geojson["features"] = rural_df_zip_code_geojsons
            df_rural = df[df['zipcode'].isin(rural_zipcodes)]
            folium.Choropleth(geo_data=copy_zip_codes_geojson, data=df_rural, columns=['zipcode', value_column],
                    key_on='properties.plz', fill_color="BuGn", fill_opacity=0.5, line_opacity=0.1,legend_name="Rural Utilization (%)",bins=bins).add_to(map)

        if urban_zipcodes != None:
            urban_df_zip_code_geojsons = [geo for geo in df_zip_code_geojsons if geo["properties"]["plz"] in urban_zipcodes]
            copy_zip_codes_geojson = zip_codes_geojson.copy()
            copy_zip_codes_geojson["features"] = urban_df_zip_code_geojsons
            df_urban = df[df['zipcode'].isin(urban_zipcodes)]
            folium.Choropleth(geo_data=copy_zip_codes_geojson, data=df_urban, columns=['zipcode', value_column],
                    key_on='properties.plz', fill_color="OrRd", fill_opacity=0.5, line_opacity=0.1,legend_name="Urban Utilization (%)",bins=bins).add_to(map)

    if open_facility_locations != None:
        for fac in open_facility_locations:
            folium.CircleMarker(location=fac,radius=1, color="black").add_to(map)

    folium.LayerControl().add_to(map)

    map.save(os.path.join(os.path.dirname(__file__), "..", "own_results", output_file_name))

