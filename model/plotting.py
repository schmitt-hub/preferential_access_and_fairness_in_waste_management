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
from sklearn.linear_model import LinearRegression
import scipy.stats as stats


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




