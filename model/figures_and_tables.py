"""
module for creating the exact figures and tables included in the paper
"""

from plotting import *
from results import *


# create the figures included in the paper

def create_figure3a(output_filename='overall_access.pdf', output_abs_path=None, input_data_abs_path=None):
    """
    creates figure 3a of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_overall_access_plot(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                             budget_factor_list)


def create_figure3b(output_filename='distance_percentiles.pdf', output_abs_path=None, input_data_abs_path=None):
    """
    creates figure 3b of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    percentiles = [10, 50, 90]
    save_distance_percentiles_plot(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                                   percentiles, budget_factor_list)


def create_figure4a(output_filename='utilization_percentiles.pdf', output_abs_path=None, input_data_abs_path=None):
    """
    creates figure 4a of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    percentiles = [10, 50, 90]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_utilization_percentiles_plot(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                                      percentiles, budget_factor_list)


def create_figure4b(output_filename='utilization_distribution.pdf', output_abs_path=None, input_data_abs_path=None):
    """
    creates figure 4b of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_utilization_distribution_plot(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                                       budget_factor_list)


def create_figure5a(output_filename='strict_vs_loose.pdf', output_abs_path=None, input_data_abs_path=None):
    """
    creates figure 5a of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_strict_vs_loose_plot(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                              budget_factor_list)


def create_figure5b(output_filename='cutoff_vs_nocutoff.pdf', output_abs_path=None, input_data_abs_path=None):
    """
    creates figure 5b of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    users = users[:round(len(users) * 0.3)]
    facs = facs[:round(len(facs) * 0.3)]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_cutoff_vs_nocutoff_plot(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                                 budget_factor_list, main_preqlinearize=1)


def create_figure6(output_filename='cap_vs_access.pdf', output_abs_path=None, input_data_abs_path=None):
    """
    creates figure 6 of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    save_cap_vs_access_plot(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                            budget_factor=0.3)


def create_figure7(output_filename='utilization_distribution_rural.pdf', output_abs_path=None,
                   input_data_abs_path=None):
    """
    creates figure S1a of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_utilization_distribution_plot(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                                       budget_factor_list, facility_region='rural')


def create_figure8(output_filename='utilization_distribution_urban.pdf', output_abs_path=None,
                   input_data_abs_path=None):
    """
    creates figure S1b of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_utilization_distribution_plot(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                                       budget_factor_list, facility_region='urban')


# create the tables included in the paper

def create_table1(output_filename='overall_results.xlsx', output_abs_path=None, input_data_abs_path=None):
    """
    creates table 1 of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    percentiles = [10, 50, 90]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_overall_results(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                         percentiles, budget_factor_list)


def create_table2(output_filename='strict_vs_loose_results.xlsx', output_abs_path=None, input_data_abs_path=None):
    """
    creates table 2 of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    save_strict_vs_loose_results(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                                 budget_factor=0.3)


def create_table3(output_filename='fairness_results.xlsx', output_abs_path=None, input_data_abs_path=None):
    """
    creates table 3 of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_fairness_results(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path,
                          budget_factor_list)


def create_table4(output_filename='pof_results.xlsx', output_abs_path=None, input_data_abs_path=None):
    """
    creates table 4 of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_pof_results(users_and_facs_df, travel_dict, users, facs, output_filename, output_abs_path, budget_factor_list)


def create_table5(output_filename='greedy_results_nocutoff.xlsx', output_abs_path=None, input_data_abs_path=None):
    """
    creates table 5 of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    instance_dict = {(0.1 * k, 0.1 * k):
                         {0: {'users': users[:round(0.1 * k * len(users))], 'facs': facs[:round(0.1 * k * len(facs))]}}
                     for k in range(1, 11)}
    save_greedy_results(users_and_facs_df, travel_dict, instance_dict, output_filename, output_abs_path,
                        budget_factor=0.7, cutoff=0.0)


def create_table6(output_filename='cutoff_results.xlsx', output_abs_path=None, input_data_abs_path=None):
    """
    creates table S1 of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    instance_dict = {(0.1 * i, 0.1 * j):
                         {0: {'users': users[:round(0.1 * i * len(users))], 'facs': facs[:round(0.1 * j * len(facs))]}}
                     for i in range(1, 6) for j in range(i - 1, i + 2) if 0 < j < 6}
    save_cutoff_results(users_and_facs_df, travel_dict, instance_dict, output_filename, output_abs_path,
                        budget_factor=0.7, main_time_limit=10800, main_preqlinearize_cutoff=1)


def create_table7(output_filename='greedy_results_cutoff.xlsx', output_abs_path=None, input_data_abs_path=None):
    """
    creates table S2 of the paper
    :param output_filename: string containing the name of the output file
    :param output_abs_path: string containing the absolute path of the desired output directory;
        if None, the output file will be created in a folder called "own_results"
    :param input_data_abs_path: string containing the absolute path of the directory containing the input files;
        if None, the input files will be searched for in the same directory as this script
    """
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2",
                                                     input_data_abs_path)
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    instance_dict = {(0.1 * k, 0.1 * k):
                         {0: {'users': users[:round(0.1 * k * len(users))], 'facs': facs[:round(0.1 * k * len(facs))]}}
                     for k in range(1, 11)}
    save_greedy_results(users_and_facs_df, travel_dict, instance_dict, output_filename, output_abs_path,
                        budget_factor=0.7, main_time_limit=10800, main_preqlinearize=1, greedy_time_limit=10800)


create_table4()