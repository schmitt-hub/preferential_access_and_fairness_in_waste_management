"""
module for creating the exact figures and tables included in the paper
"""

from plotting import *
from results import *


# create the figures included in the paper

def create_figure3a(output_filename='overall_access.pdf'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_overall_access_plot(users_and_facs_df, travel_dict, users, facs, output_filename, budget_factor_list)


def create_figure3b(output_filename='distance_percentiles.pdf'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    percentiles = [10, 50, 90]
    save_distance_percentiles_plot(users_and_facs_df, travel_dict, users, facs, output_filename, percentiles,
                                   budget_factor_list)


def create_figure4a(output_filename='utilization_percentiles.pdf'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    percentiles = [10, 50, 90]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_utilization_percentiles_plot(users_and_facs_df, travel_dict, users, facs, output_filename, percentiles,
                                      budget_factor_list)


def create_figure4b(output_filename='utilization_distribution.pdf'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_utilization_distribution_plot(users_and_facs_df, travel_dict, users, facs, output_filename, budget_factor_list)


def create_figure5a(output_filename='strict_vs_loose.pdf'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_strict_vs_loose_plot(users_and_facs_df, travel_dict, users, facs, output_filename, budget_factor_list)


def create_figure5b(output_filename='cutoff_vs_nocutoff.pdf'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    users = users[:round(len(users) * 0.3)]
    facs = facs[:round(len(facs) * 0.3)]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_cutoff_vs_nocutoff_plot(users_and_facs_df, travel_dict, users, facs, output_filename, budget_factor_list,
                                 main_preqlinearize=1)


def create_figureS1a(output_filename='utilization_distribution_rural.pdf'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_utilization_distribution_plot(users_and_facs_df, travel_dict, users, facs, output_filename,
                                       budget_factor_list, facility_region='rural')


def create_figureS1b(output_filename='utilization_distribution_urban.pdf'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_utilization_distribution_plot(users_and_facs_df, travel_dict, users, facs, output_filename,
                                       budget_factor_list, facility_region='urban')


def create_figureS2(output_filename='cap_vs_access.pdf'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    save_cap_vs_access_plot(users_and_facs_df, travel_dict, users, facs, output_filename, budget_factor=0.3)


# create the tables included in the paper

def create_table1(output_filename='overall_results.xlsx'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    percentiles = [10, 50, 90]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_overall_results(users_and_facs_df, travel_dict, users, facs, output_filename, percentiles, budget_factor_list)


def create_table2(output_filename='strict_vs_loose_results.xlsx'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    save_strict_vs_loose_results(users_and_facs_df, travel_dict, users, facs, output_filename, budget_factor=0.3)


def create_table3(output_filename='fairness_results.xlsx'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_fairness_results(users_and_facs_df, travel_dict, users, facs, output_filename, budget_factor_list)


def create_table4(output_filename='pof_results.xlsx'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    budget_factor_list = [0.1 * b for b in range(3, 11)]
    save_pof_results(users_and_facs_df, travel_dict, users, facs, output_filename, budget_factor_list)


def create_table5(output_filename='greedy_results_nocutoff.xlsx'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    instance_dict = {(0.1 * k, 0.1 * k):
                         {0: {'users': users[:round(0.1 * k * len(users))], 'facs': facs[:round(0.1 * k * len(facs))]}}
                     for k in range(1, 11)}
    save_greedy_results(users_and_facs_df, travel_dict, instance_dict, output_filename, budget_factor=0.7, cutoff=0.0)


def create_tableS1(output_filename='cutoff_results.xlsx'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    instance_dict = {(0.1 * i, 0.1 * j):
                         {0: {'users': users[:round(0.1 * i * len(users))], 'facs': facs[:round(0.1 * j * len(facs))]}}
                     for i in range(1, 6) for j in range(i - 1, i + 2) if 0 < j < 6}
    save_cutoff_results(users_and_facs_df, travel_dict, instance_dict, output_filename, budget_factor=0.7,
                        main_time_limit=10800, main_preqlinearize_cutoff=1)


def create_tableS2(output_filename='greedy_results_cutoff.xlsx'):
    users_and_facs_df, travel_dict = load_input_data("users_and_facilities.xlsx", "travel_dict.json.pbz2")
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    instance_dict = {(0.1 * k, 0.1 * k):
                         {0: {'users': users[:round(0.1 * k * len(users))], 'facs': facs[:round(0.1 * k * len(facs))]}}
                     for k in range(1, 11)}
    save_greedy_results(users_and_facs_df, travel_dict, instance_dict, output_filename, budget_factor=0.7,
                        main_time_limit=10800, main_preqlinearize=1, greedy_time_limit=10800)
