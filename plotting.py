from utils import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
import scipy.stats as stats


def save_cap_vs_access_plot(results, users_and_facs_df, travel_dict, facility_region='all',
                            output_filename='cap_vs_access.pdf'):
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
    plt.savefig(output_filename, dpi=1200)


def save_overall_access_plot(results_list, users_and_facs_df, travel_dict, output_filename='overall_access.pdf'):
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
    plt.xlim([None, 100])
    plt.savefig(output_filename, dpi=1200)


def save_utilization_percentiles_plot(results_list, users_and_facs_df, travel_dict, percentiles=None,
                                      output_filename='utilization_percentiles.pdf'):
    # init
    if percentiles is None or len(percentiles) != 3:
        percentiles = [10, 50, 90]
    regions = ['all', 'rural', 'urban']

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
    percentiles_dict = {region: {p: [np.percentile(utilization_dict[region][i], p) for i in range(nr_of_results)]
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
    plt.legend(handles=[line_all, line_rural, line_urban], fontsize="xx-large",
               prop={'family': 'Times New Roman', 'size': 'xx-large'})
    plt.xlabel('Budget [%]', fontsize="xx-large", **tnr_font)
    plt.ylabel('Utilization [%]', fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.xlim([None, 100])
    plt.savefig(output_filename, dpi=1200)


def save_utilization_distribution_plot(results_list, users_and_facs_df, travel_dict, facility_region='all',
                                       output_filename='utilization_distribution.pdf'):
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
        plt.savefig(output_filename, dpi=1200)


def save_distance_percentiles_plot(results_list, users_and_facs_df, percentiles=None, facility_region='all',
                                   output_filename='distance_percentiles.pdf'):
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
    percentiles_dict = {region: {p: [np.percentile(distance_dict[region][i], p) for i in range(nr_of_results)]
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
    plt.legend(handles=[line_all, line_rural, line_urban], fontsize="xx-large",
               prop={'family': 'Times New Roman', 'size': 'xx-large'})
    plt.xlabel('Budget [%]', fontsize="xx-large", **tnr_font)
    plt.ylabel('Distance to assigned facility [km]', fontsize="xx-large", **tnr_font)
    plt.ylim()
    plt.tick_params(labelsize="x-large")
    plt.xticks(**tnr_font)
    plt.yticks(**tnr_font)
    plt.xlim([None, 100])
    plt.savefig(output_filename, dpi=1200)


def save_strict_vs_loose_plot(results_list, output_filename='strict_vs_loose.pdf'):
    # make sure the results are sorted in increasing order by budget
    results_list = sorted(results_list, key=lambda d: d['model_details']['budget_factor'])

    # get relevant data
    strict_results_list = [results for results in results_list if results['model_details']['strict_assign_to_one']]
    strict_obj_list = [results['solution_details']['objective_value'] for results in strict_results_list]
    strict_solving_time_list = [results['solution_details']['solving_time'] for results in strict_results_list]
    strict_budget_factor_list = [round(results['model_details']['budget_factor'] * 100)
                                 for results in strict_results_list]
    loose_results_list = [results for results in results_list if not results['model_details']['strict_assign_to_one']]
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
    plt.xlim([None, 100])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(output_filename, dpi=1200)


def save_cutoff_vs_nocutoff_plot(results_list, output_filename='cutoff_vs_nocutoff.pdf'):
    # make sure the results are sorted in increasing order by budget
    results_list = sorted(results_list, key=lambda d: d['model_details']['budget_factor'])

    # get relevant data
    cutoff_results_list = [results for results in results_list if results['model_details']['cutoff'] > 0]
    if not len(set(results['model_details']['cutoff'] for results in cutoff_results_list)) == 1:
        print('Not all instances use the same cutoff')
        return None
    else:
        cutoff = cutoff_results_list[0]['model_details']['cutoff'] * 100
    cutoff_obj_list = [results['solution_details']['objective_value'] for results in cutoff_results_list]
    cutoff_solving_time_list = [results['solution_details']['solving_time'] for results in cutoff_results_list]
    cutoff_budget_factor_list = [round(results['model_details']['budget_factor'] * 100)
                                 for results in cutoff_results_list]
    nocutoff_results_list = [results for results in results_list if not results['model_details']['cutoff'] > 0]
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
    plt.legend(fontsize="xx-large", handles=[line_cutoff, line_nocutoff],
               prop={'family': 'Times New Roman', 'size': 'xx-large'})
    plt.xlim([None, 100])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(output_filename, dpi=1200)
