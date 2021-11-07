"""
Utility and help functions
"""

import os
from math import exp
import numpy as np
import pandas as pd
from geopy import distance
import json
import bz2
import _pickle as cPickle


# functions for computing travel probabilities

def f_urban(d):
    if d < 5:
        return exp(-0.2550891696011455 * d ** 0.8674531576586394)
    else:
        return 4.639450774188538 * exp(-1.4989521421856289 * d ** 0.3288777336829004)


def f_rural(d):
    if d < 5:
        return exp(-0.24990116894290326 * d ** 0.8201058149904008)
    else:
        return 1.6114912595353221 * exp(-0.6887217475464711 * d ** 0.43652329253292316)


def create_travel_dict(users_and_facs_df, users, facs):
    """
    create the travel dictionary that specifies the probabilities for each travel combination
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :return: travel dictionary that specifies the probabilities for each travel combination
    """
    travel_dict = {i: {} for i in users}
    for i in users:
        print('user', i)
        regiotype = users_and_facs_df.at[i, 'regional spatial type']
        lat_1 = users_and_facs_df.at[i, 'centroid_lat']
        lon_1 = users_and_facs_df.at[i, 'centroid_lon']
        for j in facs:
            lat_2 = users_and_facs_df.at[j, 'rc_centroid_lat']
            lon_2 = users_and_facs_df.at[j, 'rc_centroid_lon']
            dist = distance.distance((lat_1, lon_1), (lat_2, lon_2)).km
            if regiotype == "urban":
                travel_dict[i][j] = f_urban(dist)
            else:
                travel_dict[i][j] = f_rural(dist)
    with open('travel_dict.json', 'w') as outfile:
        json.dump(travel_dict, outfile)


# functions for loading and saving data files

def save_travel_dict(travel_dict, travel_dict_filename, abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    with bz2.BZ2File(abs_path + "\\" + travel_dict_filename, "w") as f:
        cPickle.dump(travel_dict, f)


def load_users_and_facs(users_and_facs_filename="users_and_facilities.xlsx", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "\\data"
    users_and_facs_df = pd.read_excel(abs_path + "\\" + users_and_facs_filename)
    return users_and_facs_df


def load_travel_dict(travel_dict_filename="travel_dict.json.pbz2", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "\\data"
    data = bz2.BZ2File(abs_path + "\\" + travel_dict_filename, 'rb')
    travel_dict = cPickle.load(data)
    travel_dict = {int(i): {int(j): travel_dict[i][j] for j in travel_dict[i]} for i in travel_dict}
    return travel_dict


def load_input_data(users_and_facs_filename="users_and_facilities.xlsx", travel_dict_filename="travel_dict.json.pbz2",
                    abs_path=None):
    users_and_facs_df = load_users_and_facs(users_and_facs_filename, abs_path)
    travel_dict = load_travel_dict(travel_dict_filename, abs_path)
    return users_and_facs_df, travel_dict


def load_results(results_filename, abs_path):
    with open(abs_path+"\\"+results_filename, 'r') as infile:
        results_list = json.load(infile)
    for results in results_list:
        results['solution_details']['assignment'] = {int(i): j for (i, j) in
                                                     results['solution_details']['assignment'].items()}
    return results_list


def save_results(results_list, results_filename, abs_path):
    if not abs_path:
        abs_path = os.getcwd() + "\\own_results"
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    with open(abs_path + "\\" + results_filename, 'w') as outfile:
        json.dump(results_list, outfile)


def safe_percentile(a, p):
    """
    compute the p-th percentile of the array a. If the array is empty, return None
    :param a: an array of values
    :param p: percentile; must be between 0 and 100
    :return: the p-th percentile of the array or None if the array is empty
    """
    if not a:
        return None
    else:
        return np.percentile(a, p)


def geodesic_distance(users_and_facs_df, i, j):
    """
    compute the geodesic distance from user i to facility j
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param i: index of the user
    :param j: index of the facility
    :return: distance in kilometers from user i to facility j
    """
    return distance.distance(
        (users_and_facs_df.at[i, 'centroid_lat'], users_and_facs_df.at[i, 'centroid_lon']),
        (users_and_facs_df.at[j, 'rc_centroid_lat'],
         users_and_facs_df.at[j, 'rc_centroid_lon'])).km


def get_lower_bound(users_and_facs_df, travel_dict, users, facs, budget_factor, cap_factor=1.5):
    """
    compute a lower bound on the optimal objective value
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: ratio of facilities that are allowed to be opened
    :param cap_factor: factor by which all capacities are scaled
    :return: a lower bound on the optimal objective value
    """
    budget = round(budget_factor * len(facs))

    # filter the data frame by the used facilities and sort it by their capacities
    sorted_facs_df = users_and_facs_df.iloc[facs].sort_values(by=['capacity'], ascending=False)
    caps = [c * cap_factor for c in sorted_facs_df['capacity']]
    largest_caps = caps[:budget]

    max_exp_travelers = [users_and_facs_df.at[i, 'population'] * max([travel_dict[i][j] for j in facs]) for i in users]
    lb = min(sum(caps) + sum(max_exp_travelers) ** 2 / sum(largest_caps) - 2 * sum(max_exp_travelers),
             sum(caps) - sum(max_exp_travelers))
    return lb


# functions for computing key figures from results

def get_region_list(user_region='all', facility_region='all'):
    if user_region == 'all':
        user_region_list = ['rural', 'urban']
    else:
        user_region_list = [user_region]
    if facility_region == 'all':
        facility_region_list = ['rural', 'urban']
    else:
        facility_region_list = [facility_region]
    return user_region_list, facility_region_list


def get_distances_to_assigned(results, users_and_facs_df, user_region='all', facility_region='all'):
    """
    compute the distance from users to the respective assigned facility
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param user_region: string indicating the considered region of origin for the users: "urban", "rural" or "all"
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
    :return: a dictionary with the the distance from users to the respective assigned facility
    """
    assignment = results['solution_details']['assignment']
    user_region_list, facility_region_list = get_region_list(user_region, facility_region)
    distance_dict = {i: geodesic_distance(users_and_facs_df, i, j) for (i, j) in assignment.items() if
                     users_and_facs_df.at[i, 'regional spatial type'] in user_region_list and
                     users_and_facs_df.at[j, 'regional spatial type'] in facility_region_list}
    return distance_dict


def get_access(results, users_and_facs_df, travel_dict, user_region='all', facility_region='all'):
    """
    compute the abs access of facilities to users
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param user_region: string indicating the considered region of origin for the users: "urban", "rural" or "all"
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
    :return: a dictionary with the abs access of facilities to users
    """
    assignment = results['solution_details']['assignment']
    open_facs = results['solution_details']['open_facs']
    user_region_list, facility_region_list = get_region_list(user_region, facility_region)
    access = {j: sum(users_and_facs_df.at[i, 'population'] * travel_dict[i][j] for i in assignment.keys()
                     if assignment[i] == j and users_and_facs_df.at[i, 'regional spatial type'] in user_region_list)
              for j in open_facs if users_and_facs_df.at[j, 'regional spatial type'] in facility_region_list}
    return access


def get_overall_access(results, users_and_facs_df, travel_dict, user_region='all', facility_region='all'):
    """
    compute the overall access
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param user_region: string indicating the considered region of origin for the users: "urban", "rural" or "all"
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
    :return: a scalar specifying the fraction of users that has access
    """
    assignment = results['solution_details']['assignment']
    user_region_list, _ = get_region_list(user_region, facility_region)
    access = get_access(results, users_and_facs_df, travel_dict, user_region, facility_region)
    overall_access = sum(access.values()) / sum(users_and_facs_df.at[int(i), 'population']
                                                for i in assignment.keys() if
                                                users_and_facs_df.at[i, 'regional spatial type'] in user_region_list)
    return overall_access


def get_utilization(results, users_and_facs_df, travel_dict, user_region='all', facility_region='all'):
    """
    compute the utilization of open facilities
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param user_region: string indicating the considered region of origin for the users: "urban", "rural" or "all"
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
    :return: a dictionary with the utilization of open facilities
    """
    open_facs = results['solution_details']['open_facs']
    cap_factor = results['model_details']['cap_factor']
    _, facility_region_list = get_region_list(user_region, facility_region)
    access = get_access(results, users_and_facs_df, travel_dict, user_region, facility_region)
    utilization = {j: access[j] / (users_and_facs_df.at[j, 'capacity'] * cap_factor)
                   for j in open_facs if users_and_facs_df.at[j, 'regional spatial type'] in facility_region_list}
    return utilization


def get_weighted_utilization_figures(results, users_and_facs_df, travel_dict, facility_region='all'):
    """
    compute the weighted mean, weighted variance and the weighted Coefficient of Variation for the utilization of
    open facilites
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
    :return: 3 scalars, the weighted mean, weighted variance and the weighted Coefficient of Variation
    for the utilization of open facilites
    """
    utilization = get_utilization(results, users_and_facs_df, travel_dict, facility_region=facility_region)
    weighted_utilization_mean = sum(users_and_facs_df.at[j, 'capacity'] * utilization[j] for j in utilization) / \
                                sum(users_and_facs_df.at[j, 'capacity'] for j in utilization)
    weighted_utilization_std = (sum(users_and_facs_df.at[j, 'capacity'] *
                                    (utilization[j] - weighted_utilization_mean) ** 2 for j in utilization) /
                                sum(users_and_facs_df.at[j, 'capacity'] for j in utilization)) ** 0.5
    weighted_utilization_cv = weighted_utilization_std / weighted_utilization_mean
    return weighted_utilization_mean, weighted_utilization_std, weighted_utilization_cv


def get_dof(results, users_and_facs_df, travel_dict):
    """
    compute the DoF and the DoF'
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :return: 2 scalars, the DoF and the DoF'
    """
    assignment = results['solution_details']['assignment']
    open_facs = results['solution_details']['open_facs']
    utilization = get_utilization(results, users_and_facs_df, travel_dict)
    j_u_tuples = [(j, utilization[j]) for j in open_facs]
    nr_of_warranted_fairness_pairs = 0
    nr_of_compensatory_fairness_pairs = 0
    for (j1, u1) in j_u_tuples:
        assigned_users = [int(i) for (i, j) in assignment.items() if j == j1]
        for (j2, u2) in j_u_tuples:
            if j2 == j1:
                continue
            is_most_preferred = True
            for i in assigned_users:
                if travel_dict[i][j1] < travel_dict[i][j2]:
                    is_most_preferred = False
                    break
            if is_most_preferred:
                nr_of_warranted_fairness_pairs += 1
            else:
                if u1 < u2:
                    nr_of_compensatory_fairness_pairs += 1

    nr_of_facility_tuples = len(open_facs) * (len(open_facs) - 1)
    dof = (nr_of_warranted_fairness_pairs + nr_of_compensatory_fairness_pairs) / nr_of_facility_tuples
    if nr_of_facility_tuples == nr_of_warranted_fairness_pairs:
        # in that case the dof_prime is not defined
        return dof, None
    dof_prime = nr_of_compensatory_fairness_pairs / (nr_of_facility_tuples - nr_of_warranted_fairness_pairs)
    return dof, dof_prime
