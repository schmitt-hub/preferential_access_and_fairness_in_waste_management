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
from typing import List

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


def load_users_and_facs(users_and_facs_filename="users_and_facilities.csv", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "/data"
    users_and_facs_df = pd.read_csv(abs_path + "/" + users_and_facs_filename)
    return users_and_facs_df


def load_travel_dict(travel_dict_filename="travel_dict.json.pbz2", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "/data"
    data = bz2.BZ2File(abs_path + "/" + travel_dict_filename, 'rb')
    travel_dict = cPickle.load(data)
    travel_dict = {int(i): {int(j): travel_dict[i][j] for j in travel_dict[i]} for i in travel_dict}
    return travel_dict


def load_input_data(
    users_and_facilities_filename="users_and_facilities.csv",
    travel_dict_filename="travel_dict.json.pbz2",
    abs_path=None
):
    users_and_facs_df = load_users_and_facs(users_and_facilities_filename, abs_path)
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
    """"
    return rural and urban for all in list, otherwise just the specific region
    :param user_region: region used for the user
    :param facility_region: region used for the facility
    """
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
    access_dict = {j: sum(users_and_facs_df.at[i, 'population'] * travel_dict[i][j] for i in assignment.keys()
                     if assignment[i] == j and users_and_facs_df.at[i, 'regional spatial type'] in user_region_list)
              for j in open_facs if users_and_facs_df.at[j, 'regional spatial type'] in facility_region_list}
    return access_dict


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
    access_dict = get_access(results, users_and_facs_df, travel_dict, user_region, facility_region)
    overall_access = sum(access_dict.values()) / sum(users_and_facs_df.at[int(i), 'population']
                                                for i in assignment.keys() if
                                                users_and_facs_df.at[i, 'regional spatial type'] in user_region_list)
    return overall_access

def get_access_split(results, users_and_facs_df, travel_dict, user_region='all', facility_region='all', perspective = 'facs'):
    """
    compute the proportion of rural or urban that make up overall access
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param user_region: string indicating the considered region of origin for the users: "urban", "rural" or "all"
    :param facility_region: string indicating the considered region of location for the facilities:
        "urban", "rural" or "all"
    :param perspective: do we consider this from persepctive of facilities or users
    :return: a scalar specifying the fraction of users that has access
    """
    access = get_access(results, users_and_facs_df, travel_dict, user_region, facility_region)
    access_denom = access
    if perspective == 'facs':
        access_denom = get_access(results, users_and_facs_df, travel_dict, 'all', facility_region)
    elif perspective == 'users':
        access_denom = get_access(results, users_and_facs_df, travel_dict, user_region, 'all')
    else:
        return -1
    denom = sum(access_denom.values())
    if denom == 0:
        return -1
    overall = sum(access.values())/denom
    return overall


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

def get_proportion_open(results, users_and_facs_df):
    """
    compute the proportion of faciliities open in all, rural and urban regions
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :return the proportion of open facilities in all, rural, urban regions in that order
    """
    num_open_rural = 0
    num_open_urban = 0
    num_rural = 0
    num_urban = 0
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    for j in results['solution_details']['open_facs']:
        if users_and_facs_df.at[j,'regional spatial type'] == 'urban':
            num_open_urban += 1
        elif users_and_facs_df.at[j,'regional spatial type'] == 'rural':
            num_open_rural += 1
    for j in facs:
        if users_and_facs_df.at[j,'regional spatial type'] == 'urban':
            num_urban += 1
        elif users_and_facs_df.at[j,'regional spatial type'] == 'rural':
            num_rural += 1
    prop_all = (num_open_rural + num_open_urban)/(num_rural+num_urban)
    prop_rural = num_open_rural/num_rural
    prop_urban = num_open_urban/num_urban
    return prop_all, prop_rural, prop_urban

def get_proportion_open_by_capacity(results, users_and_facs_df):
    """
    compute the proportion of capacities that are open in all, rural and urban regions
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :return the proportion capacitities that remain open in all, rural, urban regions in that order
    """
    num_open_rural = 0
    num_open_urban = 0
    num_rural = 0
    num_urban = 0
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    for j in results['solution_details']['open_facs']:
        if users_and_facs_df.at[j,'regional spatial type'] == 'urban':
            num_open_urban += users_and_facs_df.at[j,'capacity']
        elif users_and_facs_df.at[j,'regional spatial type'] == 'rural':
            num_open_rural += users_and_facs_df.at[j,'capacity']
    for j in facs:
        if users_and_facs_df.at[j,'regional spatial type'] == 'urban':
            num_urban += users_and_facs_df.at[j,'capacity']
        elif users_and_facs_df.at[j,'regional spatial type'] == 'rural':
            num_rural += users_and_facs_df.at[j,'capacity']
    prop_all = (num_open_rural + num_open_urban)/(num_rural+num_urban)
    prop_rural = num_open_rural/num_rural
    prop_urban = num_open_urban/num_urban
    return prop_all, prop_rural, prop_urban

def split_rural_urban(users_and_facs_df: pd.DataFrame, users: List[int] = None, facs: List[int] = None):
    """
    split the users and facilities into two lists by rural and urban
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param users: indexes in users_and_facs_df which are users
    :param facilities: indexes in users_and_facs_df which are facilities
    :return users in rural, facilities in rural, users in urban, facilitiies in urban areas
    """
    if users == None:
        users = [int(i) for i in users_and_facs_df.index]
    if facs == None: 
        facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]

    facs_rural = [i for i in facs if users_and_facs_df.at[i,'regional spatial type'] == 'rural']
    facs_urban = [i for i in facs if users_and_facs_df.at[i,'regional spatial type'] == 'urban']
    users_rural = [i for i in users if users_and_facs_df.at[i,'regional spatial type'] == 'rural']
    users_urban = [i for i in users if users_and_facs_df.at[i,'regional spatial type'] == 'urban']
    return users_rural, facs_rural, users_urban, facs_urban

def combine_results(results_1, results_2):
    """
    combine the results from two runs (on seperate user, facility data)
    :param results_1 the first dictionary of results
    :param results_2 the second dictionary of results
    :return the "union" of the results
    """
    same_check = list(results_1['model_details'].keys())
    same_check.remove('users')
    same_check.remove('facs')
    for x in same_check:
        if results_1['model_details'][x] != results_2['model_details'][x]:
            print("Model data is not the same for " + x)

    results = {"solution_details":
                   {"assignment": results_1['solution_details']['assignment'] | results_2['solution_details']['assignment'],
                    "open_facs": results_1['solution_details']['open_facs'] + results_2['solution_details']['open_facs'],
                    "objective_value": results_1['solution_details']['objective_value'] + results_2['solution_details']['objective_value'],
                    "lower_bound": results_1['solution_details']['lower_bound'] + results_2['solution_details']['lower_bound'], "solving_time": results_1['solution_details']['solving_time'] + results_2['solution_details']['solving_time']},
               "model_details":
                   {"users": results_1['model_details']['users'] + results_2['model_details']['users'],
                    "facs": results_1['model_details']['facs'] + results_2['model_details']['facs'], 
                    "cap_factor": results_1['model_details']['cap_factor'], "budget_factor": results_1['model_details']['budget_factor'],
                    "turnover_factor": results_1['model_details']['turnover_factor'], "tolerance": results_1['model_details']['tolerance'],
                     "time_limit": results_1['model_details']['time_limit'], "iteration_limit": results_1['model_details']['iteration_limit']}
               }
    return results

def average_max_P(users_and_facs_df, travel_dict, users, facs):
    """
    computes the average maximum P for urban and rural area
    :param users_and_facs_df: dataframe of users and facilities and associated data
    :param travel_dict: dictionary of travel probabilities from each ZIP code to each recycling center
    :param users: list of users to index into users_and_facs_df
    :param facs: list of faciliities to index into users_and_facs_df
    """
    n_urban = 0
    n_rural = 0
    sum_urban = 0
    sum_rural = 0
    for i in users:
        max_pref = 0
        for j in facs:
            if max_pref < travel_dict[i][j]:
                max_pref = travel_dict[i][j]
        if users_and_facs_df.at[i,'regional spatial type'] == 'urban':
            n_urban += 1
            sum_urban += max_pref
        if users_and_facs_df.at[i,'regional spatial type'] == 'rural':
            n_rural += 1
            sum_rural += max_pref
    average_urban = sum_urban / n_urban
    average_rural = sum_rural / n_rural
    print("average urban is " + str(average_urban))
    print("average rural is " + str(average_rural))

def get_dof_of_subset(results, users_and_facs_df, travel_dict, facs_subset = None):
    """
    compute the DoF for the subset of facilities
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param facs_subset: subset of facilities we are computing the DOF of
    :return: the DOF
    """
    assignment = results['solution_details']['assignment']
    open_facs = results['solution_details']['open_facs']
    if not facs_subset:
        facs_subset = open_facs
    utilization = get_utilization(results, users_and_facs_df, travel_dict)
    j_u_tuples = [(j, utilization[j]) for j in open_facs]
    j_u_tuples_subset =  [(j, utilization[j]) for j in open_facs if j in facs_subset]
    nr_of_warranted_fairness_pairs = 0
    nr_of_compensatory_fairness_pairs = 0
    for (j1, u1) in j_u_tuples_subset:
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

    nr_of_facility_tuples = len(j_u_tuples_subset) * (len(open_facs) - 1)
    if nr_of_facility_tuples == 0:
        return -1
    dof = (nr_of_warranted_fairness_pairs + nr_of_compensatory_fairness_pairs) / nr_of_facility_tuples
    return dof

def read_results_list(filename):
    """
    read a results list and fix keys being string instead of int
    :param filename: filename including path
    :return results list with fixed keys being int
    """
    f = open(filename)
    results_list = json.load(f)
    for i in range(len(results_list)):
        d = results_list[i]
        assignment = d['solution_details']['assignment']
        new_assignment = {}
        for key in assignment.keys():
            new_assignment[int(key)] = assignment[key]
        results_list[i]['solution_details']['assignment'] = new_assignment
    f.close()
    return results_list

def capacity_constraints_slack(users_and_facs_df, travel_dict, facs, assignment, cap_factor=1.5):
    """
    check how much slack there is in the capacity
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param facs: list of the facilities used in the instance
    :param cap_factor: factor by which all capacities are scaled
    :param assignment: dictionary of key user, value facility
    :return: return a dictionary of slack for each facility, array of facilities who do not have satisfied, dictionary of users assigned by facility
    """
    slacks = {}
    not_satisfed = [] 
    assigned_users = {}
    for j in facs:
        assigned_users[j] = [k for k,v in assignment.items() if v == j]
        total_use = sum([ users_and_facs_df.at[i, 'population'] * travel_dict[i][j] for i in assigned_users[j]])
        slacks[j] =  users_and_facs_df.at[j, 'capacity'] * cap_factor - total_use
        if slacks[j] < 0:
           not_satisfed.append(j)
    return slacks, not_satisfed, assigned_users