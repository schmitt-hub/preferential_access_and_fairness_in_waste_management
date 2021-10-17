from math import exp
import pandas as pd
from geopy import distance
import random as rn
import json


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


# functions for loading data files

def load_users_and_facs(users_and_facs_filename):
    users_and_facs_df = pd.read_excel(users_and_facs_filename)
    return users_and_facs_df


def load_travel_dict(travel_dict_filename):
    with open(travel_dict_filename, 'rb') as infile:
        travel_dict = json.load(infile)
    travel_dict = {int(i): {int(j): travel_dict[i][j] for j in travel_dict[i]} for i in travel_dict}
    return travel_dict


def load_input_data(users_and_facs_filename, travel_dict_filename):
    users_and_facs_df = load_users_and_facs(users_and_facs_filename)
    travel_dict = load_travel_dict(travel_dict_filename)
    return users_and_facs_df, travel_dict


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


def create_instance_batches(users, facs, sizes, instances_per_size=5):
    """
    create a dictionary with several selections of users and facilities for different instance sizes
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param sizes: list of tuples consisting of the fraction of considered users and facilities, respectively
    :param instances_per_size: number of instances that are created for each size
    :return: dictionary with several selections of users and facilities for different instance sizes
    """
    instance_dict = {size: {i: {} for i in range(instances_per_size)}
                     for size in sizes}
    for size in sizes:
        for i in range(instances_per_size):
            users_sample, facs_sample = select_random_instance(users, facs, size[0], size[1])
            instance_dict[size][i] = {'users': users_sample, 'facs': facs_sample}
    return instance_dict


def select_random_instance(users, facs, area_factor, fac_factor):
    """
    generate a random subset of users and facilities of the given size
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param area_factor:  fraction of users that are considered
    :param fac_factor: fraction of facilities that are considered
    :return: random subset of users and facilities
    """
    users_sample = rn.sample(users, round(len(users) * area_factor))
    facs_sample = rn.sample(facs, round(len(facs) * fac_factor))
    return users_sample, facs_sample


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

def get_distances_to_assigned(results, users_and_facs_df, region='all'):
    """
    compute the distance to the assigned facility for each user
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param region: string indicating the considered region of origin for the users: "urban", "rural" or "all"
    :return: a dictionary with the the distance to the assigned facility for each user
    """
    assignment = results['solution_details']['assignment']
    if region == 'rural':
        distance_dict = {i: geodesic_distance(users_and_facs_df, i, j) for (i, j) in assignment.items() if
                         users_and_facs_df.at[int(i), 'regional spatial type'] == 'rural'}
    elif region == 'urban':
        distance_dict = {i: geodesic_distance(users_and_facs_df, i, j) for (i, j) in assignment.items() if
                         users_and_facs_df.at[int(i), 'regional spatial type'] == 'urban'}
    else:
        distance_dict = {i: geodesic_distance(users_and_facs_df, i, j) for (i, j) in assignment.items()}
    return distance_dict


def get_utilization(results, users_and_facs_df, travel_dict, region='all'):
    """
    compute the utilization for each facility
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param region: string indicating the considered region of location for the facilities: "urban", "rural" or "all"
    :return: a dictionary with the utilization for each facility
    """
    assignment = results['solution_details']['assignment']
    open_facs = results['solution_details']['open_facs']
    cap_factor = results['model_details']['cap_factor']
    if region == 'rural':
        utilization = {j: sum(users_and_facs_df.at[i, 'population'] * travel_dict[i][j]
                              for i in assignment if assignment[i] == j) /
                          (cap_factor * users_and_facs_df.at[j, 'capacity'])
                       for j in open_facs if users_and_facs_df.at[j, 'regional spatial type'] == 'rural'}
    elif region == 'urban':
        utilization = {j: sum(users_and_facs_df.at[i, 'population'] * travel_dict[i][j]
                              for i in assignment if assignment[i] == j) /
                          (cap_factor * users_and_facs_df.at[j, 'capacity'])
                       for j in open_facs if users_and_facs_df.at[j, 'regional spatial type'] == 'urban'}
    else:
        utilization = {j: sum(users_and_facs_df.at[i, 'population'] * travel_dict[i][j]
                              for i in assignment if assignment[i] == j) /
                          (cap_factor * users_and_facs_df.at[j, 'capacity'])
                       for j in open_facs}
    return utilization


def get_overall_access(results, users_and_facs_df, travel_dict, region='all'):
    """
    compute the overall access
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param region: string indicating the considered region of origin for the users: "urban", "rural" or "all"
    :return: a scalar specifying the fraction of users that has access
    """
    assignment = results['solution_details']['assignment']
    if region == 'rural':
        overall_access = sum(
            users_and_facs_df.at[int(i), 'population'] * travel_dict[i][j] for (i, j) in assignment.items()
            if users_and_facs_df.at[int(i), 'regional spatial type'] == 'rural') / \
                         sum(users_and_facs_df.at[int(i), 'population'] for i in assignment.keys() if
                             users_and_facs_df.at[int(i), 'regional spatial type'] == 'rural')
    elif region == 'urban':
        overall_access = sum(
            users_and_facs_df.at[int(i), 'population'] * travel_dict[i][j] for (i, j) in assignment.items()
            if users_and_facs_df.at[int(i), 'regional spatial type'] == 'urban') / \
                         sum(users_and_facs_df.at[int(i), 'population'] for i in assignment.keys() if
                             users_and_facs_df.at[int(i), 'regional spatial type'] == 'urban')
    else:
        overall_access = sum(users_and_facs_df.at[int(i), 'population'] * travel_dict[i][j]
                             for (i, j) in assignment.items()) / \
                         sum(users_and_facs_df.at[i, 'population'] for i in assignment.keys())
    return overall_access


def get_weighted_utilization_figures(results, users_and_facs_df, travel_dict, region='all'):
    """
    compute the weighted mean, weighted variance and the weighted Coefficient of Variation for the utilization of
    open facilites
    :param results: dictionary of the results
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param region: string indicating the considered region: "urban", "rural" or "all"
    :return: 3 scalars, the weighted mean, weighted variance and the weighted Coefficient of Variation
    for the utilization of open facilites
    """
    utilization = get_utilization(results, users_and_facs_df, travel_dict, region)
    weighted_utilization_mean = sum(users_and_facs_df.at[j, 'capacity'] * utilization[j] for j in utilization) / \
                                sum(users_and_facs_df.at[j, 'capacity'] for j in utilization)
    weighted_utilization_var = sum(users_and_facs_df.at[j, 'capacity'] *
                                   (utilization[j] - weighted_utilization_mean) ** 2 for j in utilization) / \
                               sum(users_and_facs_df.at[j, 'capacity'] for j in utilization)
    weighted_utilization_cv = weighted_utilization_var ** 0.5 / weighted_utilization_mean
    return weighted_utilization_mean, weighted_utilization_var, weighted_utilization_cv


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
        # the dof_prime is not defined
        return dof, "undefined"
    dof_prime = nr_of_compensatory_fairness_pairs / (nr_of_facility_tuples - nr_of_warranted_fairness_pairs)
    return dof, dof_prime



