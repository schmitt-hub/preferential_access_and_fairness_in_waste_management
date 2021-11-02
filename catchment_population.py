"""
module for efficient computation of a facility's catchment population
"""
import pandas as pd
import numpy as np
from geopy.distance import distance


def get_fac_catchment_population_old(cells, facs):
    """
    calculates the catchment population of each facility, i.e. the total population for which this facility is closest.
     The number of distance computations is significantly reduced by omitting distance computations from grid cells to
     those facilities that are provably impossible to be the closest (because their latitudes and longitudes differ too
     much from those of the grid cell.)
    :param cells: list of dictionaries containing the latitude and longitude of each cell's centroid as well as
        its population
    :param facs: list of dictionaries containing the zip code, latitude and longitude of each facility
    :return: updated list of dictionaries containing the zip code, latitude, longitude and catchment population of each
        facility
    """
    nr_of_cells = len(cells)
    for fac in facs:
        fac['catchment_population'] = 0
    for index, cell in enumerate(cells):
        if index % 1000 == 0:
            print('cell', index, 'of', nr_of_cells)
        candidates = []
        box_size = 0
        while not candidates:
            box_size += 0.1
            candidates = [{'fac_index': i, 'lat': fac['lat'], 'lon': fac['lon']} for i, fac in enumerate(facs)
                          if fac['lat'] - box_size < cell['lat'] < fac['lat'] + box_size
                          and fac['lon'] - box_size < cell['lon'] < fac['lon'] + box_size]
        closest_outside_dist = distance((cell['lat'], cell['lon']), (cell['lat'], cell['lon'] + box_size)).km
        candidate_distances = [distance((cell['lat'], cell['lon']), (c['lat'], c['lon'])).km for c in candidates]
        min_index = np.argmin(candidate_distances)
        min_dist = candidate_distances[min_index]

        if min_dist > closest_outside_dist:
            box_size = box_size * min_dist / closest_outside_dist
            box_size += 0.1
            candidates = [{'fac_index': i, 'lat': fac['lat'], 'lon': fac['lon']} for (i, fac) in enumerate(facs)
                          if fac['lat'] - box_size < cell['lat'] < fac['lat'] + box_size
                          and fac['lon'] - box_size < cell['lon'] < fac['lon'] + box_size]
            candidate_distances = [distance((cell['lat'], cell['lon']), (c['lat'], c['lon'])).km for c in candidates]
            min_index = np.argmin(candidate_distances)

        min_fac_index = candidates[min_index]['fac_index']
        facs[min_fac_index]['catchment_population'] += cell['population']
    return facs


def load_raw_data(grid_population_input_filename, fac_location_input_filename):
    """
    load the input data and convert it to lists for fast iteration
    :param grid_population_input_filename: name of the csv file containing the latitude and longitude of each cell's
        centroid as well as its population
    :param fac_location_input_filename: name of the csv file containing the zip code, latitude and longitude of each
        facility
    :return: a list of dictionaries containing the latitude and longitude of each cell's centroid as well as
        its population; a list of dictionaries containing the zip code, latitude and longitude of each facility
    """
    cells_input_df = pd.read_csv(grid_population_input_filename)
    facs_input_df = pd.read_csv(fac_location_input_filename)
    cells = [{'lat': c['lat'], 'lon': c['lon'], 'population': c['population']} for i, c in cells_input_df.iterrows()]
    facs = [{'zip_code': fac['zip_code'], 'lat': fac['lat'], 'lon': fac['lon']} for _, fac in facs_input_df.iterrows()]
    return cells, facs


def save_data(facs, fac_data_output_filename):
    """
    save the zip code, latitude and longitude and catchment population of each facility to a csv file
    :param facs: a list of dictionaries containing the zip code, latitude, longitude and catchment population
        of each facility
    :param fac_data_output_filename: name of the output .csv file
    """
    facs_output_df = pd.DataFrame(facs)
    facs_output_df.to_csv(fac_data_output_filename, index=False)


if __name__ == "__main__":
    cells, facs = load_raw_data('bavaria_grid_population.csv', 'rc_locations.csv')
    facs = get_fac_catchment_population_old(cells, facs)
    save_data(facs, 'rc_data.csv')
