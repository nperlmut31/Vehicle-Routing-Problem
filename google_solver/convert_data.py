
import torch
import numpy as np


def convert_tensor(x):
    x = x.long().numpy().astype('int')

    if len(x.shape) == 1:
        return list(x)
    else:
        return [list(x[i]) for i in range(x.shape[0])]



def make_time_windows(start_time, end_time):
    return torch.cat([start_time, end_time], dim=2)



def convert_data(input, scale_factor):

    graph_data, fleet_data = input

    start_times = graph_data['start_times']
    end_times = graph_data['end_times']

    distance_matrix = graph_data['distance_matrix']
    time_matrix = graph_data['time_matrix']

    time_windows = make_time_windows(start_times, end_times)


    batch_size = distance_matrix.shape[0]
    data = []
    for i in range(batch_size):

        num_vehicles = distance_matrix[i].shape[1]

        space_mat = distance_matrix[i] * scale_factor
        time_mat = time_matrix[i] * scale_factor
        windows = time_windows[i] * scale_factor

        space_mat = convert_tensor(space_mat)
        time_mat = convert_tensor(time_mat)
        windows = convert_tensor(windows)


        D = {'distance_matrix': space_mat,
             'time_matrix': time_mat,
             'time_windows': windows,
             'depot': 0,
             'num_vehicles': num_vehicles
             }

        data.append(D)
    return data