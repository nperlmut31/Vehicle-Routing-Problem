
import torch
import numpy as np


def convert_tensor(x):
    x = x.long().numpy().astype('int')

    if len(x.shape) == 1:
        return list(x)
    else:
        return [list(x[i]) for i in range(x.shape[0])]


def apply_unload_times(end_times, unload_times):
    return end_times - unload_times


def correct_matrices(distance_matrix, drive_time_matrix, compatibility_matrix):

    max_distance = distance_matrix.max()
    max_drive_time = drive_time_matrix.max()

    distance_matrix = distance_matrix + (1 - compatibility_matrix)*max_distance*100
    drive_time_matrix = distance_matrix + (1 - compatibility_matrix)*max_drive_time*100

    return distance_matrix, drive_time_matrix


def make_time_windows(start_time, end_time):
    return torch.cat([start_time, end_time], dim=2)



def convert_data(input, scale_factor):

    graph_data, fleet_data = input

    start_times = graph_data['start_times']
    end_times = graph_data['end_times']
    volume_demand = graph_data['volume_demand']
    weight_demand = graph_data['weight_demand']
    unload_times = graph_data['unload_times']
    depot = graph_data['depot']
    node_vector = graph_data['node_vector']
    node_node_compatibilty_matrix = graph_data['node_node_compatibility_matrix']
    distance_matrix = graph_data['distance_matrix']
    time_matrix = graph_data['time_matrix']


    car_start_time = fleet_data['start_time']
    volume_capacity = fleet_data['volume_capacity']
    weight_capacity = fleet_data['weight_capacity']
    car_start_node = fleet_data['car_start_node']


    end_times = apply_unload_times(end_times, unload_times)
    distance_matrix, time_matrix = correct_matrices(distance_matrix,
                                                    time_matrix,
                                                    node_node_compatibilty_matrix)

    time_windows = make_time_windows(start_times, end_times)


    batch_size = distance_matrix.shape[0]
    data = []
    for i in range(batch_size):

        space_mat = distance_matrix[i] * scale_factor
        time_mat = time_matrix[i] * scale_factor
        windows = time_windows[i] * scale_factor

        volume_d = volume_demand[i].squeeze(1) * scale_factor
        weight_d = weight_demand[i].squeeze(1) * scale_factor

        volume_cap = volume_capacity[i].squeeze(1) * scale_factor
        weight_cap = weight_capacity[i].squeeze(1) * scale_factor



        space_mat = convert_tensor(space_mat)
        time_mat = convert_tensor(time_mat)
        windows = convert_tensor(windows)

        volume_d = convert_tensor(volume_d)
        weight_d = convert_tensor(weight_d)

        volume_cap = convert_tensor(volume_cap)
        weight_cap = convert_tensor(weight_cap)


        D = {'distance_matrix': space_mat,
             'time_matrix': time_mat,
             'time_windows': windows,
             'volumes': volume_d,
             'weights': weight_d,
             'volume_capacity': volume_cap,
             'weight_capacity': weight_cap}

        data.append(D)
    return data