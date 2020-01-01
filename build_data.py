import os
from datetime import datetime, timedelta
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
q = os.path.join(dir_path, '..')
sys.path.append(q)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import shuffle


class Raw_VRP_Data(object):


    def __init__(self, dataset_size=1000,
                 num_nodes=50, num_depots=1,
                 *args, **kwargs):


        self.dataset_size = dataset_size

        self.num_nodes = num_nodes
        num_cars = num_nodes
        self.num_depots = num_depots

        # fleet data
        launch_time = torch.zeros(self.dataset_size, num_cars, 1)
        car_start_node = torch.randint(low=0, high=num_depots, size=(self.dataset_size, num_cars, 1))

        fleet = {
            'start_time': launch_time,
            'car_start_node': car_start_node,
        }

        # graph data
        a = torch.arange(num_nodes).reshape(1, 1, -1).repeat(self.dataset_size, num_cars, 1)
        b = car_start_node.repeat(1, 1, num_nodes)
        depot = ((a == b).sum(dim=1) > 0).float().unsqueeze(2)

        start_times = (torch.rand(self.dataset_size, num_nodes, 1) * 2 + 3) * (1 - depot)
        end_times = start_times + (0.1 + 0.5 * torch.rand(self.dataset_size, num_nodes, 1)) * (1 - depot)

        node_positions = torch.rand(dataset_size, num_nodes, 2)
        distance_matrix = self.compute_distance_matrix(node_positions)
        time_matrix = distance_matrix

        graph = {
            'start_times': start_times,
            'end_times': end_times,
            'depot': depot,
            'node_vector': node_positions,
            'distance_matrix': distance_matrix,
            'time_matrix': time_matrix
        }


        self.data = {
            'fleet': fleet,
            'graph': graph
        }


    def compute_distance_matrix(self, node_positions):
        x = node_positions.unsqueeze(1).repeat(1, self.num_nodes, 1, 1)
        y = node_positions.unsqueeze(2).repeat(1, 1, self.num_nodes, 1)
        distance = (((x - y)**2).sum(dim=3))**(0.5)
        return distance


    def get_data(self):
        return self.data


    def save_data(self, fp):
        torch.save(self.data, fp)



if __name__ == '__main__':

    size = 50000
    num_nodes = 20
    num_depots = 1
    num_components = 3

    start = datetime.now()

    raw_data = Raw_VRP_Data(dataset_size=size, num_nodes=num_nodes, num_cars=num_nodes, num_depots=num_depots, num_components=num_components)

    path = os.path.join(dir_path, 'Data', 'VRP_20.pt')

    with open(path, 'wb') as f:
        raw_data.save_data(f)
        f.close()

    end = datetime.now()

    s = (end - start).seconds
    print(s)
