import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import shuffle
import os



class VRP_Dataset(Dataset):

    def __init__(self, dataset_size, num_nodes, num_depots, device='cpu', *args, **kwargs):
        super().__init__()

        self.device = device
        self.dataset_size = dataset_size
        self.num_nodes = num_nodes
        self.num_depots = num_depots

        self.dataset_size = dataset_size

        self.num_nodes = num_nodes
        num_cars = num_nodes
        self.num_depots = num_depots

        # fleet data
        launch_time = torch.zeros(self.dataset_size, num_cars, 1)
        car_start_node = torch.randint(low=0, high=num_depots, size=(self.dataset_size, num_cars, 1))

        self.fleet_data = {
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

        self.graph_data = {
            'start_times': start_times,
            'end_times': end_times,
            'depot': depot,
            'node_vector': node_positions,
            'distance_matrix': distance_matrix,
            'time_matrix': time_matrix
        }



    def compute_distance_matrix(self, node_positions):
        x = node_positions.unsqueeze(1).repeat(1, self.num_nodes, 1, 1)
        y = node_positions.unsqueeze(2).repeat(1, 1, self.num_nodes, 1)
        distance = (((x - y) ** 2).sum(dim=3)) ** (0.5)
        return distance



    def __getitem__(self, idx):

        A = {}
        for key in self.graph_data:
            A[key] = self.graph_data[key][idx].unsqueeze(0).to(self.device)

        B = {}
        for key in self.fleet_data:
            B[key] = self.fleet_data[key][idx].unsqueeze(0).to(self.device)

        return A, B



    def __len__(self):
        return self.dataset_size



    def collate(self, batch):

        graph_data = {key: [] for key in self.graph_data.keys()}
        fleet_data = {key: [] for key in self.fleet_data.keys()}

        for datum in batch:
            for key in datum[0]:
                value = datum[0][key]
                graph_data[key].append(value)

            for key in datum[1]:
                value = datum[1][key]
                fleet_data[key].append(value)

        for key in graph_data:
            graph_data[key] = torch.cat(graph_data[key], dim=0)
        for key in fleet_data:
            fleet_data[key] = torch.cat(fleet_data[key], dim=0)

        return graph_data, fleet_data



    def get_batch(self, idx, batch_size=10):
        batch = [self.__getitem__(i) for i in range(idx, idx+batch_size)]
        return self.collate(batch)



    def get_data(self):
        return self.graph_data, self.fleet_data



    def model_input_length(self):
        return 3 + self.graph_data['node_vector'].shape[2]



    def save_data(self, fp):
        data = (self.graph_data, self.fleet_data)
        with open(fp, 'wb') as f:
            torch.save(data, f)
            f.close()
