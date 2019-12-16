import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import shuffle
import os


class VRP_Dataset(Dataset):

    def __init__(self, dataset_size=1000, num_nodes=20,
                 num_cars=50, num_depots=1,
                 num_components=2,
                 device='cpu',
                 *args, **kwargs):
        super().__init__()


        self.device = device
        self.num_components = num_components
        self.dataset_size = dataset_size



        self.num_nodes = num_nodes
        self.num_cars = num_cars
        self.num_depots = num_depots


        #fleet data
        launch_time = torch.zeros(self.dataset_size, num_cars, 1)
        volume_capacity = num_nodes/5 + (num_nodes/5)*torch.rand(self.dataset_size, num_cars, 1)
        weight_capacity = num_nodes / 5 + (num_nodes / 5) * torch.rand(self.dataset_size, num_cars, 1)
        car_start_node = torch.randint(low=0, high=num_depots, size=(self.dataset_size, num_cars, 1))
        car_node_compatibility = (torch.rand(self.dataset_size, num_cars, num_nodes) < 0.8).float()
        car_vector = self.construct_car_vector(car_node_compatibility)

        car_node_compatibility = 1 - 0*car_node_compatibility
        car_vector = 0*car_vector


        self.fleet_data = {
            'start_time': launch_time,
            'volume_capacity': volume_capacity,
            'weight_capacity': weight_capacity,
            'car_start_node': car_start_node,
            'car_vector': car_vector,
            'car_node_compatibility': car_node_compatibility,
        }


        #graph data
        a = torch.arange(num_nodes).reshape(1, 1, -1).repeat(self.dataset_size, num_cars, 1)
        b = car_start_node.repeat(1, 1, num_nodes)
        depot = ((a == b).sum(dim=1) > 0).float().unsqueeze(2)

        start_times = (torch.rand(self.dataset_size, num_nodes, 1)*2 + 3)*(1 - depot)
        end_times = start_times + (0.1 + 0.5 * torch.rand(self.dataset_size, num_nodes, 1))*(1 - depot)
        volume_demand = torch.rand(self.dataset_size, num_nodes, 1)*(1 - depot)
        weight_demand = torch.rand(self.dataset_size, num_nodes, 1)*(1 - depot)
        unload_times = 0.01*torch.rand(self.dataset_size, num_nodes, 1)*(1 - depot)


        node_node_compatibility = 1 + 0*self.create_node_node_campatibility_matrix(depot)
        D = self.construct_distance_matrix()
        distance_matrix = D + 0.01*torch.rand(self.dataset_size, num_nodes, num_nodes)
        time_matrix = D + 0.01*torch.rand(self.dataset_size, num_nodes, num_nodes)

        node_positions = self.construct_node_vectors(distance_matrix, time_matrix, node_node_compatibility)


        self.graph_data = {
            'start_times': start_times,
            'end_times': end_times,
            'volume_demand': volume_demand,
            'weight_demand': weight_demand,
            'unload_times': unload_times,
            'depot': depot,
            'node_vector': node_positions,
            'node_node_compatibility_matrix': node_node_compatibility,
            'distance_matrix': distance_matrix,
            'time_matrix': time_matrix
        }



    def create_node_node_campatibility_matrix(self, depot):

        mat = (torch.rand(self.dataset_size, self.num_nodes, self.num_nodes) < 0.8).float()

        diag = torch.diag(torch.ones(self.num_nodes)).unsqueeze(0).repeat(self.dataset_size, 1, 1)

        a = depot.reshape(self.dataset_size, 1, self.num_nodes).repeat(1, self.num_nodes, 1)
        b = depot.reshape(self.dataset_size, self.num_nodes, 1).repeat(1, 1, self.num_nodes)
        to_from_depot = (a + b > 0).float()

        compatible = (mat + diag + to_from_depot > 0).float()
        return compatible



    def construct_node_vectors(self, distance_matrix, time_matrix, node_node_compatibility_matrix):

        distance_mat = (distance_matrix - distance_matrix.mean())/distance_matrix.std()
        time_mat = (time_matrix - time_matrix.mean())/time_matrix.std()

        non_compat_mat = 1 - node_node_compatibility_matrix
        non_compat_mat = (non_compat_mat - non_compat_mat.mean())/non_compat_mat.std()


        svd_1 = TruncatedSVD(n_components=self.num_components)
        D = distance_mat.numpy()
        A = []
        for i in range(D.shape[0]):
            x = svd_1.fit_transform(D[i])
            A.append(torch.from_numpy(x).unsqueeze(0).float())
        a = torch.cat(A, dim=0)


        svd_2 = TruncatedSVD(n_components=self.num_components)
        T = time_mat
        B = []
        for i in range(T.shape[0]):
            x = svd_2.fit_transform(T[i])
            B.append(torch.from_numpy(x).unsqueeze(0).float())
        b = torch.cat(B, dim=0)


        svd_3 = TruncatedSVD(n_components=self.num_components)
        N = non_compat_mat
        C = []
        for i in range(N.shape[0]):
            x = svd_3.fit_transform(N[i])
            C.append(torch.from_numpy(x).unsqueeze(0).float())
        c = torch.cat(C, dim=0)


        node_vector = torch.cat([a, b, c], dim=2)
        return node_vector


    def construct_car_vector(self, car_node_compatibility_matrix):

        car_node_non_compat = 1 - car_node_compatibility_matrix
        car_node_non_compat = (car_node_non_compat - car_node_non_compat.mean())/car_node_non_compat.std()

        svd = TruncatedSVD(n_components=self.num_components)
        N = car_node_non_compat
        C = []
        for i in range(N.shape[0]):
            x = svd.fit_transform(N[i])
            C.append(torch.from_numpy(x).unsqueeze(0).float())
        x = torch.cat(C, dim=0)
        return x



    def construct_distance_matrix(self):

        positions = torch.rand(self.dataset_size, self.num_nodes, 2)
        a = positions.unsqueeze(1).repeat(1, self.num_nodes, 1, 1)
        b = positions.unsqueeze(2).repeat(1, 1, self.num_nodes, 1)

        distance_matrix = (((a - b)**(2)).sum(dim=3))**(0.5)
        return distance_matrix



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


    def model_input_length(self):
        return 5 + 3*self.num_components

    def decoder_input_length(self):
        return 3 + self.num_components

