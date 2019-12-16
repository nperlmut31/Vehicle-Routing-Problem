import torch
from fleet_beam_search_2.utils.actor_utils import widen


class Graph(object):

    def __init__(self, graph_data, device='cpu'):

        self.device = device

        self.start_time = graph_data['start_times']
        self.end_time = graph_data['end_times']
        self.volume_demand = graph_data['volume_demand']
        self.weight_demand = graph_data['weight_demand']
        self.unload_times = graph_data['unload_times']
        self.depot = graph_data['depot']
        self.node_positions = graph_data['node_vector']
        self.distance_matrix = graph_data['distance_matrix']
        self.time_matrix = graph_data['time_matrix']
        self.node_node_compatibility = graph_data['node_node_compatibility_matrix']

        self.num_nodes = self.distance_matrix.shape[1]
        self.batch_size = self.distance_matrix.shape[0]

        self.correct_depot_features()

        self.time_window_compatibility = self.compute_time_window_compatibility()
        self.max_dist = self.distance_matrix.max()
        self.max_drive_time = self.time_matrix.max()

        self.incorporate_unload_times()


    def correct_depot_features(self):
        self.weight_demand = self.weight_demand * (1 - self.depot)
        self.volume_demand = self.volume_demand * (1 - self.depot)
        self.start_time = self.start_time * (1 - self.depot)
        self.end_time = self.end_time * (1 - self.depot)
        self.unload_times = self.unload_times * (1 - self.depot)


    def construct_vector(self):
        L = [self.node_positions, self.start_time, self.end_time,
             self.volume_demand, self.weight_demand, self.depot]
        self.vector = torch.cat(L, dim=2)
        return self.vector


    def get_drive_times(self, from_nodes, to_nodes):

        num_elements = from_nodes.shape[1]
        assert num_elements == to_nodes.shape[1]

        ind_1 = from_nodes.reshape(self.batch_size, num_elements, 1).reshape(1, 1, self.num_nodes)
        dist = torch.gather(self.time_matrix, dim=1, index=ind_1)

        ind_2 = to_nodes.reshape(self.batch_size, num_elements, 1)
        drive_times = torch.gather(dist, dim=2, index=ind_2)
        return drive_times


    def get_distances(self, from_nodes, to_nodes):
        num_elements = from_nodes.shape[1]
        assert num_elements == to_nodes.shape[1]

        ind_1 = from_nodes.reshape(self.batch_size, num_elements, 1).reshape(1, 1, self.num_nodes)
        dist = torch.gather(self.distance_matrix, dim=1, index=ind_1)

        ind_2 = to_nodes.reshape(self.batch_size, num_elements, 1)
        distances = torch.gather(dist, dim=2, index=ind_2)
        return distances


    def compute_time_window_compatibility(self):
        #check for drive times being too long
        x = self.start_time.reshape(self.batch_size, self.num_nodes, 1).repeat(1, 1, self.num_nodes)
        y = self.end_time.reshape(self.batch_size, 1, self.num_nodes).repeat(1, self.num_nodes, 1)
        time_mask = (x + self.time_matrix <= y).float()
        return time_mask


    def incorporate_unload_times(self):
        self.end_time = self.end_time - self.unload_times


