import torch
from fleet_beam_search_2.utils.actor_utils import widen

class Fleet(object):

    def __init__(self, fleet_data, device='cpu'):

        self.device = device

        #These fields are static
        self.start_time = fleet_data['start_time']
        self.volume_capacity = fleet_data['volume_capacity']
        self.weight_capacity = fleet_data['weight_capacity']
        self.car_vector = fleet_data['car_vector']
        self.car_node_compatibility = fleet_data['car_node_compatibility']
        self.car_start_node = fleet_data['car_start_node']


        self.batch_size = self.start_time.shape[0]
        self.num_cars = self.start_time.shape[1]
        self.num_nodes = self.car_node_compatibility.shape[2]

        #indicates which nodes are incompatible
        #this gets updated as the tour progresses
        self.incompatible_nodes = 1 - self.car_node_compatibility

        #records the depot associated to each car
        self.depot = self.car_start_node.reshape(self.batch_size, self.num_cars).long()

        a = torch.arange(self.num_nodes).reshape(1, 1, -1).repeat(self.batch_size, self.num_cars, 1).to(self.device)
        b = self.depot.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
        self.num_depots = ((a == b).float().sum(dim=1) > 0).float().sum(dim=1).long()

        #These fields are dynamic. They will be updated as the tour is computed.
        self.volume = self.volume_capacity
        self.weight = self.weight_capacity
        self.time = self.start_time

        #Path records the nodes that each car visited. Arrival times are the times when the car made it there.
        self.path = self.depot.unsqueeze(2)
        self.arrival_times = self.time
        self.late_time = torch.zeros(self.batch_size).to(self.device)

        #node is the current node of each car
        self.node = self.depot

        #traversed_nodes indicates which nodes have been visited
        self.traversed_nodes = self.initialize_traversed_nodes()

        #max capacities
        self.max_volume_capacity = self.volume_capacity.max()
        self.max_weight_capacity = self.weight_capacity.max()

        #indicates whether or not the car has finished its route.
        self.finished = torch.zeros(self.batch_size, self.num_cars).to(self.device)


    def initialize_traversed_nodes(self):

        a = torch.arange(self.num_nodes).reshape(
            1, -1, 1).repeat(
            self.batch_size, 1, self.num_cars).float().to(self.device)

        b = self.depot.reshape(self.batch_size, 1, self.num_cars).repeat(1, self.num_nodes, 1).float()
        return ((a == b).float().sum(dim=2) > 0)


    def construct_vector(self):

        batch_size = self.time.shape[0]

        L = [self.time.reshape(batch_size, self.num_cars, 1),
             self.volume.reshape(batch_size, self.num_cars, 1),
             self.weight.reshape(batch_size, self.num_cars, 1),
             self.car_vector]

        return torch.cat(L, dim=2)

