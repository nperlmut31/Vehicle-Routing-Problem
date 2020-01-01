import torch


class Fleet(object):

    def __init__(self, fleet_data, num_nodes, device='cpu'):

        self.device = device

        self.num_nodes = num_nodes

        #These fields are static
        self.start_time = fleet_data['start_time']
        self.car_start_node = fleet_data['car_start_node']

        self.batch_size = self.start_time.shape[0]
        self.num_cars = self.start_time.shape[1]


        #records the depot associated to each car
        self.depot = self.car_start_node.reshape(self.batch_size, self.num_cars).long()

        a = torch.arange(self.num_nodes).reshape(1, 1, -1).repeat(self.batch_size, self.num_cars, 1).to(self.device)
        b = self.depot.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
        self.num_depots = ((a == b).float().sum(dim=1) > 0).float().sum(dim=1).long()

        #These fields are dynamic. They will be updated as the tour is computed.
        self.time = self.start_time
        self.distance = torch.zeros(self.batch_size, self.num_cars, 1).to(self.device)
        self.late_time = torch.zeros(self.batch_size, self.num_cars, 1).to(self.device)

        #Path records the nodes that each car visited. Arrival times are the times when the car made it there.
        self.path = self.depot.unsqueeze(2)
        self.arrival_times = self.time


        #node is the current node of each car
        self.node = self.depot

        #traversed_nodes indicates which nodes have been visited
        self.traversed_nodes = self.initialize_traversed_nodes()

        #indicates whether or not the car has finished its route.
        self.finished = torch.zeros(self.batch_size, self.num_cars).to(self.device)



    def initialize_traversed_nodes(self):

        a = torch.arange(self.num_nodes).reshape(
            1, -1, 1).repeat(
            self.batch_size, 1, self.num_cars).float().to(self.device)

        b = self.depot.reshape(self.batch_size, 1, self.num_cars).repeat(1, self.num_nodes, 1).float()
        return ((a == b).float().sum(dim=2) > 0)


    def construct_vector(self):

        return self.time.reshape(self.batch_size, self.num_cars, 1)

