import torch


class Normalization(object):

    def __init__(self, actor, normalize_position=False, device='cpu'):

        self.normalize_position = normalize_position
        self.device = device

        graph = actor.graph
        fleet = actor.fleet

        batch_size = graph.distance_matrix.shape[0]
        num_nodes = graph.distance_matrix.shape[1]
        num_cars = fleet.start_time.shape[1]

        self.greatest_drive_time = graph.time_matrix.reshape(batch_size, -1).max(dim=1)[0]
        self.greatest_distance = graph.distance_matrix.reshape(batch_size, -1).max(dim=1)[0]

        a = fleet.start_time.reshape(batch_size, -1)
        b = graph.start_time.reshape(batch_size, -1)
        self.earliest_start_time = torch.cat([a, b], dim=1).min(dim=1)[0]

        self.mean_positions = graph.node_positions.mean(dim=1)
        self.std_positions = torch.std(graph.node_positions, dim=1)


    def normalize(self, actor):

        batch_size = actor.graph.distance_matrix.shape[0]
        num_nodes = actor.graph.distance_matrix.shape[1]
        num_cars = actor.fleet.start_time.shape[1]

        d = self.greatest_distance.reshape(batch_size, 1, 1).repeat(1, num_nodes, num_nodes)
        actor.graph.distance_matrix = actor.graph.distance_matrix / d

        t = self.greatest_drive_time.reshape(batch_size, 1, 1).repeat(1, num_nodes, num_nodes)
        actor.graph.time_matrix = actor.graph.time_matrix / t

        s = self.earliest_start_time.reshape(batch_size, 1, 1).repeat(1, num_nodes, 1)
        t = self.greatest_drive_time.reshape(batch_size, 1, 1).repeat(1, num_nodes, 1)
        actor.graph.start_time = (actor.graph.start_time - s) / t
        actor.graph.end_time = (actor.graph.end_time - s) / t


        t = self.greatest_drive_time.reshape(batch_size)
        actor.fleet.late_time = actor.fleet.late_time / t

        s = actor.fleet.arrival_times.shape
        t = self.greatest_drive_time.reshape(batch_size, 1, 1).repeat(1, s[1], s[2])
        actor.fleet.arrival_times = actor.fleet.arrival_times / t


        if self.normalize_position:
            m = self.mean_positions.reshape(batch_size, 1, self.mean_positions.shape[-1]).repeat(1, num_nodes, 1)
            st = self.std_positions.reshape(batch_size, 1, self.std_positions.shape[-1]).repeat(1, num_nodes, 1)
            actor.graph.node_positions = (actor.graph.node_positions - m) / st

    def inverse_normalize(self, actor):

        batch_size = actor.graph.distance_matrix.shape[0]
        num_nodes = actor.graph.distance_matrix.shape[1]
        num_cars = actor.fleet.start_time.shape[1]

        d = self.greatest_distance.reshape(batch_size, 1, 1).repeat(1, num_nodes, num_nodes)
        actor.graph.distance_matrix = actor.graph.distance_matrix * d

        t = self.greatest_drive_time.reshape(batch_size, 1, 1).repeat(1, num_nodes, num_nodes)
        actor.graph.time_matrix = actor.graph.time_matrix * t

        s = self.earliest_start_time.reshape(batch_size, 1, 1).repeat(1, num_nodes, 1)
        t = self.greatest_drive_time.reshape(batch_size, 1, 1).repeat(1, num_nodes, 1)
        actor.graph.start_time = actor.graph.start_time * t + s
        actor.graph.end_time = actor.graph.end_time * t + s

        t = self.greatest_drive_time.reshape(batch_size)
        actor.fleet.late_time = actor.fleet.late_time * t

        s = actor.fleet.arrival_times.shape
        t = self.greatest_drive_time.reshape(batch_size, 1, 1).repeat(1, s[1], s[2])
        actor.fleet.arrival_times = actor.fleet.arrival_times * t


        if self.normalize_position:
            m = self.mean_positions.reshape(batch_size, 1, self.mean_positions.shape[-1]).repeat(1, num_nodes, 1)
            st = self.std_positions.reshape(batch_size, 1, self.std_positions.shape[-1]).repeat(1, num_nodes, 1)
            actor.graph.node_positions = actor.graph.node_positions * st + m
