from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import torch
from fleet_beam_search_2.google_solver.convert_data import convert_data


class GoogleActor(object):

    def __init__(self, scale_factor=100):

        if scale_factor is None:
            self.scale_factor = 1
        else:
            self.scale_factor = scale_factor


    def __call__(self, input):

        total_distance = 0
        data = convert_data(input, self.scale_factor)
        batch_size = len(data)
        for datum in data:
            num_nodes = len(datum['distance_matrix'])
            routing, assignment = self.compute_route(datum)
            distance = self.compute_distance(routing, assignment, num_nodes)
            total_distance += distance

        return total_distance


    def compute_distance(self, routing, assignment, num_nodes):
        """Prints solution on console."""
        cumulative_route_distance = 0
        for vehicle_id in range(num_nodes):
            index = routing.Start(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                previous_index = index
                index = assignment.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            cumulative_route_distance += route_distance

        cumulative_route_distance = cumulative_route_distance / self.scale_factor
        return cumulative_route_distance



    def compute_route(self, input):
        distance_matrix = input['distance_matrix']
        time_matrix = input['time_matrix']
        time_windows = input['time_windows']
        volumes = input['volumes']
        weights = input['weights']

        num_nodes = len(distance_matrix)
        volume_capacity = input['volume_capacity']
        weight_capacity = input['volume_capacity']

        num_vehicles = len(volume_capacity)
        depot = 0

        manager = pywrapcp.RoutingIndexManager(
            num_nodes, num_vehicles, depot)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return time_matrix[from_node][to_node]

        def distance_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        def volume_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return volumes[from_node]

        def weight_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return weights[from_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        volume_index = routing.RegisterUnaryTransitCallback(volume_callback)
        weight_index = routing.RegisterUnaryTransitCallback(weight_callback)

        # this sets the cost to compute arc lengths
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

        time = 'Time'
        routing.AddDimension(
            time_callback_index,
            1000 * self.scale_factor,  # allow waiting time
            1000 * self.scale_factor,  # maximum time per vehicle
            True,  # Don't force start cumul to zero.
            time)

        distance = 'distance'
        routing.AddDimension(
            distance_callback_index,
            0,  # no slack
            3000 * self.scale_factor,  # vehicle maximum travel distance
            True,  # start cumul to zero
            distance)

        routing.AddDimensionWithVehicleCapacity(
            volume_index,
            0,  # null capacity slack
            volume_capacity,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Volume')

        routing.AddDimensionWithVehicleCapacity(
            weight_index,
            0,  # null capacity slack
            weight_capacity,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Weight')

        time_dimension = routing.GetDimensionOrDie(time)
        distance_dimension = routing.GetDimensionOrDie(distance)

        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(time_windows):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            x, y = int(time_window[0]), int(time_window[1])
            time_dimension.CumulVar(index).SetRange(x, y)

        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            x, y = int(time_windows[0][0]), int(time_windows[0][1])
            time_dimension.CumulVar(index).SetRange(x, y)

        # Not quite sure what this does
        for i in range(num_vehicles):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i)))

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # this stores the computed route
        assignment = routing.SolveWithParameters(search_parameters)
        return routing, assignment
