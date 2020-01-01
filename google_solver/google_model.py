from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import torch
from just_time_windows.google_solver.convert_data import convert_data


class GoogleActor(object):

    def __init__(self, scale_factor=100):

        if scale_factor is None:
            self.scale_factor = 1
        else:
            self.scale_factor = scale_factor


    def __call__(self, input):

        drive_times = []
        data = convert_data(input, self.scale_factor)
        batch_size = len(data)
        for datum in data:
            routing, assignment = self.compute_route(datum)
            total_time = self.compute_total_time(datum, routing, assignment)
            drive_times.append(total_time)

        drive_times = torch.tensor(drive_times).float()
        return drive_times


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


    def compute_total_time(self, data, routing, assignment):
        time_dimension = routing.GetDimensionOrDie('Time')
        total_time = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                index = assignment.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            total_time += assignment.Min(time_var)
        total_time = total_time/self.scale_factor
        return total_time



    def compute_route(self, input):

        distance_matrix = input['distance_matrix']
        time_matrix = input['time_matrix']
        time_windows = input['time_windows']
        num_vehicles = input['num_vehicles']
        depot = input['depot']


        manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicles, depot)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return time_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            10000,  # allow waiting time
            10000,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)

        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(time_windows):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            a, b = int(time_window[0]), int(time_window[1])
            time_dimension.CumulVar(index).SetRange(a, b)

        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            a, b = int(time_windows[0][0]), int(time_windows[0][1])
            time_dimension.CumulVar(index).SetRange(a, b)

        for i in range(num_vehicles):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i)))

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        assignment = routing.SolveWithParameters(search_parameters)

        return routing, assignment




def evaluate_google_model(validation_dataset):
    validation_dataset.device = 'cpu'
    data = validation_dataset.get_data()
    model = GoogleActor(scale_factor=100)
    scores = model(data)
    return scores

