import torch
import torch.nn as nn
import torch.nn.functional as F
from fleet_beam_search_2.Actor.graph import Graph
from fleet_beam_search_2.Actor.fleet import Fleet
from fleet_beam_search_2.utils.actor_utils import widen, widen_dict


#This is the updated version of the actor
class Actor(nn.Module):

    def __init__(self, model=None, num_movers=5, num_neighbors_encoder=5,
                 num_neighbors_action=5, normalize=False,
                 device='cpu'):
        super().__init__()

        self.device = device
        self.num_movers = num_movers
        self.num_neighbors_encoder = num_neighbors_encoder
        self.num_neighbors_action = num_neighbors_action

        self.apply_normalization = normalize
        self.normalization_params = None


        if model is None:
            self.mode = 'nearest_neighbors'
        else:
            self.encoder = model.encoder.to(self.device)
            self.decoder = model.decoder.to(self.device)
            self.projections = model.projections.to(self.device)
            self.mode = 'train'


        self.sample_size = 1
        self.beam_size = 1


    def train_mode(self, sample_size=1):
        self.train()
        self.sample_size = sample_size
        self.mode = 'train'


    def greedy_search(self):
        self.eval()
        self.mode = 'greedy'


    def nearest_neighbors(self):
        self.eval()
        self.mode = 'nearest_neighbors'


    def sample_mode(self, sample_size=10):
        self.sample_size = sample_size
        self.eval()
        self.mode = 'sample'


    def beam_search(self, beam_size=10):
        self.beam_size=beam_size
        self.eval()
        self.mode = 'beam_search'


    def update_batch_size(self):
        self.batch_size = self.fleet.time.shape[0]
        self.fleet.batch_size = self.fleet.time.shape[0]
        self.graph.batch_size  = self.fleet.time.shape[0]


    def widen_data(self):

        F = dir(self.fleet)
        for s in F:
            x = getattr(self.fleet, s)
            if isinstance(x, torch.Tensor):
                if len(x.shape) > 0:
                    y = widen(x, factor=self.beam_size)
                    setattr(self.fleet, s, y)

        G = dir(self.graph)
        for s in G:
            x = getattr(self.graph, s)
            if isinstance(x, torch.Tensor):
                if len(x.shape) > 0:
                    y = widen(x, factor=self.beam_size)
                    setattr(self.graph, s, y)

        self.node_embeddings = widen(self.node_embeddings, factor=self.beam_size)
        self.node_projections = widen(self.node_projections, factor=self.batch_size)
        self.log_probs = widen(self.log_probs, factor=self.beam_size)


    def select_data(self, index):
        m = index.max().item()

        F = dir(self.fleet)
        for s in F:
            x = getattr(self.fleet, s)
            if isinstance(x, torch.Tensor):
                if (len(x.shape) > 0) and (x.shape[0] >= m):
                    setattr(self.fleet, s, x[index])

        G = dir(self.graph)
        for s in G:
            x = getattr(self.graph, s)
            if isinstance(x, torch.Tensor):
                if (len(x.shape) > 0) and (x.shape[0] >= m):
                    setattr(self.graph, s, x[index])

        self.node_embeddings = self.node_embeddings[index]
        self.node_projections = self.node_projections[index]
        self.log_probs = self.log_probs[index]



    def forward(self, batch, *args, **kwargs):

        graph_data, fleet_data = batch

        self.original_batch_size = graph_data['distance_matrix'].shape[0]
        self.batch_size = self.original_batch_size

        if self.mode == 'sample':
            graph_data = widen_dict(graph_data, self.sample_size)
            fleet_data = widen_dict(fleet_data, self.sample_size)


        self.graph = Graph(graph_data, device=self.device)
        self.fleet = Fleet(fleet_data, device=self.device)
        self.num_nodes = self.graph.distance_matrix.shape[1]
        self.num_cars = self.fleet.volume_capacity.shape[1]
        self.update_batch_size()


        if self.apply_normalization:
            self.normalize()


        self.num_depots = self.fleet.num_depots.max().item()
        self.num_movers_corrected = int(min(max(self.num_movers, self.num_depots), self.num_cars))


        if self.mode != 'nearest_neighbors':
            encoder_input = self.graph.construct_vector()
            encoder_mask = self.compute_encoder_mask()
            self.node_embeddings = self.encoder(encoder_input, encoder_mask)
            self.node_projections = self.projections(self.node_embeddings)


        self.log_probs = torch.zeros(self.batch_size).to(self.device)
        self.counter = 0
        while self.loop_condition() and (self.counter < self.num_nodes*4):

            unavailable_moves = self.check_non_depot_options(use_time=True)
            mover_indices = self.get_mover_indices(unavailable_moves=unavailable_moves)
            action_mask = self.compute_action_mask(mover_indices=mover_indices,
                                                   unavailable_moves=unavailable_moves)

            if self.mode != 'nearest_neighbors':
                decoder_input = self.construct_decoder_input(mover_indices=mover_indices)
                decoder_mask = self.compute_decoder_mask(mover_indices=mover_indices,
                                                         unavailable_moves=unavailable_moves)

                decoder_output = self.decoder(decoder_input=decoder_input,
                                            projections=self.node_projections,
                                            mask=decoder_mask)
            else:
                decoder_output = None

            next_node, car_to_move, log_prob = self.compute_action(decoder_output, action_mask, mover_indices)

            if self.mode == 'beam_search':
                self.widen_data()
                self.update_batch_size()

            self.log_probs += log_prob

            self.update_time(next_node, car_to_move)
            self.update_storage(next_node, car_to_move)
            self.update_node_path(next_node, car_to_move)
            self.update_compatibility()
            self.update_traversed_nodes()

            self.return_to_depot()
            self.update_traversed_nodes()

            if (self.mode == 'beam_search') and (self.counter > 0):
                self.consolidate_beams()

            self.update_batch_size()
            self.counter += 1


        if self.mode == 'beam_search':
            # we now must select out the best beam
            p = self.log_probs.reshape(self.original_batch_size, self.beam_size)
            a = torch.argmax(p, dim=1)

            b = torch.arange(self.original_batch_size).to(self.device)
            b = b * self.beam_size

            index = a + b
            self.select_data(index=index)

        self.batch_size = self.fleet.time.shape[0]
        self.adjust_arrival_times()


        output = {
            'cost': self.compute_cost().detach(),
            'log_probs': self.log_probs,
            'late_time': self.fleet.late_time.detach(),
            'incomplete': self.check_complete(),
            'path': self.fleet.path,
            'arrival_times': self.fleet.arrival_times
        }


        if self.mode == 'sample':

            cost = output.get('cost')
            batch_size = cost.shape[0]//self.sample_size
            cost_1 = cost.reshape(batch_size, self.sample_size)
            arg_min = torch.argmin(cost_1, dim=1)

            def recover_results(x, ind, sample_size):
                batch_size = x.shape[0]//sample_size
                x_1 = x.reshape(batch_size, sample_size, *x.shape[1:])
                s = [1 for i in range(len(x.shape)-1)]
                ind = ind.reshape(batch_size, 1, *s).repeat(1, 1, *x.shape[1:])
                y = torch.gather(x_1, dim=1, index=ind)
                return y

            for key in output:
                x = output[key]
                y = recover_results(x, arg_min, self.sample_size)
                output[key] = y

        return output


    def consolidate_beams(self):
        p = self.log_probs.reshape(self.original_batch_size, self.beam_size * self.beam_size)
        a = torch.topk(p, dim=1, k=self.beam_size, largest=True)[1]

        b = torch.arange(self.original_batch_size).unsqueeze(1).repeat(1, self.beam_size).to(self.device)
        b = b * self.beam_size * self.beam_size

        ind = (a + b).reshape(-1)
        self.select_data(ind)


    def adjust_arrival_times(self):

        if self.apply_normalization and (self.normalization_params is not None):
            num_steps = self.fleet.arrival_times.shape[2]
            a = self.normalization_params['earliest_start_time'].reshape(self.batch_size, 1, 1).repeat(1, self.num_cars, num_steps)
            b = self.normalization_params['greatest_drive_time'].reshape(self.batch_size, 1, 1).repeat(1, self.num_cars, num_steps)
            self.fleet.arrival_times = self.fleet.arrival_times*b + a

            a = self.normalization_params['earliest_start_time']
            b = self.normalization_params['greatest_drive_time']
            self.fleet.late_time = self.fleet.late_time*b + a


    def compute_encoder_mask(self):
        #check for drive times being too long
        time_window_non_compatibility = 1 - self.graph.time_window_compatibility

        #compatibility mask
        non_compatibility_mask = 1 - self.graph.node_node_compatibility

        #compute diag mask
        diag = torch.diag(torch.ones(self.num_nodes)).unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)

        # compute neighbors mask
        m = (time_window_non_compatibility + non_compatibility_mask + diag > 0).float()

        dist = self.graph.distance_matrix*(1 - m) + m*self.graph.max_dist*10

        K = min(self.num_nodes, self.num_neighbors_encoder)
        neighbors_index = torch.topk(dist, k=K, dim=2, largest=False)[1] # ~ [batch, num_nodes, num_neighbors]

        a = torch.arange(self.num_nodes).reshape(1, 1, -1, 1).repeat(self.batch_size, self.num_nodes, 1, K).to(self.device)
        b = neighbors_index.unsqueeze(2).repeat(1, 1, self.num_nodes, 1)
        neighbors_mask = (a == b).float().sum(dim=3)


        m = (time_window_non_compatibility + non_compatibility_mask == 0).float()
        neighbs_time_mask = neighbors_mask*m

        #compute depot mask
        v = self.graph.depot.reshape(self.batch_size, 1, self.num_nodes).repeat(1, self.num_nodes, 1)
        w = self.graph.depot.reshape(self.batch_size, self.num_nodes, 1).repeat(1, 1, self.num_nodes)
        depot_mask = (v + w > 0).float()

        diag = torch.diag(torch.ones(self.num_nodes)).unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)

        mask = (neighbs_time_mask + depot_mask + diag > 0).float()
        return mask


    def compute_depoyment_priority_score(self):
        #number of available nodes
        available_nodes = (self.check_non_depot_options().float() == 0).float()
        num_available_nodes = available_nodes.sum(dim=2)

        #excess
        ind = self.fleet.node.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
        distances = torch.gather(self.graph.distance_matrix, dim=1, index=ind)

        max_distance = self.graph.distance_matrix.reshape(self.batch_size, -1).max(dim=1)[0].reshape(
                                    self.batch_size, 1, 1).repeat(1, self.num_cars, self.num_nodes)
        excess = ((max_distance - distances)*available_nodes).sum(dim=2)

        #available volume
        volume = self.fleet.volume.reshape(self.batch_size, self.num_cars)

        max_excess = excess.max()
        max_volume = volume.max()

        score = (num_available_nodes + max_excess + max_volume)*100 + (excess + max_volume)*10 + volume
        return score


    def compute_excess(self):
        available_nodes = (self.check_non_depot_options().float() == 0).float()

        ind = self.fleet.node.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
        distances = torch.gather(self.graph.distance_matrix, dim=1, index=ind)

        max_distance = self.graph.distance_matrix.reshape(self.batch_size, -1).max(dim=1)[0].reshape(
                                    self.batch_size, 1, 1).repeat(1, self.num_cars, self.num_nodes)

        score = ((max_distance - distances)*available_nodes).sum(dim=2)
        return score


    def check_complete(self):
        has_untraversed_nodes = ((self.fleet.traversed_nodes == 0).float().sum(dim=1) > 0)
        has_cars_in_field = ((self.fleet.node != self.fleet.depot).sum(dim=1) > 0)
        incomplete = (has_untraversed_nodes | has_cars_in_field)
        return incomplete


    def loop_condition(self):
        a = self.check_complete().sum().item()
        if a == 0:
            return False
        else:
            return True


    def compute_loss(self):
        self.loss = self.total_distance.detach()*self.log_probs


    def compute_cost(self):
        p_2 = self.fleet.path
        p_1 = torch.cat([p_2[:,:,-1:], p_2[:,:,:-1]], dim=2)

        mat = self.graph.distance_matrix.reshape(self.batch_size, 1, self.num_nodes, self.num_nodes).repeat(1, self.num_cars, 1, 1)
        ind_1 = p_1.unsqueeze(3).repeat(1, 1, 1, self.num_nodes)

        d = torch.gather(mat, dim=2, index=ind_1)
        ind_2 = p_2.unsqueeze(3)
        pairwise_distances = torch.gather(d, dim=3, index=ind_2).squeeze(3)

        self.car_distances = pairwise_distances.sum(dim=2)
        self.total_distance = self.car_distances.sum(dim=1)

        if self.apply_normalization and (self.normalization_params is not None):
            distance_multiplier = self.normalization_params['greatest_distance']
            self.total_distance = self.total_distance*distance_multiplier.reshape(self.batch_size)

        return self.total_distance


    def update_traversed_nodes(self):
        x = ((self.graph.volume_demand == 0) & (self.graph.weight_demand == 0)).float()
        self.fleet.traversed_nodes = x.reshape(self.batch_size, self.num_nodes)



    def compute_action(self, decoder_output, action_mask, mover_indices):

        if self.mode == 'nearest_neighbors':

            assert decoder_output is None

            num_movers = mover_indices.shape[1]
            mover_nodes = torch.gather(self.fleet.node, dim=1, index=mover_indices)

            ind = mover_nodes.reshape(self.batch_size, num_movers, 1).repeat(1, 1, self.num_nodes)
            distances = torch.gather(self.graph.distance_matrix, dim=1, index=ind)

            a = mover_nodes.reshape(self.batch_size, num_movers, 1).repeat(1, 1, self.num_nodes)
            b = torch.arange(self.num_nodes).reshape(1, 1, self.num_nodes).repeat(self.batch_size, num_movers, 1).to(self.device)
            current_node_indicator = (a == b).float()

            a = action_mask.reshape(self.batch_size, -1)
            no_options = (a.sum(dim=1) == 0).reshape(-1, 1, 1).repeat(1, num_movers, self.num_nodes).float()

            mask = action_mask * (1 - no_options) + current_node_indicator * no_options
            masked_distances = distances*mask + self.graph.max_dist*(1 - mask)*10

            x = masked_distances.reshape(self.batch_size, -1)
            ind = torch.argmin(x, dim=1).unsqueeze(1)

            mover_index = ind // self.num_nodes
            next_node = ind % self.num_nodes
            car_to_move = torch.gather(mover_indices, dim=1, index=mover_index)

            log_prob = torch.zeros(self.batch_size).to(self.device)
            return next_node, car_to_move, log_prob

        else:

            assert decoder_output is not None

            num_movers = mover_indices.shape[1]
            assert (num_movers == action_mask.shape[1]) and (num_movers == decoder_output.shape[1])

            a = action_mask.reshape(self.batch_size, -1)
            no_options = (a.sum(dim=1) == 0).reshape(-1, 1, 1).repeat(1, num_movers, self.num_nodes).float()

            mover_nodes = torch.gather(self.fleet.node, dim=1, index=mover_indices)
            a = torch.arange(self.num_nodes).reshape(1, 1, -1).repeat(self.batch_size, num_movers, 1).to(self.device)
            b = mover_nodes.reshape(self.batch_size, num_movers, 1).repeat(1, 1, self.num_nodes)
            default_option = (a == b).float()

            mask = action_mask*(1 - no_options) + default_option*no_options
            masked_decoder_output = decoder_output + mask.log()

            x = masked_decoder_output.reshape(self.batch_size, -1)
            probs = torch.softmax(x, dim=1)

            if self.mode == 'greedy':
                prob, ind = torch.max(probs, dim=1)[0].unsqueeze(1), torch.argmax(probs, dim=1).unsqueeze(1)

            elif self.mode in {'train', 'sample'}:
                ind = torch.multinomial(probs, num_samples=1)
                prob = torch.gather(probs, dim=1, index=ind)

            elif self.mode == 'beam_search':
                prob, ind = torch.topk(probs, dim=1, k=self.beam_size, largest=True)

            mover_index = ind // self.num_nodes
            next_node = ind % self.num_nodes
            car_to_move = torch.gather(mover_indices, dim=1, index=mover_index)

            next_node = next_node.reshape(-1)
            car_to_move = car_to_move.reshape(-1)
            prob = prob.reshape(-1)

            return next_node, car_to_move, prob.log()



    def update_time(self, next_node, car_to_move):
        #get current node of mover
        ind = car_to_move.reshape(self.batch_size, 1)
        n = self.fleet.node.reshape(self.batch_size, self.num_cars)
        current_node = torch.gather(n, dim=1, index=ind).squeeze(1)

        #compute time to next node
        ind_1 = current_node.reshape(-1, 1, 1).repeat(1, 1, self.num_nodes)
        drive_times = torch.gather(self.graph.time_matrix, dim=1, index=ind_1).squeeze(1)
        ind_2 = next_node.reshape(-1, 1)
        time_to_next_node = torch.gather(drive_times, dim=1, index=ind_2).squeeze(1)

        #compute start time at next node
        start_time = self.graph.start_time.reshape(self.batch_size, self.num_nodes)
        ind = next_node.reshape(self.batch_size, 1)
        next_start_time = torch.gather(start_time, dim=1, index=ind).squeeze(1)

        #compute current time
        t = self.fleet.time.reshape(self.batch_size, self.num_cars)
        current_node_time = torch.gather(t, dim=1, index=car_to_move.reshape(self.batch_size, 1)).squeeze(1)

        #compute updated_time
        a = time_to_next_node + current_node_time
        b = next_start_time
        updated_time = (a > b).float()*a + (a <= b).float()*b

        #compute end_time at next node
        ind = next_node.reshape(-1, 1)
        end_times = self.graph.end_time.reshape(self.batch_size, self.num_nodes)
        end_time_next_node = torch.gather(end_times, dim=1, index=ind).squeeze(1)

        #update_late_time
        self.fleet.late_time = self.fleet.late_time + F.relu(updated_time - end_time_next_node)

        #compute mover mask
        a = torch.arange(self.num_cars).reshape(1, -1).repeat(self.batch_size, 1).to(self.device)
        b = car_to_move.reshape(self.batch_size, 1).repeat(1, self.num_cars)
        update_mask = (a == b).float().unsqueeze(2)

        #update time
        t = updated_time.reshape(self.batch_size, 1, 1).repeat(1, self.num_cars, 1)
        self.fleet.time = self.fleet.time*(1 - update_mask) + t*update_mask



    def reset_storage(self):
        node = self.fleet.node.long().reshape(self.batch_size, self.num_cars)
        depot = self.fleet.depot.long().reshape(self.batch_size, self.num_cars)
        at_depot = (node == depot).float().reshape(self.batch_size, self.num_cars, 1)

        self.fleet.volume = self.fleet.volume*(1 - at_depot) + self.fleet.volume_capacity*at_depot
        self.fleet.weight = self.fleet.weight*(1 - at_depot) + self.fleet.weight_capacity*at_depot


    #this method simultaneously updates the storage of the fleet and of the graph
    def update_storage(self, next_node, car_to_move):
        #get demands at next node
        ind = next_node.reshape(self.batch_size, 1)
        v = self.graph.volume_demand.reshape(self.batch_size, self.num_nodes)
        w = self.graph.weight_demand.reshape(self.batch_size, self.num_nodes)
        next_node_volume = torch.gather(v, dim=1, index=ind).squeeze(1)
        next_node_weight = torch.gather(w, dim=1, index=ind).squeeze(1)

        #compute mover mask
        a = torch.arange(self.num_cars).reshape(1, -1).repeat(self.batch_size, 1).to(self.device)
        b = car_to_move.reshape(self.batch_size, 1).repeat(1, self.num_cars)
        fleet_mask = (a == b).float()

        #compute graph mask
        a = torch.arange(self.num_nodes).reshape(1, -1).repeat(self.batch_size, 1)
        b = next_node.reshape(self.batch_size, 1).repeat(1, self.num_nodes)
        graph_mask = (a == b).float().unsqueeze(2)


        #next node's volume and weight
        volume = next_node_volume.reshape(self.batch_size, 1).repeat(1, self.num_cars)
        weight = next_node_weight.reshape(self.batch_size, 1).repeat(1, self.num_cars)

        #current car's volume and weight
        v = self.fleet.volume.reshape(self.batch_size, self.num_cars)
        w = self.fleet.weight.reshape(self.batch_size, self.num_cars)
        ind = car_to_move.reshape(self.batch_size, 1)
        car_volume = torch.gather(v, dim=1, index=ind).unsqueeze(1).repeat(1, self.num_nodes, 1)
        car_weight = torch.gather(w, dim=1, index=ind).unsqueeze(1).repeat(1, self.num_nodes, 1)

        #compute update values
        new_graph_volume = F.relu(self.graph.volume_demand - car_volume) * graph_mask + self.graph.volume_demand * (1 - graph_mask)
        new_graph_weight = F.relu(self.graph.weight_demand - car_weight) * graph_mask + self.graph.weight_demand * (1 - graph_mask)
        new_fleet_volume = F.relu(self.fleet.volume - (fleet_mask * volume).unsqueeze(2))
        new_fleet_weight = F.relu(self.fleet.weight - (fleet_mask * weight).unsqueeze(2))

        #update
        self.graph.volume_demand = new_graph_volume
        self.graph.weight_demand = new_graph_weight
        self.fleet.volume = new_fleet_volume
        self.fleet.weight = new_fleet_weight


    def update_node_path(self, next_node, car_to_move):
        #compute mover mask
        a = torch.arange(self.num_cars).reshape(1, -1).repeat(self.batch_size, 1).to(self.device)
        b = car_to_move.reshape(self.batch_size, 1).repeat(1, self.num_cars)
        update_mask = (a == b).long()

        new_node = next_node.reshape(self.batch_size, 1).repeat(1, self.num_cars)
        self.fleet.node = update_mask*new_node + self.fleet.node*(1 - update_mask)

        L = [self.fleet.path, self.fleet.node.unsqueeze(2)]
        self.fleet.path = torch.cat(L, dim=2)

        t = self.fleet.time.reshape(self.batch_size, self.num_cars, 1)
        H = [self.fleet.arrival_times, t]
        self.fleet.arrival_times = torch.cat(H, dim=2)


    def update_compatibility(self):
        ind = self.fleet.node.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes).long()
        mat = self.graph.node_node_compatibility
        x = torch.gather(mat, dim=1, index=ind)
        updated_incompatibility = ((self.fleet.incompatible_nodes + (1 - x)) > 0).float()
        self.fleet.incompatible_nodes = updated_incompatibility


    def check_non_depot_options(self, use_time=True):
        '''
        Output is a byte tensor of shape [batch, num_cars, num_nodes] with entry (i,j) = 1 if
        the move of car i to node j is invalid
        '''

        #check for incompatibility
        incompatible = (self.fleet.incompatible_nodes == 1)

        #check for volume
        low_volume = 1 - self.check_volume()

        #check for weight
        low_weight = 1 - self.check_weight()

        if use_time:
            #check for arrival times
            too_far = 1 - self.check_arival_times()

        #is depot
        a = self.fleet.depot.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
        b = torch.arange(self.num_nodes).reshape(1, 1, -1).repeat(self.batch_size, self.num_cars, 1).to(self.device)
        is_depot = (a == b)

        #check traversed nodes
        a = self.fleet.traversed_nodes.reshape(self.batch_size, 1, self.num_nodes).repeat(1, self.num_cars, 1)
        traversed_nodes = (a == 1)

        # has value of 1 if the move to that node is NOT possible
        if use_time:
            unavailable_moves = (low_volume | low_weight | too_far | is_depot | incompatible | traversed_nodes)
        else:
            unavailable_moves = (low_volume | low_weight | is_depot | incompatible | traversed_nodes)

        return unavailable_moves


    def compute_decoder_mask(self, mover_indices, unavailable_moves):

        num_movers = mover_indices.shape[1]

        mover_nodes = torch.gather(self.fleet.node, dim=1, index=mover_indices)

        ind = mover_indices.reshape(self.batch_size, num_movers, 1).repeat(1, 1, self.num_nodes)
        unavailable_moves = torch.gather(unavailable_moves, dim=1, index=ind)

        no_options = ((1 - unavailable_moves.float()).sum(dim=2) == 0).unsqueeze(2).repeat(1, 1, self.num_nodes).float()

        a = mover_nodes.reshape(self.batch_size, num_movers, 1).repeat(1, 1, self.num_nodes)
        b = torch.arange(self.num_nodes).reshape(1, 1, -1).repeat(self.batch_size, num_movers, 1).to(self.device)
        default_move = (a == b).float()

        decoder_mask = (1 - unavailable_moves.float())*(1 - no_options) + default_move*no_options
        return decoder_mask


    def compute_action_mask(self, mover_indices, unavailable_moves):

        if self.mode != 'nearest_neighbors':
            ######################################################
            d = torch.diag(torch.ones(self.num_nodes)).unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)
            ind = self.fleet.node.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)

            diag = torch.gather(d, dim=1, index=ind)
            distances = torch.gather(self.graph.distance_matrix, dim=1, index=ind)
            m = (unavailable_moves.float() + diag > 0).float()

            masked_distances = distances*(1 - m) + m*self.graph.max_dist*100
            K = min(self.num_neighbors_action, self.num_nodes)
            neighbor_indices = torch.topk(masked_distances, dim=2, k=K, largest=False)[1]

            a = torch.arange(self.num_nodes).reshape(1, 1, -1, 1).repeat(self.batch_size, self.num_cars, 1, K).to(self.device)
            b = neighbor_indices.reshape(self.batch_size, self.num_cars, 1, K).repeat(1, 1, self.num_nodes, 1)

            non_neighbor_mask = ((a == b).float().sum(dim=3) == 0)
            mask = 1 - (non_neighbor_mask | unavailable_moves).float()
            ######################################################
        else:
            mask = 1 - unavailable_moves.float()

        num_movers = mover_indices.shape[1]
        ind = mover_indices.reshape(self.batch_size, num_movers, 1).repeat(1, 1, self.num_nodes)
        action_mask = torch.gather(mask, dim=1, index=ind)
        return action_mask



    def construct_decoder_input(self, mover_indices):

        embedding_size = self.node_embeddings.shape[2]
        mask = self.check_non_depot_options(use_time=True)
        mask = mask.permute(0, 2, 1).unsqueeze(3).repeat(1, 1, 1, embedding_size).float()

        node_vectors = self.node_embeddings.reshape(self.batch_size, self.num_nodes, 1, embedding_size).repeat(1, 1, self.num_cars, 1)

        #depot vector
        start_ind = self.fleet.depot.reshape(self.batch_size, 1, self.num_cars, 1).repeat(1, 1, 1, embedding_size)
        a = torch.gather(node_vectors, dim=1, index=start_ind)
        depot_vector = a.squeeze(1)

        #current node vector
        current_ind = self.fleet.node.reshape(self.batch_size, 1, self.num_cars, 1).repeat(1, 1, 1, embedding_size)
        b = torch.gather(node_vectors, dim=1, index=current_ind)
        current_node_vector = b.squeeze(1)

        #mean graph vector
        mean_graph_vector = node_vectors.mean(dim=1)


        #other cars in field
        num_movers = mover_indices.shape[1]
        if num_movers == 1:
            movers_vector = current_node_vector*0
        else:
            a = mover_indices.reshape(self.batch_size, num_movers, 1).repeat(1, 1, self.num_cars)
            b = torch.arange(self.num_cars).reshape(1, 1, self.num_cars).repeat(self.batch_size, num_movers, 1).to(self.device)
            c = ((a == b).float().sum(dim=1) > 0).float()
            movers_mask = c.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, embedding_size)
            movers_vector = ((current_node_vector*movers_mask).sum(dim=1).unsqueeze(1) - current_node_vector)/(num_movers - 1)


        #current feature values
        feature_values = self.fleet.construct_vector()

        L = [current_node_vector, depot_vector, mean_graph_vector, movers_vector, feature_values]
        pre_output = torch.cat(L, dim=2)


        num_movers = mover_indices.shape[1]
        ind = mover_indices.reshape(self.batch_size, num_movers, 1).repeat(1, 1, pre_output.shape[2])
        output = torch.gather(pre_output, dim=1, index=ind)

        return output



    def get_mover_indices(self, unavailable_moves):
        depot = self.fleet.depot.reshape(self.batch_size, self.num_cars).long()
        current_node = self.fleet.node.reshape(self.batch_size, self.num_cars).long()

        # find all cars with "no options"
        has_option = ((1 - unavailable_moves.float()).sum(dim=2) > 0)

        at_depot = (current_node == depot)
        in_field = (current_node != depot)
        active_in_field = in_field & has_option
        active_at_depot = at_depot & has_option

        deployment_score = self.compute_depoyment_priority_score()
        max_deployment_score = deployment_score.max()

        A = max_deployment_score * 100
        B = deployment_score * 10

        score = active_in_field.float() * A + active_at_depot.float() * B
        score = score.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_depots)

        a = self.fleet.depot.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_depots)
        b = torch.arange(self.num_depots).reshape(1, 1, self.num_depots).repeat(self.batch_size, self.num_cars, 1).to(self.device)
        m = (a == b).float()

        score = score*m

        K = self.num_movers_corrected
        indices = torch.topk(score, k=K, dim=1, largest=True)[1]
        indices = indices.reshape(self.batch_size, self.num_depots*K)
        return indices


    def check_arival_times(self):
        #check for arrival times
        d = self.graph.time_matrix.unsqueeze(1).repeat(1, self.num_cars, 1, 1)
        ind = self.fleet.node.reshape(self.batch_size, self.num_cars, 1, 1).repeat(1, 1, 1, self.num_nodes)
        drive_times = torch.gather(d, dim=2, index=ind).squeeze(2)
        t = self.fleet.time.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
        arival_times = drive_times + t
        b = self.graph.end_time.reshape(self.batch_size, 1, self.num_nodes).repeat(1, self.num_cars, 1)
        attainable = (arival_times <= b)
        return attainable



    def check_volume(self):
        # check for volume
        v = self.fleet.volume.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
        a = self.graph.volume_demand.reshape(self.batch_size, 1, self.num_nodes).repeat(1, self.num_cars, 1)
        enough_volume = (v >= a)
        return enough_volume


    def check_weight(self):
        # check for weight
        w = self.fleet.weight.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
        b = self.graph.weight_demand.reshape(self.batch_size, 1, self.num_nodes).repeat(1, self.num_cars, 1)
        enough_weight = (w >= b)
        return enough_weight



    def return_to_depot(self):
        unavailable_moves = 1 - self.check_non_depot_options(use_time=False).float()
        return_to_depot = (unavailable_moves.reshape(self.batch_size, -1).sum(dim=1) == 0).float()
        return_to_depot = return_to_depot.reshape(self.batch_size, 1).repeat(1, self.num_cars)


        if return_to_depot.sum().item() > 0:

            depot = self.fleet.depot.reshape(self.batch_size, self.num_cars).long()
            node = self.fleet.node.reshape(self.batch_size, self.num_cars).long()

            #compute next node
            return_to_depot = return_to_depot.long()
            next_node = return_to_depot*depot + (1 - return_to_depot)*node


            #update time
            return_to_depot = return_to_depot.unsqueeze(2).float()

            current_node = self.fleet.node
            ind_1 = current_node.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
            drive_times = torch.gather(self.graph.time_matrix, dim=1, index=ind_1)
            ind_2 = next_node.reshape(self.batch_size, self.num_cars, 1)
            time_to_next_node = torch.gather(drive_times, dim=2, index=ind_2)
            self.fleet.time = self.fleet.time*(1 - return_to_depot) + time_to_next_node*return_to_depot

            #update node
            self.fleet.node = next_node

            #update path
            n = self.fleet.node.reshape(self.batch_size, self.num_cars, 1)
            self.fleet.path = torch.cat([self.fleet.path, n], dim=2)

            #update storage
            self.fleet.volume = self.fleet.volume * (1 - return_to_depot) + self.fleet.volume_capacity * return_to_depot
            self.fleet.weight = self.fleet.weight * (1 - return_to_depot) + self.fleet.weight_capacity * return_to_depot

            #update incompatibility
            return_to_depot = return_to_depot.reshape(self.batch_size, self.num_cars, 1).repeat(1, 1, self.num_nodes)
            self.fleet.incompatible_nodes = \
                self.fleet.incompatible_nodes*(1 - return_to_depot) + \
                (1 - self.fleet.car_node_compatibility)*return_to_depot


    def normalize(self, normalize_position=False):

        greatest_drive_time = self.graph.time_matrix.reshape(self.batch_size, -1).max(dim=1)[0]
        greatest_distance = self.graph.distance_matrix.reshape(self.batch_size, -1).max(dim=1)[0]

        greatest_volume = self.graph.volume_demand.reshape(self.batch_size, -1).max(dim=1)[0]
        greatest_weight = self.graph.weight_demand.reshape(self.batch_size, -1).max(dim=1)[0]

        a = self.fleet.start_time.reshape(self.batch_size, -1)
        b = self.graph.start_time.reshape(self.batch_size, -1)
        earliest_start_time = torch.cat([a, b], dim=1).min(dim=1)[0]


        mean_positions = self.graph.node_positions.mean(dim=1)
        std_positions = torch.std(self.graph.node_positions, dim=1)

        d = greatest_distance.reshape(self.batch_size, 1, 1).repeat(1, self.num_nodes, self.num_nodes)
        self.graph.distance_matrix = self.graph.distance_matrix/d

        t = greatest_drive_time.reshape(self.batch_size, 1, 1).repeat(1, self.num_nodes, self.num_nodes)
        self.graph.time_matrix = self.graph.time_matrix/t

        s = earliest_start_time.reshape(self.batch_size, 1, 1).repeat(1, self.num_nodes, 1)
        t = greatest_drive_time.reshape(self.batch_size, 1, 1).repeat(1, self.num_nodes, 1)
        self.graph.start_time = (self.graph.start_time - s)/t
        self.graph.end_time = (self.graph.end_time - s)/t
        self.graph.unload_times = self.graph.unload_times/t


        v = greatest_volume.reshape(self.batch_size, 1, 1).repeat(1, self.num_nodes, 1)
        self.graph.volume_demand = self.graph.volume_demand/v

        w = greatest_weight.reshape(self.batch_size, 1, 1).repeat(1, self.num_nodes, 1)
        self.graph.weight_demand = self.graph.weight_demand/w

        v = greatest_volume.reshape(self.batch_size, 1, 1).repeat(1, self.num_cars, 1)
        self.fleet.volume_capacity = self.fleet.volume_capacity/v

        w = greatest_weight.reshape(self.batch_size, 1, 1).repeat(1, self.num_cars, 1)
        self.fleet.weight_capacity = self.fleet.weight_capacity/w


        if normalize_position:
            m = mean_positions.reshape(self.batch_size, 1, mean_positions.shape[-1]).repeat(1, self.num_nodes, 1)
            st = std_positions.reshape(self.batch_size, 1, std_positions.shape[-1]).repeat(1, self.num_nodes, 1)
            self.graph.node_positions = (self.graph.node_positions - m)/st

        self.normalization_params = {
            'greatest_distance': greatest_distance, 'greatest_drive_time': greatest_drive_time,
            'earliest_start_time': earliest_start_time, 'greatest_volume': greatest_volume,
            'greatest_weight': greatest_weight, 'mean_positions': mean_positions, 'std_positions': std_positions
        }