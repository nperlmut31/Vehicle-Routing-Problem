
import os
import sys
import torch
import torch.nn as nn

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, '..')
sys.path.append(p)

from fleet_beam_search_2.nets.model_2 import Model
from fleet_beam_search_2.Actor.actor_2 import Actor
from fleet_beam_search_2.dataloader import VRP_Dataset
from fleet_beam_search_2.google_solver.google_model import GoogleActor

import torch.optim as optim
from torch.utils.data import DataLoader
import json


with open('params.json', 'r') as f:
    params = json.load(f)
    f.close()


device = params['device']
run_tests = params['run_tests']
save_results = params['save_results']

train_dataset_size = params['train_dataset_size']
test_dataset_size = params['test_dataset_size']


num_nodes = params['num_nodes']
num_depots = params['num_depots']
num_cars = params['num_cars']

embedding_size = params['embedding_size']
sample_size = params['sample_size']
beam_size = params['beam_size']
num_neighbors_encoder = params['num_neighbors_encoder']
num_neighbors_action = params['num_neighbors_action']
num_movers = params['num_movers']
learning_rate = params['learning_rate']
batch_size = params['batch_size']
test_batch_size = params['test_batch_size']
baseline_update_period = params['baseline_update_period']




train_dataset = VRP_Dataset(dataset_size=train_dataset_size, num_nodes=num_nodes,
                            num_cars=num_cars,num_depots=num_depots, device=device)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate)

#compute google's score
google_actor = GoogleActor()
data = train_dataset.get_batch(0, train_dataset_size)
google_score = google_actor(data)



input_size = train_dataset.model_input_length()
decoder_input_size = train_dataset.decoder_input_length()


model = Model(input_size=input_size, decoder_input_size=decoder_input_size, embedding_size=embedding_size)
actor = Actor(model=model, num_movers=num_movers,
              num_neighbors_encoder=num_neighbors_encoder,
              num_neighbors_action=num_neighbors_action,
              device=device, normalize=False)
actor.train_mode()


baseline_model = Model(input_size=input_size, decoder_input_size=decoder_input_size, embedding_size=embedding_size)
baseline_actor = Actor(model=baseline_model, num_movers=num_movers,
                        num_neighbors_encoder=num_neighbors_encoder, num_neighbors_action=num_neighbors_action,
                        device=device, normalize=False)
baseline_actor.greedy_search()
baseline_actor.load_state_dict(actor.state_dict())

nn_actor = Actor(model=None, num_movers=1,
              num_neighbors_encoder=num_neighbors_encoder, num_neighbors_action=1,
              device=device, normalize=False)
nn_actor.nearest_neighbors()


learning_rate = params['learning_rate']
optimizer = optim.Adam(params=actor.parameters(), lr=learning_rate)

record = 100
num_epochs = params['num_epochs']
for epoch in range(num_epochs):

    tot_actor_cost = 0
    tot_nn_cost = 0

    for i, batch in enumerate(train_dataloader):

        with torch.no_grad():
            nn_output = nn_actor(batch)
            nn_cost = nn_output['cost'].sum().item()

            actor.sample_mode(sample_size=sample_size)
            #actor.greedy_search()
            actor_output = actor(batch)
            actor_cost = actor_output['cost'].sum().item()

        tot_actor_cost += actor_cost
        tot_nn_cost += nn_cost
        nn_ratio = actor_cost/nn_cost
        google_ratio = actor_cost/google_score

        print(epoch, i, nn_ratio, google_ratio, record, flush=True)


        actor.train_mode()
        train_output = actor(batch)
        train_cost = train_output['cost']
        log_probs = train_output['log_probs']

        with torch.no_grad():
            baseline_actor.greedy_search()
            baseline_output = baseline_actor(batch)
            baseline_cost = baseline_output['cost']

        loss = ((train_cost - baseline_cost).detach()*log_probs).mean()

        optimizer.zero_grad()
        loss.backward()

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(
                group['params'],
                1,
                norm_type=2
            )

        optimizer.step()


    tot_ratio = tot_actor_cost / tot_nn_cost

    if tot_ratio < record:
        record = tot_ratio
        if epoch % baseline_update_period == 0:
            baseline_actor.load_state_dict(actor.state_dict())

