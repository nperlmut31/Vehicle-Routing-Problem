
import os
import sys
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, '..')
sys.path.append(p)

from fleet_beam_search_2.nets.model import Model
from fleet_beam_search_2.Actor.actor import Actor
from fleet_beam_search_2.dataloader import VRP_Dataset

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
num_neighbors_encoder = params['num_neighbors_encoder']
num_neighbors_action = params['num_neighbors_action']
num_movers = params['num_movers']
learning_rate = params['learning_rate']
batch_size = params['batch_size']
test_batch_size = params['test_batch_size']
baseline_update_period = params['baseline_update_period']




test_dataset = VRP_Dataset(dataset_size=test_dataset_size, num_nodes=num_nodes, num_cars=num_cars,num_depots=num_depots, device=device)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, collate_fn=test_dataset.collate)


input_size = test_dataset.model_input_length()
decoder_input_size = test_dataset.decoder_input_length()


model = Model(input_size=input_size, decoder_input_size=decoder_input_size, embedding_size=embedding_size)
actor = Actor(model=model, num_movers=num_movers,
              num_neighbors_encoder=num_neighbors_encoder,
              num_neighbors_action=num_neighbors_action,
              device=device, normalize=True)
actor.train_mode()


baseline_model = Model(input_size=input_size, decoder_input_size=decoder_input_size, embedding_size=embedding_size)
baseline_actor = Actor(model=baseline_model, num_movers=num_movers,
                        num_neighbors_encoder=num_neighbors_encoder, num_neighbors_action=num_neighbors_action,
                        device=device, normalize=True)
baseline_actor.greedy_search()
baseline_actor.load_state_dict(actor.state_dict())


nn_actor = Actor(model=None, num_movers=1,
              num_neighbors_encoder=num_neighbors_encoder, num_neighbors_action=1,
              device=device, normalize=True)
nn_actor.nearest_neighbors()


learning_rate = params['learning_rate']
optimizer = optim.Adam(params=actor.parameters(), lr=learning_rate)

greedy_record = 100
sample_record = 100


num_epochs = params['num_epochs']
for epoch in range(num_epochs):

    if (epoch % 50 == 0) and (epoch > 0):
        num_neighbors_action += 3

        actor.num_neighbors_action = num_neighbors_action
        baseline_actor.num_neighbors_action = num_neighbors_action

    train_dataset = VRP_Dataset(dataset_size=train_dataset_size, num_nodes=num_nodes, num_cars=num_cars, num_depots=num_depots, device=device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate)

    for i, batch in enumerate(test_dataloader):

        actor.train_mode()
        actor_output = actor(batch)
        actor_cost, log_probs = actor_output['cost'], actor_output['log_probs']

        baseline_actor.greedy_search()
        with torch.no_grad():
            baseline_output = baseline_actor(batch)
            baseline_cost = baseline_output['cost']

            nn_output = nn_actor(batch)
            nn_cost = nn_output['cost']

        loss = ((actor_cost - baseline_cost).detach()*log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ratio = actor_cost.mean().item()/nn_cost.mean().item()
        print(epoch, i, ratio)

    
    tot_sample_cost = 0
    tot_nn_cost = 0
    tot_greedy_cost = 0

    for i, batch in enumerate(test_dataloader):

        with torch.no_grad():
            nn_output = nn_actor(batch)
            nn_cost = nn_output['cost'].sum().item()

            actor.greedy_search()
            sample_output = actor(batch)
            sample_cost = sample_output['cost'].sum().item()

            actor.greedy_search()
            greedy_output = actor(batch)
            greedy_cost = greedy_output['cost'].sum().item()


        tot_sample_cost += sample_cost
        tot_nn_cost += nn_cost
        tot_greedy_cost += greedy_cost


    sample_ratio = tot_sample_cost / tot_nn_cost
    greedy_ratio = tot_greedy_cost/tot_nn_cost

    if greedy_ratio < greedy_record:
        greedy_record = greedy_ratio
        if epoch % baseline_update_period == 0:
            baseline_actor.load_state_dict(actor.state_dict())

    if sample_ratio < sample_record:
        sample_record = sample_ratio

    print('\n')
    print('test:')
    print(epoch, sample_ratio, greedy_ratio, flush=True)
    print('\n')






