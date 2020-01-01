
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from torch.nn.utils import clip_grad_norm_


dir_path = os.path.dirname(os.path.realpath(__file__))
q = os.path.join(dir_path, '..')
sys.path.append(q)

from nets.model import Model
from Actor.actor import Actor
from dataloader import VRP_Dataset
from google_solver.google_model import evaluate_google_model


import torch.optim as optim
from torch.utils.data import DataLoader
import json


with open('params.json', 'r') as f:
    params = json.load(f)
    f.close()


device = params['device']
run_tests = params['run_tests']
save_results = params['save_results']


now = datetime.now()
dt_string = now.strftime("%d-%m-%y %H:%M:%S")

print('current time: ' + dt_string + '\n')

#declare the state_dict path
if save_results:
    results_dir = os.path.join(dir_path, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    experiment_path = os.path.join(results_dir, dt_string)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
        f = open(experiment_path + '/train_results.txt', 'w+')
        g = open(experiment_path + '/test_results.txt', 'w+')
        f.close()
        g.close()

        path = os.path.join(experiment_path, 'params.json')
        with open(path, 'w+') as f:
            json.dump(params, f)
            f.close()

    else:
        print('Results directory ' + dt_string + ' already exits.')
        exit()

    problem_instances = os.path.join(experiment_path, 'problem_instances')
    os.mkdir(problem_instances)


train_dataset_size = params['train_dataset_size']
validation_dataset_size = params['validation_dataset_size']
baseline_dataset_size = params['baseline_dataset_size']


num_nodes = params['num_nodes']
num_depots = params['num_depots']


embedding_size = params['embedding_size']
sample_size = params['sample_size']
gradient_clipping = params['gradient_clipping']


num_neighbors_encoder = params['num_neighbors_encoder']
num_neighbors_action = params['num_neighbors_action']
num_movers = params['num_movers']
use_fleet_attention = params['use_fleet_attention']

learning_rate = params['learning_rate']
batch_size = params['batch_size']
test_batch_size = params['test_batch_size']
baseline_update_period = params['baseline_update_period']



validation_dataset = VRP_Dataset(dataset_size=validation_dataset_size, num_nodes=num_nodes, num_depots=num_depots, device=device)
baseline_dataset = VRP_Dataset(dataset_size=train_dataset_size, num_nodes=num_nodes, num_depots=num_depots, device=device)

if params['overfit_test']:
    train_dataset = VRP_Dataset(dataset_size=train_dataset_size, num_nodes=num_nodes, num_depots=num_depots, device=device)
    baseline_dataset = train_dataset
    validation_dataset = train_dataset


#evaluate google model
google_scores = evaluate_google_model(validation_dataset)
tot_google_scores = google_scores.sum().item()

validation_dataset.device = device



input_size = validation_dataset.model_input_length()

model = Model(input_size=input_size, embedding_size=embedding_size)
actor = Actor(model=model, num_movers=num_movers,
              num_neighbors_encoder=num_neighbors_encoder,
              num_neighbors_action=num_neighbors_action,
              device=device, normalize=False)
actor.train_mode()


baseline_model = Model(input_size=input_size, embedding_size=embedding_size)
baseline_actor = Actor(model=baseline_model, num_movers=num_movers,
                        num_neighbors_encoder=num_neighbors_encoder,
                       num_neighbors_action=num_neighbors_action,
                        device=device, normalize=False)
baseline_actor.greedy_search()
baseline_actor.load_state_dict(actor.state_dict())


nn_actor = Actor(model=None, num_movers=1, num_neighbors_action=1, device=device)
nn_actor.nearest_neighbors()


optimizer = optim.Adam(params=actor.parameters(), lr=params['learning_rate'])


train_batch_record = 100
validation_record = 100
baseline_record = None

num_epochs = params['num_epochs']
for epoch in range(num_epochs):

    if not params['overfit_test']:
        train_dataset = VRP_Dataset(dataset_size=train_dataset_size, num_nodes=num_nodes, num_depots=num_depots, device=device)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate)
    for i, batch in enumerate(train_dataloader):

        with torch.no_grad():
            nn_actor.nearest_neighbors()
            nn_output = nn_actor(batch)
            tot_nn_cost = nn_output['total_time'].sum().item()

            baseline_actor.greedy_search()
            baseline_cost = baseline_actor(batch)['total_time']

        actor.train_mode()
        actor_output = actor(batch)
        actor_cost, log_probs = actor_output['total_time'], actor_output['log_probs']

        loss = ((actor_cost - baseline_cost).detach() * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()

        if gradient_clipping:
            for group in optimizer.param_groups:
                clip_grad_norm_(
                    group['params'],
                    1,
                    norm_type=2
                )

        optimizer.step()

        tot_actor_cost = actor_cost.sum().item()
        tot_baseline_cost = baseline_cost.sum().item()

        actor_nn_ratio = tot_actor_cost / tot_nn_cost
        actor_baseline_ratio = tot_actor_cost / tot_baseline_cost

        if actor_nn_ratio < train_batch_record:
            train_batch_record = actor_nn_ratio

        result = '% d, %d, %f, %f, %f' % (epoch, i, actor_nn_ratio, actor_baseline_ratio, train_batch_record)
        print(result, flush=True)

        if save_results:
            path = os.path.join(experiment_path, 'train_results.txt')
            with open(path, 'a') as f:
                f.write(result + '\n')
                f.close()

        del batch



    if epoch % 5 == 0:
        baseline_dataloader = DataLoader(baseline_dataset, batch_size=batch_size, collate_fn=baseline_dataset.collate)

        tot_cost = []
        counter = 0

        for batch in baseline_dataloader:

            with torch.no_grad():
                actor.greedy_search()
                actor_output = actor(batch)
                cost = actor_output['total_time']

            tot_cost.append(cost)
            counter += cost.shape[0]

        del batch

        tot_cost = torch.cat(tot_cost, dim=0)
        if baseline_record is None:
            baseline_record = tot_cost
        else:
            p = (tot_cost < baseline_record).float().mean().item()
            if p > 0.9:
                baseline_record = tot_cost
                baseline_actor.load_state_dict(actor.state_dict())
                print('\n')
                print('new baseline record')
                print('\n')



    if (epoch % 10 == 0) and run_tests:
        b = max(int(batch_size // sample_size**2), 1)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=b, collate_fn=validation_dataset.collate)

        tot_cost = 0
        tot_nn_cost = 0
        counter = 0
        for batch in validation_dataloader:
            with torch.no_grad():
                actor.beam_search(sample_size)
                actor_output = actor(batch)
                cost = actor_output['total_time']

                nn_actor.nearest_neighbors()
                nn_output = nn_actor(batch)
                nn_cost = nn_output['total_time']

            tot_cost += cost.sum().item()
            tot_nn_cost += nn_cost.sum().item()
            counter += cost.shape[0]

        ratio = tot_cost / tot_nn_cost
        if ratio < validation_record:
            validation_record = ratio

            if save_results:
                path = os.path.join(experiment_path, 'model_state_dict.pt')
                with open(path, 'wb') as f:
                    torch.save(actor.state_dict(), f)
                    f.close()

                path = os.path.join(experiment_path, 'optimizer_state_dict.pt')
                with open(path, 'wb') as f:
                    torch.save(optimizer.state_dict(), f)
                    f.close()

        actor_google_ratio = tot_cost/tot_google_scores


        print('\n', flush=True)
        print('test results \n', flush=True)
        print('actor/google ratio: %f, actor/nn ratio: %f, actor/nn ratio record: %f' % (actor_google_ratio, ratio, validation_record), flush=True)
        print('\n', flush=True)

        result = '% d, % f, % f, %f' % (epoch, actor_google_ratio, ratio, validation_record)
        if save_results:
            path = os.path.join(experiment_path, 'test_results.txt')
            with open(path, 'a') as f:
                f.write(result + '\n')
                f.close()

        del batch

