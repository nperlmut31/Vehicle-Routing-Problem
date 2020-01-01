

from just_time_windows.google_solver.google_model import evaluate_google_model
from just_time_windows.Actor.actor import Actor as NN_Actor
from just_time_windows.build_data import Raw_VRP_Data
from just_time_windows.dataloader import VRP_Dataset


dataset = VRP_Dataset(dataset_size=10, num_depots=1, num_nodes=12)

batch = dataset.get_batch(0, 10)

nn_actor = NN_Actor(model=None, num_movers=10, num_neighbors_action=1)


nn_output = nn_actor(batch)
time = nn_output['total_time']
arrival_times = nn_output['arrival_times']


output = evaluate_google_model(dataset)



print(arrival_times)
print(time.mean().item())
print(output.mean().item())