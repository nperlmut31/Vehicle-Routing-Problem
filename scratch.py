from just_time_windows.build_data import Raw_VRP_Data
from just_time_windows.dataloader import VRP_Dataset
from just_time_windows.Actor.actor import Actor
from just_time_windows.nets.model import Model
from just_time_windows.google_solver.google_model import GoogleActor


dataset = VRP_Dataset(dataset_size=100, num_nodes=10, num_depots=1)
batch = dataset.get_batch(0, 1)


input_size = dataset.model_input_length()


model = Model(input_size=input_size, embedding_size=100)

actor = Actor(model=model, num_neighbors_encoder=30,
              num_neighbors_action=5, num_movers=3, normalize=True)


actor.beam_search(5)
output = actor(batch)
distance = output['total_time'].sum().item()

print(output)