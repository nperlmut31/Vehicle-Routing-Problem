from build_data import Raw_VRP_Data
from dataloader import VRP_Dataset
from Actor.actor import Actor
from nets.model import Model
from google_solver.google_model import GoogleActor


dataset = VRP_Dataset(dataset_size=100, num_nodes=10, num_depots=1)
batch = dataset.get_batch(0, 1)


input_size = dataset.model_input_length()


model = Model(input_size=input_size, embedding_size=100)

actor = Actor(model=model, num_neighbors_encoder=30,
              num_neighbors_action=5, num_movers=3, normalize=True, use_fleet_attention=True)


actor.beam_search(5)
output = actor(batch)
distance = output['total_time'].sum().item()

print(output)