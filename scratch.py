from fleet_beam_search_2.dataloader import VRP_Dataset
from fleet_beam_search_2.Actor.actor import Actor
from fleet_beam_search_2.nets.model import Model
from fleet_beam_search_2.google_solver.google_model import GoogleActor


dataset = VRP_Dataset(dataset_size=50, num_nodes=10, num_cars=10, num_depots=1)
batch = dataset.get_batch(0, 10)


input_size = dataset.model_input_length()
decoder_input_size = dataset.decoder_input_length()

model = Model(input_size=input_size, decoder_input_size=decoder_input_size, embedding_size=100)

actor = Actor(model=model, num_neighbors_encoder=30,
              num_neighbors_action=5, num_movers=1)


output = actor(batch)
distance = output['cost'].sum().item()


google_actor = GoogleActor()
google_dist = google_actor(batch)


print(distance)
print(google_dist)