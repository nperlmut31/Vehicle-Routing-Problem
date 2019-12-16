
from fleet_beam_search_2.google_solver.google_model import GoogleActor as Actor
from fleet_beam_search_2.dataloader import VRP_Dataset


dataset = VRP_Dataset(dataset_size=10, num_nodes=10, num_cars=10, num_depots=1)

batch = dataset.get_batch(0, 10)


actor = Actor(scale_factor=100)
output = actor(batch)

print(output)