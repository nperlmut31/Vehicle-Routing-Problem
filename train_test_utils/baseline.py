import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


def update_baseline(actor, baseline, validation_set, record_scores, batch_size=100, threshold=0.95):

    val_dataloader = DataLoader(dataset=validation_set,
                                batch_size=batch_size,
                                collate_fn=validation_set.collate)

    actor.greedy_search()
    actor.eval()

    actor_scores = []
    for batch in val_dataloader:
        with torch.no_grad():
            actor_output = actor(batch)
            actor_cost = actor_output['total_time']
            actor_cost.reshape(-1)
        actor_scores.append(actor_cost)
    actor_scores = torch.cat(actor_scores, dim=0)


    if record_scores is None:
        baseline.load_state_dict(actor.state_dict())
        record_scores = actor_scores
        return record_scores
    else:

        if actor_scores.mean().item() < record_scores.mean().item():
            print('\n', flush=True)
            print('baseline updated', flush=True)
            print('\n', flush=True)

            baseline.load_state_dict(actor.state_dict())
            record_scores = actor_scores
            return record_scores
        else:
            return record_scores
