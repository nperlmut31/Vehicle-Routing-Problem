
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


def validation(actor, validation_dataset, batch_size):

    val_dataloader = DataLoader(dataset=validation_dataset,
                                batch_size=batch_size,
                                collate_fn=validation_dataset.collate)

    scores = []
    for batch in val_dataloader:
        with torch.no_grad():
            actor_output = actor(batch)
            cost = actor_output['total_time']

            scores.append(cost.reshape(-1))

    scores = torch.cat(scores, dim=0)

    return scores
