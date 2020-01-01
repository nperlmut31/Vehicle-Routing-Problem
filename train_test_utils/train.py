import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from just_time_windows.Actor.actor import Actor




def train_batch(actor, baseline, batch, optimizer, gradient_clipping=True, comparison_model=None, compute_cost_ratio=True):

    device = actor.device

    actor.train_mode()
    actor.train()
    actor_output = actor(batch)
    actor_cost, log_probs = actor_output['total_time'], actor_output['log_probs']


    with torch.no_grad():
        baseline.greedy_search()
        baseline_output = baseline(batch)
        baseline_cost = baseline_output['total_time']

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

    if compute_cost_ratio and (comparison_model is None):
        normalize = actor.apply_normalization
        comparison_model = Actor(model=None, num_neighbors_action=1, normalize=normalize, device=device)

    if compute_cost_ratio:
        with torch.no_grad():
            comp_output = comparison_model(batch)
            comp_cost = comp_output['total_time']

        a = comp_cost.sum().item()
        b = actor_cost.sum().item()
        return b / a
    else:
        return None

