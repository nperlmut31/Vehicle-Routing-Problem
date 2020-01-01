import torch


def widen_tensor(datum, factor):

    if len(datum.shape) == 0:
        return datum

    L = list(datum.shape)
    a = [1, factor] + [1 for _ in range(len(L) - 1)]

    datum = datum.unsqueeze(1)
    datum = datum.repeat(*a)

    if len(L) > 1:
        b = [L[0] * factor] + L[1:]
    else:
        b = [L[0] * factor]

    datum = datum.reshape(*b)
    return datum


def widen_data(actor, include_embeddings=True, include_projections=True):

    F = dir(actor.fleet)
    for s in F:
        x = getattr(actor.fleet, s)
        if isinstance(x, torch.Tensor):
            if len(x.shape) > 0:
                y = widen_tensor(x, factor=actor.sample_size)
                setattr(actor.fleet, s, y)

    G = dir(actor.graph)
    for s in G:
        x = getattr(actor.graph, s)
        if isinstance(x, torch.Tensor):
            if len(x.shape) > 0:
                y = widen_tensor(x, factor=actor.sample_size)
                setattr(actor.graph, s, y)

    actor.log_probs = widen_tensor(actor.log_probs, factor=actor.sample_size)

    if include_embeddings:
        actor.node_embeddings = widen_tensor(actor.node_embeddings, factor=actor.sample_size)

    if include_projections:
        def widen_projection(x, size):
            if len(x.shape) > 3:
                y = x.unsqueeze(2).repeat(1, 1, size, 1, 1)
                return y.reshape(x.shape[0], x.shape[1] * size, x.shape[2], x.shape[3])
            else:
                return widen_tensor(x, size)

        actor.node_projections = {key : widen_projection(actor.node_projections[key], actor.sample_size)
                                 for key in actor.node_projections}



def select_data(self, index, include_embeddings=True, include_projections=True):
    m = index.max().item()

    F = dir(self.fleet)
    for s in F:
        x = getattr(self.fleet, s)
        if isinstance(x, torch.Tensor):
            if (len(x.shape) > 0) and (x.shape[0] >= m):
                setattr(self.fleet, s, x[index])

    G = dir(self.graph)
    for s in G:
        x = getattr(self.graph, s)
        if isinstance(x, torch.Tensor):
            if (len(x.shape) > 0) and (x.shape[0] >= m):
                setattr(self.graph, s, x[index])

    self.log_probs = self.log_probs[index]

    if include_embeddings:
        self.node_embeddings = self.node_embeddings[index]

    if include_projections:
        def select_projection(x, index):
            if len(x.shape) > 3:
                return x[:,index,:,:]
            else:
                return x[index]

        self.node_projections = {key : select_projection(self.node_projections[key], index)
                                 for key in self.node_projections}

