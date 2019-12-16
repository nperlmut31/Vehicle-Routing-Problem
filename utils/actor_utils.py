
def widen_batch(data, widen_factor=10):

    if isinstance(data, dict):
        D = {}
        for key in data:
            x = data[key]
            s = [widen_factor] + [1 for i in range(len(x.shape))]
            y = x.unsqueeze(0).repeat(*s)
            z = y.reshape(widen_factor*x.shape[0], *x.shape[1:])
            D[key] = z
        return D
    else:
        x = data
        s = [widen_factor] + [1 for i in range(len(x.shape))]
        y = x.unsqueeze(0).repeat(*s)
        z = y.reshape(widen_factor * x.shape[0], *x.shape[1:])
        return z



def widen(datum, factor):

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


def widen_dict(data_dict, factor):
    D = {}
    for key in data_dict:
        D[key] = widen(data_dict[key], factor)
    return D

