def remove_false_spike(data, x_min, x_max):
    if float(x_min) > float(x_max):
        raise ValueError

    idx_min = None
    idx_max = None

    for i, x in enumerate(data['data'][0]['x']):
        x_str = f'{x:.3f}'
        if x_str == x_min:
            idx_min = i
        if x_str == x_max:
            idx_max = i
        if idx_min and idx_max:
            break

    if idx_min is None or idx_max is None:
        raise ValueError

    if idx_min == 0:
        raise ValueError

    if idx_max == len(data['data'][0]['x']) - 1:
        raise ValueError

    for z in data['data'][0]['z']:
        mean = (z[idx_min - 1] + z[idx_max + 1]) / 2.0
        for idx in range(idx_min, idx_max + 1):
            z[idx] = mean

    return data
