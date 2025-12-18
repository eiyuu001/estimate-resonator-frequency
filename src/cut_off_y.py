def cut_off_y(data, y_min):
    idx = None

    for i, y in enumerate(data['data'][0]['y']):
        y_str = f'{y:.1f}'
        if y_str == y_min:
            idx = i

    if idx is None:
        raise ValueError

    data['data'][0]['y'] = data['data'][0]['y'][idx:]
    data['data'][0]['z'] = data['data'][0]['z'][idx:]

    return data
