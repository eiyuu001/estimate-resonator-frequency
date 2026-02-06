import argparse
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from remove_false_spike import remove_false_spike
from estimate_resonator_frequency import estimate_resonator_frequency


def simplify_layout(data):
    data['layout']['template']['layout']['xaxis']['tick0'] = 9.8
    data['layout']['template']['layout']['xaxis']['dtick'] = 0.1

    data['data'][0]['showscale'] = False
    data['layout']['title']['text'] = ''
    data['layout']['xaxis']['title']['text'] = ''
    data['layout']['xaxis']['showticklabels'] = False
    data['layout']['xaxis']['ticks'] = ''
    data['layout']['yaxis']['title']['text'] = ''
    data['layout']['yaxis']['ticks'] = 'outside'
    data['layout']['yaxis']['tick0'] = -40
    data['layout']['yaxis']['dtick'] = 10
    data['layout']['yaxis']['tickfont'] = {
        'size': 16,
    }
    data['layout']['margin'] = {
        'l': 40,
        'r': 10,
        't': 10,
        'b': 10,
    }
    data['layout']['height'] = 600

    return data


phases = [
    # 'simplify_layout',
    'filtered_images',
    'mark',
    'marked_images',
    'plot',
]


def process_data(conf, data, idx):
    img_dir = 'images'

    if 'simplify_layout' in phases:
        data = simplify_layout(data)

    if 'filtered_images' in phases or 'mark' in phases or 'marked_images' in phases:
        for item in conf['remove_false_spike']:
            data = remove_false_spike(data, *item)

    fig = go.Figure(**data)

    if 'filtered_images' in phases:
        output_path = os.path.join(img_dir, f'filtered_mux_{idx:02}.png')
        fig.write_image(output_path)

    if 'mark' in phases or 'marked_images' in phases:
        resonances, rests = estimate_resonator_frequency(
            data['data'][0]['y'],
            data['data'][0]['z'],
            **conf['estimate_resonator_frequency'],
        )

        for i, high_power_peaks in enumerate(
            sorted(
                [
                    res.high_power_peaks
                    for res in resonances + rests
                    if res.high_power_peaks is not None
                ],
                key=lambda pg: pg.x,
            )
        ):

            xs = []
            ys = []

            for peak in high_power_peaks.peaks:
                xs.append(data['data'][0]['x'][peak[0]])
                ys.append(data['data'][0]['y'][peak[1]])

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode='markers',
                    marker=dict(
                        color=px.colors.qualitative.Plotly[
                            i % len(px.colors.qualitative.Plotly)
                        ],
                        size=8,
                        symbol='x',
                    ),
                    showlegend=False,
                )
            )

        for resonance in resonances:
            fig.add_vline(
                x=data['data'][0]['x'][resonance.x],
                line_width=1,
                line_color='red',
                line_dash='dash',
            )

        for resonance in rests:
            fig.add_vline(
                x=data['data'][0]['x'][resonance.x],
                line_width=1,
                line_color='orange',
                line_dash='dash',
            )

    if 'marked_images' in phases:
        output_path = os.path.join(img_dir, f'marked_mux_{idx:02}.png')
        fig.write_image(output_path)

    if 'plot' in phases:
        fig.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf-file', required=True)
    parser.add_argument('-f', '--input-file', required=True)
    parser.add_argument('--mux', type=int, required=True)
    args = parser.parse_args()

    with open(args.conf_file) as f:
        conf = json.load(f)

    with open(args.input_file) as f:
        data = json.load(f)

    process_data(conf, data, args.mux)


if __name__ == '__main__':
    main()
