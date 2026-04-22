import plotly.graph_objects as go
import plotly.express as px


def mark(data, resonances, rests, fig):
    for i, resonance in enumerate(sorted(resonances + rests, key=lambda res: res.x)):
        xs = []
        ys = []

        if resonance.high_power_peaks:
            for peak in resonance.high_power_peaks.peaks:
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

        if resonance.low_power_peak:
            fig.add_trace(
                go.Scatter(
                    x=[data['data'][0]['x'][resonance.low_power_peak.x]],
                    y=[data['data'][0]['y'][resonance.low_power_peak.y]],
                    mode='markers',
                    marker=dict(
                        color=px.colors.qualitative.Plotly[
                            i % len(px.colors.qualitative.Plotly)
                        ],
                        size=8,
                        symbol='circle',
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

    return fig


def output_images(data, resonances, rests, image_path_prefix, plot):
    fig = go.Figure(**data)

    if image_path_prefix:
        output_path = image_path_prefix + '0_filtered.png'
        fig.write_image(output_path)

    fig = mark(data, resonances, rests, fig)

    if image_path_prefix:
        output_path = image_path_prefix + '1_marked.png'
        fig.write_image(output_path)

    if plot:
        fig.show()
