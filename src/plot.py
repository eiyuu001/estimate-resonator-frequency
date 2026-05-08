import itertools
import plotly.graph_objects as go
import plotly.express as px


def mark(
    data,
    resonances,
    rests,
    local_boundaries,
    minimum_usable_power,
    fig,
    debug,
):
    x_diff = data['data'][0]['x'][1] - data['data'][0]['x'][0]
    items = itertools.zip_longest(resonances + rests, local_boundaries)
    for i, (i_ord, (resonance, local_boundary)) in enumerate(
        sorted(enumerate(items), key=lambda item: item[1][0].x)
    ):
        if resonance.high_power_peaks:
            xs = []
            ys = []
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

        if debug or i_ord < len(resonances):
            if resonance.complementary_peaks:
                xs = []
                ys = []
                for peak in resonance.complementary_peaks:
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
                            symbol='diamond',
                        ),
                        showlegend=False,
                    )
                )

            if local_boundary:
                fig.add_trace(
                    go.Scatter(
                        x=[data['data'][0]['x'][resonance.x] + x_diff * 8],
                        y=[local_boundary.low_power],
                        mode='markers',
                        marker=dict(
                            color=px.colors.qualitative.Plotly[
                                i % len(px.colors.qualitative.Plotly)
                            ],
                            size=6,
                            symbol='triangle-left',
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

    fig.add_hline(
        y=minimum_usable_power,
        line_width=1,
        line_color='yellow',
        line_dash='dot',
    )
    return fig


def output_images(
    data,
    resonances,
    rests,
    local_boundaries,
    minimum_usable_power,
    image_path_prefix,
    plot,
    debug,
):
    fig = go.Figure(**data)

    if image_path_prefix:
        output_path = image_path_prefix + '0_filtered.png'
        fig.write_image(output_path)

    fig = mark(
        data,
        resonances,
        rests,
        local_boundaries,
        minimum_usable_power,
        fig,
        debug,
    )

    if image_path_prefix:
        output_path = image_path_prefix + '1_marked.png'
        fig.write_image(output_path)

    if plot:
        fig.show()
