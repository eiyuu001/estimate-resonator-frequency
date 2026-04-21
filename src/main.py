import argparse
import json
import os
from dataclasses import dataclass
from typing import Any
from bare_shift_boundary_estimator import (
    BareShiftBoundary,
    BareShiftDebugOptions,
)
from config import create_bare_shift_boundary_estimator
from estimate_resonator_frequency import Resonance, estimate_resonator_frequency
from plot import output_images
from remove_false_spike import remove_false_spike


@dataclass(frozen=True)
class MainArgs:
    conf_file: str
    input_file: str
    mux: int
    image_dir: str | None
    image_prefix: str | None
    plot: bool
    debug: bool


@dataclass(frozen=True)
class OutputPaths:
    bare_shift_artifact_prefix: str | None
    spectroscopy_image_prefix: str | None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf-file', required=True)
    parser.add_argument('-f', '--input-file', required=True)
    parser.add_argument('--mux', type=int, required=True)
    parser.add_argument('--image-dir')
    parser.add_argument('--image-prefix')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    namespace = parser.parse_args()
    return MainArgs(
        conf_file=namespace.conf_file,
        input_file=namespace.input_file,
        mux=namespace.mux,
        image_dir=namespace.image_dir,
        image_prefix=namespace.image_prefix,
        plot=namespace.plot,
        debug=namespace.debug,
    )


def load_inputs(args: MainArgs) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(args.input_file) as f:
        data = json.load(f)

    with open(args.conf_file) as f:
        conf = json.load(f)

    return data, conf


def build_output_paths(
    data: dict[str, Any], image_dir: str | None, image_prefix: str | None
) -> OutputPaths:
    if image_dir is None:
        return OutputPaths(
            bare_shift_artifact_prefix=None,
            spectroscopy_image_prefix=None,
        )

    mux = data['layout']['title']['text'][-5:]

    if image_prefix is None:
        image_prefix = ''

    return OutputPaths(
        bare_shift_artifact_prefix=os.path.join(image_dir, f'{image_prefix}{mux}_2_'),
        spectroscopy_image_prefix=os.path.join(image_dir, f'{image_prefix}{mux}_'),
    )


def denoise_data(
    data: dict[str, Any],
    conf: dict[str, Any],
) -> dict[str, Any]:
    for item in conf['remove_false_spike']:
        data = remove_false_spike(data, *item)
    return data


def estimate_bare_shift_boundary(
    data: dict[str, Any],
    conf: dict[str, Any],
    *,
    bare_shift_artifact_prefix: str | None = None,
) -> BareShiftBoundary:
    estimator = create_bare_shift_boundary_estimator(conf)
    debug = BareShiftDebugOptions(
        artifact_prefix=bare_shift_artifact_prefix,
    )

    boundary = estimator.estimate_bare_shift_boundary(
        data['data'][0]['x'],
        data['data'][0]['y'],
        data['data'][0]['z'],
        debug=debug,
    )

    if boundary.high_power_min is None or boundary.high_power_max is None:
        raise ValueError('failed to estimate bare shift boundary')

    return boundary


def estimate_resonances(
    data: dict[str, Any],
    conf: dict[str, Any],
    boundary: BareShiftBoundary,
):
    return estimate_resonator_frequency(
        data['data'][0]['y'],
        data['data'][0]['z'],
        high_power_min=boundary.high_power_min,
        high_power_max=boundary.high_power_max,
        low_power=boundary.low_power,
        **conf['estimate_resonator_frequency'],
    )


def build_result(
    args: MainArgs,
    data: dict[str, Any],
    resonances: list[Resonance],
) -> list[dict[str, Any]]:
    if len(resonances) < 4:
        return [
            dict(
                mux=args.mux,
                qubit=None,
                frequency=data['data'][0]['x'][resonance.x],
            )
            for resonance in resonances
        ]

    resonances = [resonances[1], resonances[3], resonances[2], resonances[0]]
    return [
        dict(
            mux=args.mux,
            qubit=args.mux * 4 + i,
            frequency=data['data'][0]['x'][resonance.x],
        )
        for i, resonance in enumerate(resonances)
    ]


def build_debug_output(boundary: BareShiftBoundary) -> dict[str, Any]:
    return {'bare_shift_boundary': boundary.__dict__}


def print_result(
    result: list[dict[str, Any]],
    debug: bool,
    debug_output: dict[str, Any] | None = None,
):
    if debug:
        if debug_output is None:
            debug_output = {}
        print(json.dumps({'result': result, 'debug': debug_output}))
    else:
        print(json.dumps(result))


def maybe_output_spectroscopy_images(
    data: dict[str, Any],
    resonances: list[Resonance],
    rests: list[Resonance],
    image_prefix: str | None,
    plot: bool,
) -> None:
    if image_prefix or plot:
        output_images(
            data,
            resonances,
            rests,
            image_prefix,
            plot,
        )


def main():
    args = parse_args()
    data, conf = load_inputs(args)
    output_paths = build_output_paths(data, args.image_dir, args.image_prefix)
    data = denoise_data(data, conf)
    boundary = estimate_bare_shift_boundary(
        data, conf, bare_shift_artifact_prefix=output_paths.bare_shift_artifact_prefix
    )
    resonances, rests = estimate_resonances(data, conf, boundary)
    result = build_result(args, data, resonances)
    debug_output = build_debug_output(boundary)
    print_result(result, args.debug, debug_output)
    maybe_output_spectroscopy_images(
        data, resonances, rests, output_paths.spectroscopy_image_prefix, args.plot
    )


if __name__ == '__main__':
    main()
