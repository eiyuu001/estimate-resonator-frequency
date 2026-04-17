import argparse
import json
import os
from remove_false_spike import remove_false_spike
from estimate_resonator_frequency import estimate_resonator_frequency
from config import create_bare_shift_boundary_estimator
from plot import output_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf-file', required=True)
    parser.add_argument('-f', '--input-file', required=True)
    parser.add_argument('--mux', type=int, required=True)
    parser.add_argument('--image-dir')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    with open(args.conf_file) as f:
        conf = json.load(f)

    with open(args.input_file) as f:
        data = json.load(f)

    for item in conf['remove_false_spike']:
        data = remove_false_spike(data, *item)

    if args.image_dir is not None:
        mux = data['layout']['title']['text'][-5:]
        image_path_prefix_fft = os.path.join(args.image_dir, f'{mux}_2_')
        image_path_prefix_spectroscopy = os.path.join(args.image_dir, f'{mux}_')
    else:
        image_path_prefix_fft = None
        image_path_prefix_spectroscopy = None

    bare_shift_boundary_estimator = create_bare_shift_boundary_estimator(
        conf, image_path_prefix_fft
    )

    low_power, high_power_min, high_power_max = (
        bare_shift_boundary_estimator.estimate_bare_shift_boundary(
            data['data'][0]['x'],
            data['data'][0]['y'],
            data['data'][0]['z'],
        )
    )

    if high_power_min is None or high_power_max is None:
        raise ValueError

    resonances, rests = estimate_resonator_frequency(
        data['data'][0]['y'],
        data['data'][0]['z'],
        high_power_min=high_power_min,
        high_power_max=high_power_max,
        low_power=low_power,
        **conf['estimate_resonator_frequency'],
    )

    if len(resonances) < 4:
        result = [
            dict(
                mux=args.mux,
                qubit=None,
                frequency=data['data'][0]['x'][resonance.x],
            )
            for resonance in resonances
        ]
    else:
        resonances = [resonances[1], resonances[3], resonances[2], resonances[0]]
        result = [
            dict(
                mux=args.mux,
                qubit=args.mux * 4 + i,
                frequency=data['data'][0]['x'][resonance.x],
            )
            for i, resonance in enumerate(resonances)
        ]

    if args.debug:
        print(
            json.dumps(
                {
                    'result': result,
                    'low_power': low_power,
                    'high_power_min': high_power_min,
                    'high_power_max': high_power_max,
                }
            )
        )
    else:
        print(json.dumps(result))

    if image_path_prefix_spectroscopy or args.plot:
        output_images(
            data, resonances, rests, image_path_prefix_spectroscopy, args.plot
        )


if __name__ == '__main__':
    main()
