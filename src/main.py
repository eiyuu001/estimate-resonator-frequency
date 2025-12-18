import argparse
import json
from remove_false_spike import remove_false_spike
from estimate_resonator_frequency import estimate_resonator_frequency


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

    for item in conf['remove_false_spike']:
        data = remove_false_spike(data, *item)

    resonances, _ = estimate_resonator_frequency(
        data['data'][0]['y'],
        data['data'][0]['z'],
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

    print(json.dumps(result))


if __name__ == '__main__':
    main()
