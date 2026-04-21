import argparse
import glob
import json
import operator
import os
import pathlib
from dataclasses import dataclass
from models import ResonatorSpectroscopyFile


@dataclass(frozen=True)
class MainArgs:
    src_dir: str


def to_abs_path(path: str) -> str:
    return str(pathlib.Path(path).resolve())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', type=to_abs_path)
    args = parser.parse_args()
    return MainArgs(
        src_dir=args.src_dir,
    )


def create_unique_file_set(src_dir: str) -> list[ResonatorSpectroscopyFile]:
    file_objects = map(
        ResonatorSpectroscopyFile.from_file_path,
        glob.iglob(os.path.join(src_dir, '**/*CheckResonatorSpectroscopy*_0.json')),
    )
    return list({file_obj.z_digest: file_obj for file_obj in file_objects}.values())


def print_result(file_objects: list[ResonatorSpectroscopyFile]):
    sorted_files = sorted(file_objects, key=operator.attrgetter('sort_key'))
    print(json.dumps([file_obj.__dict__ for file_obj in sorted_files]))


def main():
    args = parse_args()
    unique_file_set = create_unique_file_set(args.src_dir)
    print_result(unique_file_set)


if __name__ == '__main__':
    main()
