import argparse
import functools
import json
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from models import ResonatorSpectroscopyFile
from multiprocessing import Pool
from typing import Any


MAIN_SCRIPT = os.path.join(
    pathlib.Path(__file__).parent.parent.parent,
    'src/main.py',
)


@dataclass(frozen=True)
class MainArgs:
    batch_file: str
    conf_file: str
    dst_dir: str
    num_pool: int
    main_script: str


@dataclass(frozen=True)
class Config:
    conf: dict[str, Any]

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            conf = json.load(f)
        return cls(conf)

    def get_conf(self, qubit: str):
        return self.conf['common'] | self.conf[qubit]


@dataclass(frozen=True)
class DstDir:
    work_dir: str
    image_dir: str
    path_batch: str
    path_conf_batch: str
    path_conf_64q: str
    path_conf_144q: str
    path_result: str

    def get_conf_path(self, qubit: str):
        if qubit == '64':
            return self.path_conf_64q
        elif qubit == '144':
            return self.path_conf_144q
        else:
            raise ValueError(f'Unknown qubit {qubit}.')

    @classmethod
    def from_dir_path(cls, dir_path: str):
        work_dir = os.path.join(
            dir_path,
            datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'),
        )

        image_dir = os.path.join(work_dir, 'images')
        path_batch = os.path.join(work_dir, 'batch.json')
        path_conf_batch = os.path.join(work_dir, 'config_batch.json')
        path_conf_64q = os.path.join(work_dir, 'config_64q.json')
        path_conf_144q = os.path.join(work_dir, 'config_144q.json')
        path_result = os.path.join(work_dir, 'result.json')

        return cls(
            work_dir=work_dir,
            image_dir=image_dir,
            path_batch=path_batch,
            path_conf_batch=path_conf_batch,
            path_conf_64q=path_conf_64q,
            path_conf_144q=path_conf_144q,
            path_result=path_result,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-file')
    parser.add_argument('--conf-file')
    parser.add_argument('--dst-dir')
    parser.add_argument('--pool', type=int, default=4)
    parser.add_argument('--main-script', default=MAIN_SCRIPT)
    args = parser.parse_args()
    return MainArgs(
        batch_file=args.batch_file,
        conf_file=args.conf_file,
        dst_dir=args.dst_dir,
        num_pool=args.pool,
        main_script=args.main_script,
    )


def build_cases(batch_file: str):
    with open(batch_file) as f:
        batch = json.load(f)
    return [ResonatorSpectroscopyFile(**case) for case in batch]


def write_batch_context(dst_dir: DstDir, batch_file: str, conf_file: str, conf: Config):
    os.makedirs(dst_dir.work_dir, exist_ok=False)
    os.makedirs(dst_dir.image_dir, exist_ok=False)

    shutil.copy(batch_file, dst_dir.path_batch)
    shutil.copy(conf_file, dst_dir.path_conf_batch)

    with open(dst_dir.path_conf_64q, 'w') as f:
        json.dump(conf.get_conf('64'), f)

    with open(dst_dir.path_conf_144q, 'w') as f:
        json.dump(conf.get_conf('144'), f)


def run(
    main_script: str,
    dst_dir: DstDir,
    case: ResonatorSpectroscopyFile,
) -> dict[str, Any]:
    print(f'processing: {case.z_digest}')
    commands = [
        'uv',
        'run',
        main_script,
        '--conf-file',
        dst_dir.get_conf_path(case.qubit),
        '--input-file',
        case.src_path,
        '--mux',
        case.mux,
        '--image-dir',
        dst_dir.image_dir,
        '--image-prefix',
        f'{case.z_digest}_',
        '--debug',
    ]
    result: subprocess.CompletedProcess = subprocess.run(
        commands,
        capture_output=True,
    )
    if result.returncode == 0:
        return json.loads(result.stdout.decode('utf-8'))
    else:
        return {'returncode': result.returncode}


def batch(
    num_pool: int,
    main_script: str,
    dst_dir: DstDir,
    cases: list[ResonatorSpectroscopyFile],
) -> list[dict[str, Any]]:
    run_binded = functools.partial(
        run,
        main_script,
        dst_dir,
    )

    with Pool(num_pool) as p:
        results = p.map(run_binded, cases)

    return results


def build_batch_result(
    cases: list[ResonatorSpectroscopyFile],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {case.z_digest: result for case, result in zip(cases, results)}


def write_batch_result(dst_dir: DstDir, batch_result: dict[str, Any]):
    with open(dst_dir.path_result, 'w') as f:
        json.dump(batch_result, f, sort_keys=True)


def main():
    args = parse_args()

    conf = Config.from_file(args.conf_file)
    dst_dir = DstDir.from_dir_path(args.dst_dir)
    write_batch_context(dst_dir, args.batch_file, args.conf_file, conf)

    cases = build_cases(args.batch_file)
    results = batch(args.num_pool, args.main_script, dst_dir, cases)
    batch_result = build_batch_result(cases, results)
    write_batch_result(dst_dir, batch_result)


if __name__ == '__main__':
    main()
