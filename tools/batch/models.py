import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResonatorSpectroscopyFile:
    qubit: str
    version: str
    date: str
    mux: str
    z_digest: str
    src_path: str

    @classmethod
    def from_file_path(cls, path: str):
        items = path.split('/')
        if len(items) < 2:
            raise ValueError(f'invalid file path: {path}')
        qubit, version, date = ResonatorSpectroscopyFile.parse_dir_name(items[-2])
        data = ResonatorSpectroscopyFile.load_data(path)
        mux = ResonatorSpectroscopyFile.extract_mux(data)
        z_digest = ResonatorSpectroscopyFile.compute_z_digest(data)

        return cls(
            qubit=qubit,
            version=version,
            date=date,
            mux=mux,
            z_digest=z_digest,
            src_path=path,
        )

    @property
    def sort_key(self):
        return (int(self.qubit), int(self.version), self.date, int(self.mux))

    @staticmethod
    def parse_dir_name(dir_name: str):
        items = dir_name.split('_')
        if len(items) < 3:
            raise ValueError(f'invalid dir name: {dir_name}')
        qubit, version = ResonatorSpectroscopyFile.parse_dir_name_0(items[0])
        date = ResonatorSpectroscopyFile.parse_dir_name_2(items[2])
        return qubit, version, date

    @staticmethod
    def parse_dir_name_0(dir_name_0: str) -> tuple[str, str]:
        if match := re.match(r'^([0-9]+)Qv([0-9]+)$', dir_name_0):
            return match[1], match[2]
        else:
            raise ValueError(f'dir_name_0 unmatched: {dir_name_0}')

    @staticmethod
    def parse_dir_name_2(dir_name_2: str) -> str:
        if match := re.match(r'^[0-9]{8}$', dir_name_2):
            return match[0]
        else:
            raise ValueError(f'dir_name_2 unmatched: {dir_name_2}')

    @staticmethod
    def load_data(path: str) -> dict[str, Any]:
        with open(path) as f:
            data = json.load(f)
        return data

    @staticmethod
    def extract_mux(data: dict[str, Any]) -> str:
        title = data['layout']['title']['text']
        if match := re.match(r'.*MUX([0-9]+)$', title):
            return match[1]
        else:
            raise ValueError(f'could not extract mux: {title}')

    @staticmethod
    def compute_z_digest(data: dict[str, Any]) -> str:
        return hashlib.blake2b(
            json.dumps(
                data['data'][0]['z'], separators=(',', ':'), sort_keys=False
            ).encode('utf-8'),
            digest_size=16,
        ).hexdigest()
