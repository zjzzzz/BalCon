import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Set
import json
from utils import NumpyConverter
from enum import IntEnum
from global_variables import HOST_NUMA


@dataclass
class Host:
    cpu: int
    mem: int
    numa: int


@dataclass(frozen=True)
class VM:
    cpu: int
    mem: int
    numa: int


@dataclass
class Mapping:
    mapping: List[int]
    numas: List[List[int]]

    def __init__(self, mapping: List[int], numas: List[List[int]]):
        self.mapping = mapping
        self.numas = numas

    def __getitem__(self, key: int) -> (int, List[int]):
        return self.mapping[key], self.numas[key]

    @staticmethod
    def emtpy(n_vms: int, vm_numa_num: List[int]):
        return Mapping([-1 for _ in range(n_vms)], [[-1]*vm_numa_num[i] for i in range(n_vms)])


class Resources(IntEnum):
    CPU = 0
    MEM = 1


@dataclass
class Environment:
    hosts: List[Host]
    vms: List[VM]
    mapping: Mapping

    def save(self, path: str) -> None:
        with open(path, 'w') as fd:
            json.dump({
                    'hosts': [asdict(host) for host in self.hosts],
                    'vms': [asdict(vm) for vm in self.vms],
                    'mapping': self.mapping.mapping,
                    'numas': self.mapping.numas,
            }, fd, indent=4, default=NumpyConverter)

    @staticmethod
    def load(path: str):
        with open(path, 'r') as fd:
            data = json.load(fd)

        hosts = [Host(**host) for host in data['hosts']]
        vms = [VM(**vm) for vm in data['vms']]
        mapping = Mapping(data['mapping'], data['numas'])
        return Environment(hosts, vms, mapping)

    def validate_mapping(self, mapping: Mapping) -> bool:
        n_hosts = len(self.hosts)
        rem_nh_nr = np.zeros((n_hosts, HOST_NUMA, 2))

        for hid in range(n_hosts):
            for numa_id in range(HOST_NUMA):
                rem_nh_nr[hid, numa_id, Resources.CPU] = self.hosts[hid].cpu // HOST_NUMA
                rem_nh_nr[hid, numa_id, Resources.MEM] = self.hosts[hid].mem // HOST_NUMA

        for vmid, hid in enumerate(mapping.mapping):
            vm_numas = mapping.numas[vmid]
            vm_numa_num = self.vms[vmid].numa
            if hid == -1:
                return False
            if len(vm_numas) == 0 or np.any(vm_numas == -1):
                return False
            for numa_id in vm_numas:
                rem_nh_nr[hid, numa_id, Resources.CPU] -= self.vms[vmid].cpu // vm_numa_num
                rem_nh_nr[hid, numa_id, Resources.MEM] -= self.vms[vmid].mem // vm_numa_num

        return np.all(rem_nh_nr >= 0)


@dataclass
class Settings:
    wa: float = 10.
    wm: float = 1.
    tl: float = 600.
    max_force_steps: int = 4000
