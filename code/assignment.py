import numpy as np
from environment import Settings, Environment, Mapping, Resources
from global_variables import *
from typing import List


class Assignment:
    """
    Class for working with feasible solutions, but some VM can be unassigned.
    Basic operations: INCLUDE VM on a host, EXCLUDE VM
    Useful values kept precomputed as numpy arrays.
    """
    def __init__(self, env: Environment, settings: Settings) -> None:
        self.env = env
        self.settings = settings

        self.n_vms = len(env.vms)
        self.n_hosts = len(env.hosts)

        self.vmids = np.arange(self.n_vms)
        self.hids = np.arange(self.n_hosts)

        self.init_mapping = np.array(self.env.mapping.mapping)
        self.mapping = np.array(self.env.mapping.mapping)

        self.init_numas = self.env.mapping.numas
        self.numas = self.env.mapping.numas

        # vm需求
        self.required_nv_nr = np.array([[v.cpu // v.numa, v.mem // v.numa] for v in env.vms])
        self.required_nv_numa = np.array([v.numa for v in env.vms])[:, np.newaxis]
        self.required_nv_nr_all = np.array([[v.cpu, v.mem] for v in env.vms])
        # host容量
        self.capacity_nh_nr = np.array([[[h.cpu // HOST_NUMA, h.mem // HOST_NUMA] for _ in range(HOST_NUMA)] for h in env.hosts])
        # host已使用
        self.occupied_nh_nr = np.zeros((self.n_hosts, HOST_NUMA, 2))

        required_nv_nr_numa = np.array([[[v.cpu // v.numa, v.mem // v.numa] if numa_id in env.mapping.numas[i] else [0, 0]
                                         for numa_id in range(HOST_NUMA)] for i, v in enumerate(env.vms)])
        np.add.at(self.occupied_nh_nr, self.mapping, required_nv_nr_numa)
        # host剩余
        self.remained_nh_nr = self.capacity_nh_nr - self.occupied_nh_nr

        self.backup_mapping = None
        self.backup_numas = None
        self.backup_remained_nh_nr = None
        self.backup_occupied_nh_nr = None

        total_nr = np.sum(self.required_nv_nr_all, axis=0)
        # 每个vm的得分 cpu/总cpu + mem/总mem
        self.size_nv = np.sum(self.required_nv_nr_all / total_nr, axis=1)

    def is_assigned(self, vmid: int) -> bool:
        return self.mapping[vmid] != -1 and self.numas[vmid] != [-1] * self.env.vms[vmid].numa

    def exclude(self, vmid: int) -> None:
        assert self.is_assigned(vmid)
        hid = self.mapping[vmid]
        vm_numas = self.numas[vmid].copy()
        self.mapping[vmid] = -1
        self.numas[vmid] = [-1] * self.env.vms[vmid].numa
        self.occupied_nh_nr[hid][vm_numas] -= self.required_nv_nr[vmid]
        self.remained_nh_nr[hid][vm_numas] += self.required_nv_nr[vmid]

    def clear(self) -> None:
        for vmid in range(self.n_vms):
            if self.is_assigned(vmid):
                self.exclude(vmid)

    def include(self, vmid: int, hid: int, vm_numas: List[int] = []) -> None:
        assert (self.mapping[vmid] == -1)
        if not self.is_feasible(vmid,hid):
            a = 5
        assert self.is_feasible(vmid, hid)
        self.mapping[vmid] = hid
        if not vm_numas:
            vm_numas = []
        vm_numas = vm_numas.copy()
        if len(vm_numas) == 0:
            self.numas[vmid] = self.default_include_numa(vmid, hid)
        else:
            self.numas[vmid] = vm_numas
        self.occupied_nh_nr[hid][self.numas[vmid]] += self.required_nv_nr[vmid]
        self.remained_nh_nr[hid][self.numas[vmid]] -= self.required_nv_nr[vmid]

        a = 5

    def default_include_numa(self, vmid: int, hid: int) -> List[int]:
        assert self.is_feasible(vmid, hid)
        numas = []
        for numa_id in np.flatnonzero(self.is_feasible_num(vmid, hid)):
            if len(numas) == self.required_nv_numa[vmid]:
                break
            else:
                numas.append(numa_id)
        return numas

    def backup(self) -> None:
        self.backup_mapping = self.mapping.copy()
        self.backup_numas = self.numas.copy()
        self.backup_occupied_nh_nr = self.occupied_nh_nr.copy()
        self.backup_remained_nh_nr = self.remained_nh_nr.copy()

    def restore(self) -> None:
        self.mapping = self.backup_mapping.copy()
        self.numas = self.backup_numas.copy()
        self.occupied_nh_nr = self.backup_occupied_nh_nr.copy()
        self.remained_nh_nr = self.backup_remained_nh_nr.copy()

    def compute_score(self) -> float:
        # 开机的宿主机数目
        A = len(np.unique(self.mapping[self.mapping != -1]))
        # 发生迁移的虚机
        moved_vm_mask = self.init_mapping != self.mapping
        M = np.sum(self.required_nv_nr_all[moved_vm_mask, Resources.MEM])
        return A * self.settings.wa + (M / 1024 / 1024) * self.settings.wm

    def get_vmids_on_hid(self, hid) -> np.array:
        return np.flatnonzero(self.mapping == hid)

    def is_feasible_num(self, vmid: int, hid: int) -> np.array:
        return np.all(self.remained_nh_nr[hid] >= self.required_nv_nr[vmid], axis=1)

    def is_feasible(self, vmid: int, hid:int) -> bool:
        return np.sum(self.is_feasible_num(vmid, hid)) >= self.required_nv_numa[vmid]

    def is_feasible_nh(self, vmid: int) -> np.array:
        # 至少有一台宿主机资源可以覆盖虚机
        return np.sum(np.all(self.remained_nh_nr >= self.required_nv_nr[vmid], axis=2), axis=1) >= self.required_nv_numa[vmid]

    def get_solution(self) -> Mapping:
        return Mapping(list(self.mapping), list(self.numas))

    def get_active_hids(self) -> np.array:
        return np.flatnonzero(self.occupied_nh_nr[:, :, Resources.MEM] > 0)
