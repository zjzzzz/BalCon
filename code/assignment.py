import numpy as np
from environment import Settings, Environment, Mapping, Resources, VM
from global_variables import *
from typing import List
import matplotlib.pyplot as plt
import copy


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

        self.init_numas = self.env.mapping.numas.copy()
        self.numas = self.env.mapping.numas.copy()

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
        self.occupied_nh_nr_true = self.occupied_nh_nr.copy()

        # host剩余
        self.remained_nh_nr = self.capacity_nh_nr - self.occupied_nh_nr

        # self.required_nv_nr_flavor = None
        # self.required_nv_nr_flavor_all = None
        # self.flavor = None
        # self.required_nh_nr = None

        self.backup_mapping = None
        self.backup_numas = None
        self.backup_remained_nh_nr = None
        self.backup_occupied_nh_nr = None
        self.backup_occupied_nh_nr_true = None
        self.backup_required_nh_nr = None

        total_nr = np.sum(self.required_nv_nr_all, axis=0)
        # 每个vm的得分 cpu/总cpu + mem/总mem
        self.size_nv = np.sum(self.required_nv_nr_all / total_nr, axis=1)

    def cal_required_nh(self, flavor: VM) -> np.array:
        required_nv_nr_flavor = np.array([flavor.cpu // flavor.numa, flavor.mem // flavor.numa])
        required_nh_nr = required_nv_nr_flavor - self.remained_nh_nr
        required_nh_nr[required_nh_nr < 0] = 0
        return required_nh_nr

    def is_assigned(self, vmid: int) -> bool:
        return self.mapping[vmid] != -1 and np.all(self.numas[vmid] != -1)

    def exclude(self, vmid: int, targetVM: VM = None) -> None:
        assert self.is_assigned(vmid)
        hid = self.mapping[vmid]
        vm_numas = self.numas[vmid].copy()
        self.mapping[vmid] = -1
        self.numas[vmid] = [-1] * self.env.vms[vmid].numa
        self.occupied_nh_nr[hid][vm_numas] -= self.required_nv_nr[vmid]
        self.occupied_nh_nr_true[hid][vm_numas] -= self.required_nv_nr[vmid]
        self.remained_nh_nr[hid][vm_numas] += self.required_nv_nr[vmid]
        # if targetVM is not None:
        #     required_nv_nr_target = np.array([targetVM.cpu // targetVM.numa, targetVM.mem // targetVM.numa])
        #     self.required_nh_nr[hid][vm_numas] = required_nv_nr_target - self.remained_nh_nr[hid][vm_numas]
        #     self.required_nh_nr[self.required_nh_nr < 0] = 0

    def clear(self) -> None:
        for vmid in range(self.n_vms):
            if self.is_assigned(vmid):
                self.exclude(vmid)

    def include(self, vmid: int, hid: int, vm_numas: List[int] = None, targetVM: VM = None) -> None:
        assert (self.mapping[vmid] == -1)
        assert self.is_feasible(vmid, hid)
        self.mapping[vmid] = hid
        if vm_numas is None:
            vm_numas = []
        vm_numas = vm_numas.copy()
        if len(vm_numas) == 0:
            self.numas[vmid] = self.default_include_numa(vmid, hid)
        else:
            self.numas[vmid] = vm_numas
        self.occupied_nh_nr[hid][self.numas[vmid]] += self.required_nv_nr[vmid]
        self.occupied_nh_nr_true[hid][self.numas[vmid]] += self.required_nv_nr[vmid]
        self.remained_nh_nr[hid][self.numas[vmid]] -= self.required_nv_nr[vmid]
        # if targetVM is not None:
        #     required_nv_nr_target = np.array([targetVM.cpu // targetVM.numa, targetVM.mem // targetVM.numa])
        #     self.required_nh_nr[hid][self.numas[vmid]] = required_nv_nr_target - self.remained_nh_nr[hid][self.numas[vmid]]
        #     self.required_nh_nr[self.required_nh_nr < 0] = 0

    def fill_one(self, hid: int, targetVM: VM, vm_numas: List = None) -> bool:
        if targetVM is None:
            return False
        if not self.is_feasible_flavor(hid, targetVM):
            return False
        if vm_numas is None:
            vm_numas = []
        vm_numas = vm_numas.copy()
        if len(vm_numas) == 0:
            numas = self.default_include_numa_flavor(hid, targetVM)
        else:
            numas = vm_numas

        required_nv_nr_flavor = np.array([targetVM.cpu // targetVM.numa, targetVM.mem // targetVM.numa])
        self.occupied_nh_nr[hid][numas] += required_nv_nr_flavor
        self.remained_nh_nr[hid][numas] -= required_nv_nr_flavor
        # self.required_nh_nr[hid][numas] = required_nv_nr_flavor - self.remained_nh_nr[hid][numas]
        # self.required_nh_nr[self.required_nh_nr < 0] = 0
        return True

    def fill(self, hids: List[int], targetVM: VM) -> int:
        flavor_count = 0
        for hid in hids:
            fill_bool = True
            while fill_bool:
                fill_bool = self.fill_one(hid, targetVM)
                if fill_bool:
                    flavor_count += 1
        return flavor_count

    def default_include_numa(self, vmid: int, hid: int) -> List[int]:
        assert self.is_feasible(vmid, hid)
        numas = []
        for numa_id in np.flatnonzero(self.is_feasible_numa(vmid, hid)):
            if len(numas) == self.required_nv_numa[vmid]:
                break
            else:
                numas.append(numa_id)
        return numas

    def default_include_numa_flavor(self, hid: int, target_flavor: VM) -> List[int]:
        assert self.is_feasible_flavor(hid, target_flavor)
        numas = []
        for numa_id in np.flatnonzero(self.is_feasible_numa_flavor(hid, target_flavor)):
            if len(numas) == target_flavor.numa:
                break
            else:
                numas.append(numa_id)
        return numas

    def backup(self) -> None:
        self.backup_mapping = self.mapping.copy()
        self.backup_numas = self.numas.copy()
        self.backup_occupied_nh_nr = self.occupied_nh_nr.copy()
        self.backup_occupied_nh_nr_true = self.occupied_nh_nr_true.copy()
        self.backup_remained_nh_nr = self.remained_nh_nr.copy()
        # if self.required_nh_nr is not None:
        #     self.backup_required_nh_nr = self.required_nh_nr.copy()

    def restore(self) -> None:
        self.mapping = self.backup_mapping.copy()
        self.numas = self.backup_numas.copy()
        self.occupied_nh_nr = self.backup_occupied_nh_nr.copy()
        self.occupied_nh_nr_true = self.backup_occupied_nh_nr_true.copy()
        self.remained_nh_nr = self.backup_remained_nh_nr.copy()
        # if self.backup_required_nh_nr is not None:
        #     self.required_nh_nr = self.backup_required_nh_nr.copy()

    def compute_score(self) -> float:
        # 开机的宿主机数目
        A = len(np.unique(self.mapping[self.mapping != -1]))
        # 发生迁移的虚机
        moved_vm_mask = self.init_mapping != self.mapping
        M = np.sum(self.required_nv_nr_all[moved_vm_mask, Resources.MEM])
        return A * self.settings.wa + (M / 1024 / 1024) * self.settings.wm

    def compute_score_flavor(self) -> float:
        # 开机的宿主机数目
        A = len(np.unique(self.mapping[self.mapping != -1]))
        # 发生迁移的虚机
        moved_vm_mask = self.init_mapping != self.mapping
        M = np.sum(self.required_nv_nr_all[moved_vm_mask, Resources.MEM])
        return A * self.settings.wa + (M / 1024 / 1024) * self.settings.wm

    def get_vmids_on_hid(self, hid) -> np.array:
        return np.flatnonzero(self.mapping == hid)

    def get_require_nv_3d(self, vmids: List) -> np.array:
        # 宿主机上各个虚机在每个numa上的需求资源
        required_nv_nr_3d = np.zeros((len(vmids), HOST_NUMA, RESOURCE_NUM))
        for i, vmid in enumerate(vmids):
            required_nv_nr_3d[i][self.numas[vmid], :] = self.required_nv_nr[vmid]
        return required_nv_nr_3d

    def is_feasible_numa(self, vmid: int, hid: int) -> np.array:
        return np.all(self.remained_nh_nr[hid] >= self.required_nv_nr[vmid], axis=1)

    def is_feasible(self, vmid: int, hid: int) -> bool:
        return np.sum(self.is_feasible_numa(vmid, hid)) >= self.required_nv_numa[vmid]

    def is_feasible_numa_index(self, vmid: int, hid: int, numa_index: List) -> bool:
        return np.all(self.is_feasible_numa(vmid, hid)[numa_index])

    def is_feasible_nh(self, vmid: int) -> np.array:
        # 至少有一台宿主机资源可以覆盖虚机
        return np.sum(np.all(self.remained_nh_nr >= self.required_nv_nr[vmid], axis=2), axis=1) >= self.required_nv_numa[vmid]

    def is_feasible_numa_flavor(self, hid: int, target_flavor: VM) -> np.array:
        required_nv_nr_flavor = np.array([target_flavor.cpu // target_flavor.numa, target_flavor.mem // target_flavor.numa])
        return np.all(self.remained_nh_nr[hid] >= required_nv_nr_flavor, axis=1)

    def is_feasible_flavor(self, hid: int, target_flavor: VM) -> bool:
        return np.sum(self.is_feasible_numa_flavor(hid, target_flavor)) >= target_flavor.numa

    def is_feasible_flavor_numa_index(self, hid: int, target_flavor: VM, numa_index: List) -> bool:
        return np.all(self.is_feasible_numa_flavor(hid, target_flavor)[numa_index])

    def is_feasible_nh_flavor(self, target_flavor: VM) -> np.array:
        # 至少有一台宿主机资源可以覆盖虚机
        required_nv_nr_flavor = np.array(
            [target_flavor.cpu // target_flavor.numa, target_flavor.mem // target_flavor.numa])
        return np.sum(np.all(self.remained_nh_nr >= required_nv_nr_flavor, axis=2), axis=1) >= target_flavor.numa

    def get_solution(self) -> Mapping:
        return Mapping(list(self.mapping), list(self.numas))

    def get_active_hids(self) -> np.array:
        return np.flatnonzero(self.occupied_nh_nr[:, :, Resources.MEM] > 0)

    def plot(self):
        # # host容量
        # capacity_nh_nr = np.array(
        #     [[[h.cpu // HOST_NUMA, h.mem // HOST_NUMA] for _ in range(HOST_NUMA)] for h in env.hosts])
        # # host已使用
        # n_hosts = len(env.hosts)
        # occupied_nh_nr = np.zeros((n_hosts, HOST_NUMA, 2))
        #
        # required_nv_nr_numa = np.array([[[v.cpu // v.numa, v.mem // v.numa] if numa_id in env.mapping.numas[i] else [0, 0]
        #                                  for numa_id in range(HOST_NUMA)] for i, v in enumerate(env.vms)])
        # np.add.at(occupied_nh_nr, mapping, required_nv_nr_numa)

        # 数据
        data = self.occupied_nh_nr

        # 提取数据
        x = np.arange(data.shape[0]) + 1  # 宿主机编号
        y = np.arange(data.shape[1]) + 1  # NUMA

        # 分离 CPU 和 Memory 数据
        z_cpu = data[:, :, 0]  # CPU资源
        z_mem = data[:, :, 1]  # 内存资源

        # 创建图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制 CPU 柱状图（红色）
        for i in range(data.shape[1]):
            ax.bar(x - 0.1, z_cpu[:, i], zs=i + 1, zdir='y', width=0.2, color='r', label='CPU' if i == 0 else '')

        # 绘制 Memory 柱状图（蓝色）
        for i in range(data.shape[1]):
            ax.bar(x + 0.1, z_mem[:, i], zs=i + 1, zdir='y', width=0.2, color='b', label='Memory' if i == 0 else '')

        ax.set_xlabel('Host')
        ax.set_ylabel('NUMA')
        ax.set_zlabel('Resource')
        ax.set_title('CPU and Memory Resources in Hosts and NUMA')
        ax.set_xticks(x)
        ax.set_yticks(y)
        ax.set_yticklabels([f'NUMA {i + 1}' for i in range(data.shape[1])])

        plt.legend()
        plt.show()

    def cal_induce_degree(self, hids: List, targetVM: VM) -> float:
        f_nr = np.array([targetVM.cpu, targetVM.mem])
        rem_nh_nr = np.sum(self.remained_nh_nr[hids], axis=1)
        capacity = np.sum(np.amin(rem_nh_nr / f_nr, axis=1))
        potential_capacity = np.amin(np.sum(self.remained_nh_nr[hids], axis=(0, 1)) / f_nr)

        # f_nr = np.array([targetVM.cpu//targetVM.numa, targetVM.mem//targetVM.numa])
        # vm_numa_num = np.amin(self.remained_nh_nr[hids] / f_nr, axis=2)
        # capacity = 0
        # for i in range(len(hids)):
        #     capacity += vm_numa_num[i][np.argsort(vm_numa_num[i])[::-1][targetVM.numa-1]]
        #
        # vm_global_numa_num = np.amin(self.remained_nh_nr[hids] / f_nr, axis=2)
        # potential_capacity = 0
        # for i in range(len(hids)):
        #     potential_capacity += vm_numa_num[i][np.argsort(vm_numa_num[i])[::-1][targetVM.numa - 1]]
        # potential_capacity = np.amin(np.sum(self.remained_nh_nr[hids], axis=(0, 1)) / (f_nr * targetVM.numa))
        return capacity/potential_capacity

    def cal_global_optimal_res_diff(self, hids: List, targetVM: VM) -> np.array:
        f_nr = np.array([targetVM.cpu, targetVM.mem])
        rem_nh_nr = np.sum(self.remained_nh_nr[hids], axis=1)
        potential_capacity = np.amin(np.sum(rem_nh_nr, axis=0) / f_nr)
        optimal_resource = potential_capacity / len(hids) * f_nr
        optimal_diff = np.sum(self.remained_nh_nr[hids], axis=1) - optimal_resource

        return self.normalize(optimal_diff)

    def cal_projection_length(self, required_nv_nr_3d: np.array, target_host_nh: np.array, numa_index: List):
        target_host_nh_temp = self.normalize(target_host_nh)
        required_nv_nr_3d_temp = self.normalize(required_nv_nr_3d)
        dot_product = np.sum(required_nv_nr_3d_temp[:, numa_index, :] * target_host_nh_temp[numa_index], axis=(1, 2))
        target_norm = np.linalg.norm(target_host_nh_temp[numa_index])
        project_lengths = dot_product / (target_norm + 0.0001)
        return project_lengths, target_norm

    def cal_host_required_degree(self, required_nh: np.array) -> np.array:
        # 不应该直接累加numa，因为虚机可能只需要两个numa就行
        # 宿主机numa排序，累加numa.num个numa的需求资源
        required_nh_temp = np.sum(self.normalize(required_nh), axis=1)
        res_sum = np.sum(required_nh_temp, axis=0)
        degree = (res_sum[Resources.CPU] * required_nh_temp[:, Resources.CPU] +
                  res_sum[Resources.MEM] * required_nh_temp[:, Resources.MEM]) / sum(res_sum)

        return degree / np.sum(degree)

    def cal_host_induced_degree(self, hids: List, targetVM: VM) -> np.array:
        optimal_diff = self.cal_global_optimal_res_diff(hids, targetVM)
        induced_degree = np.abs(optimal_diff[:, Resources.CPU] - optimal_diff[:, Resources.MEM])
        return induced_degree / np.sum(induced_degree)

    def normalize(self, target: np.array) -> np.array:
        target_temp = copy.deepcopy(target)
        max_value = np.average(np.sum(self.capacity_nh_nr, axis=1), axis=0)
        target_temp = target_temp / max_value

        return target_temp