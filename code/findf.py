import copy

from assignment import Assignment
from environment import Environment, Settings, Mapping, VM
from algorithm import Algorithm
from utils import AttemptResult
from global_variables import *

import numpy as np
import heapq

from collections.abc import Iterable
from enum import Enum
from typing import List, Optional
from environment import Resources
from projection import *


class Situation(Enum):
    IMPOSSIBLE = 0
    AMPLE = 1
    BALANCED = 2
    LOPSIDED = 3


class Stash:
    """
    A priority queue that kept VMs sorted by size.
    The "form of the stash" i.e. cumulative vector of VMs requirements is
    kept precomputed.
    """

    def __init__(self, asg: Assignment, vmids=None) -> None:
        self.asg = asg
        self.vmids = list()
        self.form = np.zeros(2)
        self.add(vmids)

    def add(self, vmids: Iterable) -> None:
        for vmid in vmids:
            self.add_vmid(vmid)

    def add_vmid(self, vmid) -> None:
        # 根据虚机得分加入最大堆
        heapq.heappush(self.vmids, (self.asg.size_nv[vmid], vmid))
        # stack中的总资源
        self.form += self.asg.required_nv_nr_all[vmid]

    def pop(self) -> int:
        """ Remove largest VM from the stash and return its index """
        _, vmid = heapq.heappop(self.vmids)
        self.form -= self.asg.required_nv_nr_all[vmid]
        return vmid

    def get_form(self) -> np.array:
        return self.form

    def is_empty(self) -> bool:
        return len(self.vmids) == 0


class ProbForceFit:
    def __init__(self, asg: Assignment, max_force_steps=4000):
        self.asg = asg
        self.force_steps_counter = 0
        self.max_force_steps = max_force_steps
        self.alpha = 0.95

    def place_vmids(self, vmids: List[int], hids: List[int]) -> bool:
        stash = Stash(asg=self.asg, vmids=vmids)
        self.force_steps_counter = 0
        while not stash.is_empty():
            vmid = stash.pop()
            # 根据比值分类
            # 为放置目标VM，宿主机还缺少资源required_nh_nv
            required_nh_nv = self.asg.required_nv_nr[vmid] - self.asg.remained_nh_nr
            required_nh_nv[required_nh_nv <= 0] = 0

            situation = self.classify(hids, vmid, required_nh_nv)
            if situation == Situation.IMPOSSIBLE:
                return False
            elif situation == Situation.AMPLE:
                self.best_fit(vmid, hids)
            else:
                # return False
                self.force_steps_counter += 1
                if self.force_steps_counter >= self.max_force_steps:
                    return False

                # 集群均衡度，资源互补程度
                required_nh_nv_normalize = self.normalize(required_nh_nv)
                required_nh_sum = np.sum(required_nh_nv_normalize, axis=1)


                required_nh_sum_res = np.sum(required_nh_nv_normalize, axis=(0, 1))

                # 紧缺资源
                FST_RES, SEC_RES = (Resources.CPU, Resources.MEM)\
                    if required_nh_sum_res[Resources.CPU] > required_nh_sum_res[Resources.MEM]\
                    else (Resources.MEM, Resources.CPU)


                # 可迁移虚机加起来符合要求的numa数目要置VM Host还需要的numa数目
                hids_filtered = self.filter_hosts(vmid, hids, required_nh_nv)
                if len(hids_filtered) == 0:
                    return False

                # 目标宿主机上的numa节点，以及这些节点上总共需要多少资源
                require_nh_numa = np.zeros((len(hids_filtered), RESOURCE_NUM))
                for i, hid in enumerate(hids_filtered):
                    numa_sort_ids = np.lexsort((required_nh_nv[hid][:, Resources.CPU], required_nh_nv[hid][:, Resources.MEM]))
                    numa_ids = numa_sort_ids[:self.asg.env.vms[vmid].numa]
                    require_nh_numa[i] = np.sum(required_nh_nv[hid][numa_ids], axis=0)

                # 要根据情况选择按照cpu还是mem排序。
                sorted_index = np.lexsort((require_nh_numa[:, Resources.CPU], require_nh_numa[:, Resources.MEM]))
                hids_sorted = hids_filtered[sorted_index]

                # 确定目标宿主机，其实可以选择多个，然后概率选择。
                hid = hids_sorted[0]
                vmids_migrate = []
                # 目标宿主机上选择迁出虚机(优先选择迁过来的虚机)
                first_numa_index = self.nth_numa_index(hid, self.asg.env.vms[vmid], required_nh_nv)

                if len(first_numa_index) == 0:
                    vmids_migrate = []

                latest_numa_index = []
                for numa_id in range(HOST_NUMA):
                    if numa_id not in first_numa_index:
                        latest_numa_index.append(numa_id)

                # 对虚机规格筛选，大于vmid的不考虑
                vmids_hid = self.asg.get_vmids_on_hid(hid)
                # 和前n个numa投影尽可能大，后面的numa投影尽可能小，两者相除
                # 投影的时候需要去除量纲
                required_nv_nr_3d = self.asg.get_require_nv_3d(vmids_hid)
                # 虚机计算投影匹配度，计算虚机得分
                vm_sort_metric = self.sort_numa_projection_metric(required_nv_nr_3d, required_nh_nv[hid],
                                                             first_numa_index, latest_numa_index)

                first_key = (self.asg.init_mapping == self.asg.mapping)[vmids_hid]
                # smaller_flavor = np.any(self.asg.required_nv_nr_all[vmids_hid] > self.asg.required_nv_nr_all[vmid],
                #                             axis=1)
                # 对虚机进行逆序排序（优先选择迁过来的虚机）
                vm_sort_ids = np.lexsort((vm_sort_metric, first_key))

                vmids_migrate = vmids_hid[vm_sort_ids]
                residue = self.push_vmid(vmid, hid, vmids_migrate)
                stash.add(residue)
        return stash.is_empty()

    def nth_numa_index(self, hid: int, vm: VM, host_required: np.array) -> List[int]:
        # 固定大规格虚机，判断哪些numa可以支持required_nh，然后根据required的内存和cpu排序
        nums_filter = self.filter_numas(hid, vm, host_required)
        num = vm.numa
        if len(nums_filter) < num:
            return []
        first_key = host_required[hid][nums_filter][:, Resources.MEM]
        second_key = host_required[hid][nums_filter][:, Resources.CPU]
        numa_index = np.lexsort((second_key, first_key))
        return nums_filter[numa_index[:num]]

    def filter_numas(self, hid: int, vm: VM, host_required: np.array) -> np.array:
        """ Filter numas that can't host the flavor in any case """
        vmids = self.asg.get_vmids_on_hid(hid)
        # 固定规格比flavor大的VM，不作为可迁移虚机。不能因为需要一个小规格的空间，然后踢出一个大规格的虚机
        # migratable = ~np.any(self.asg.required_nv_nr_all[vmids] > np.array([vm.cpu, vm.mem]), axis=1)
        migratable = [True] * len(vmids)

        # 宿主机上拥有可迁移虚机的资源，要超过host的需求资源，不然即使全部迁移，也无法得到required_nh_nr[hid]
        mask = np.all(
                np.sum(self.asg.get_require_nv_3d(vmids[migratable]), axis=0) >= host_required[hid],
                axis=1
        )
        return np.arange(HOST_NUMA)[mask]

    def classify(self, hids: List[int], vmid: int, required_nh: np.array) -> Situation:
        # 有宿主机可以覆盖虚机
        if np.any(self.asg.is_feasible_nh(vmid)[hids]):
            return Situation.AMPLE

        # 宿主机的规格要大于虚机才行
        hids_filtered = self.filter_hosts(vmid, hids, required_nh)
        if len(hids_filtered) == 0:
            return Situation.IMPOSSIBLE

    def select_tightest_hid(self, hids: np.array) -> int:
        load_nh = np.sum(
                self.asg.occupied_nh_nr[hids] / self.asg.capacity_nh_nr[hids],
                axis=(1, 2)
        )
        return hids[np.argmax(load_nh)]

    def best_fit(self, vmid: int, hids: np.array) -> None:
        feasible_nh = self.asg.is_feasible_nh(vmid)[hids]
        feasible_hids = np.asarray(hids)[feasible_nh]
        tightest_hid = self.select_tightest_hid(feasible_hids)
        self.asg.include(vmid, tightest_hid)

    def filter_hosts(self, vmid: int, hids: np.array, required_nh: np.array) -> np.array:
        """ Filter hosts that can't host the VM in any case """
        filter_hids = []
        for hid in hids:
            vmids = self.asg.get_vmids_on_hid(hid)
            # 只考虑迁移规格小于目标vm的
            # mask = ~np.any(self.asg.required_nv_nr_all[vmids] > self.asg.required_nv_nr_all[vmid], axis=1)
            mask = [True] * len(vmids)
            vmids_for_migrate = vmids[mask]
            # 可迁移虚机
            if np.sum(np.all(np.sum(self.asg.get_require_nv_3d(vmids_for_migrate), axis=0) >= required_nh[hid], axis=1)) >= self.asg.env.vms[vmid].numa:
                filter_hids.append(hid)

        return np.array(filter_hids)

    def push_vmid(self, in_vmid: int, hid: int, vmids: np.array) -> np.array:
        ejected_vmids = list()
        for vmid in vmids:
            # if self.asg.env.vms[vmid].cpu > self.asg.env.vms[in_vmid].cpu and self.asg.env.vms[vmid].mem > self.asg.env.vms[in_vmid].mem:
            #     continue
            self.asg.exclude(vmid)
            ejected_vmids.append(vmid)
            if self.asg.is_feasible(in_vmid, hid):
                break

        if self.asg.is_feasible(in_vmid, hid):
            self.asg.include(in_vmid, hid)
        else:
            for vmid in ejected_vmids[::-1]:
                self.asg.include(vmid, hid, self.asg.backup_numas[vmid])

        if len(ejected_vmids) == 0:
            return []

        residue = [ejected_vmids[-1]]
        for vmid in ejected_vmids[::-1][1:]:
            if self.asg.is_feasible(vmid, hid):
                self.asg.include(vmid, hid, self.asg.backup_numas[vmid])
            else:
                residue.append(vmid)
        return residue

    def sort_numa_projection_metric(self, required_nv_nr_3d: np.array, target_host_nh: np.array, first_numa_index: List,
                                    latest_numa_index: List) -> list:
        project_len_first, target_norm = self.cal_projection_length(required_nv_nr_3d, target_host_nh,
                                                               first_numa_index)
        project_len_latest, _ = self.cal_projection_length(required_nv_nr_3d, target_host_nh,
                                                      latest_numa_index)
        # 虚机投影得分
        # 均衡度，不均衡的时候需要优先考虑缓解不均衡的VM，均衡的时候固定大于等于规格虚机，优先考虑小规格虚机，优先迁移之前迁过来的虚机
        # 紧缺资源
        vm_sort_metric = np.abs(project_len_first - target_norm) * project_len_latest
        return vm_sort_metric

    def cal_projection_length(self, required_nv_nr_3d: np.array, target_host_nh: np.array, numa_index: List):
        target_host_nh_temp = self.normalize(target_host_nh)
        required_nv_nr_3d_temp = self.normalize(required_nv_nr_3d)
        dot_product = np.sum(required_nv_nr_3d_temp[:, numa_index, :] * target_host_nh_temp[numa_index], axis=(1, 2))
        target_norm = np.linalg.norm(target_host_nh_temp[numa_index])
        project_lengths = dot_product / (target_norm + 1)
        return project_lengths, target_norm

    def normalize(self, target: np.array) -> np.array:
        target_temp = copy.deepcopy(target)
        max_value = np.average(self.asg.capacity_nh_nr, axis=0)
        target_temp = target_temp / max_value

        return target_temp


class FinDf(Algorithm):
    """Reference implementation of SerConFF algorithm """

    def __init__(self, env: Environment, settings: Settings, target_flavor: VM) -> None:
        super().__init__(env, settings)
        # 计算host和vm的资源
        self.asg = Assignment(env, settings, target_flavor)
        self.placer = ProbForceFit(self.asg, max_force_steps=settings.max_force_steps)
        self.debug = False

    def select_eliminate_vms(self, hid: int) -> np.array:
        assert HOST_NUMA >= self.asg.flavor.numa
        # host上numa排序，选择前f个numa
        first_numa_index = self.placer.nth_numa_index(hid, self.asg.flavor, self.asg.required_nh_nr)
        if len(first_numa_index) == 0:
            return []

        latest_numa_index = []
        for numa_id in range(HOST_NUMA):
            if numa_id not in first_numa_index:
                latest_numa_index.append(numa_id)

        # 对虚机规格筛选，大于flavor的不考虑
        vmids_hid = self.asg.get_vmids_on_hid(hid)

        # smaller_flavor_ids = ~np.any(self.asg.required_nv_nr_all[vmids_hid] > self.asg.required_nv_nr_flavor_all, axis=1)
        # vmids = vmids_hid[smaller_flavor_ids]
        vmids = vmids_hid
        if len(vmids) == 0:
            return []

        # 和前n个numa投影尽可能大，后面的numa投影尽可能小，两者相除
        required_nv_nr_3d = self.asg.get_require_nv_3d(vmids)
        # 虚机计算投影，计算虚机得分
        vm_sort_metric = self.placer.sort_numa_projection_metric(required_nv_nr_3d, self.asg.required_nh_nr[hid], first_numa_index, latest_numa_index)

        first_key = (self.asg.init_mapping == self.asg.mapping)[vmids]
        # 对虚机进行逆序排序（优先选择迁过来的虚机）
        vm_sort_ids = np.lexsort((vm_sort_metric, first_key))
        vmids = vmids[vm_sort_ids]

        # 按顺序选择虚机，直至覆盖required_nh_nr
        vm_eliminate_list = self.push_vmid(hid, vmids)

        return vm_eliminate_list

    def push_vmid(self, hid: int, vmids: np.array) -> np.array:
        ejected_vmids = list()
        for vmid in vmids:
            # if self.asg.env.vms[vmid].cpu > self.asg.flavor.cpu and self.asg.env.vms[vmid].mem > self.asg.flavor.mem:
            #     continue
            self.asg.exclude(vmid)
            ejected_vmids.append(vmid)
            if self.asg.is_feasible_flavor(hid):
                break

        if self.asg.is_feasible_flavor(hid):
            self.asg.fill_one(hid)
        else:
            for vmid in ejected_vmids[::-1]:
                self.asg.include(vmid, hid, self.asg.backup_numas[vmid])

        if len(ejected_vmids) == 0:
            return []

        residue = [ejected_vmids[-1]]
        for vmid in ejected_vmids[::-1][1:]:
            if self.asg.is_feasible(vmid, hid):
                self.asg.include(vmid, hid, self.asg.backup_numas[vmid])
            else:
                residue.append(vmid)
        return residue

    def choose_hid_to_try(self, hids: np.array) -> int:
        first_key = np.sum(self.asg.required_nh_nr, axis=1)[hids, Resources.MEM]
        second_key = np.sum(self.asg.required_nh_nr, axis=1)[hids, Resources.CPU]
        return hids[np.lexsort((second_key, first_key))[0]]

    def log_result(self, result: AttemptResult, hid: int) -> None:
        self.log(f'try host {hid:<5}\t'
                 f'force steps: {self.placer.force_steps_counter:<7}\t'
                 f'{result}')

    def solve_(self) -> Mapping:
        # 可选择的目标宿主机id
        allowed_hids = list(self.asg.hids)
        # 未尝试碎片整理的宿主机id
        hosts_to_try = list(self.asg.hids)
        # 使用指定flavor进行填充
        flavor_count_old = self.asg.fill(list(self.asg.hids))

        count = 0
        while hosts_to_try:
            print(count, len(self.asg.hids))
            # 环境保存
            self.asg.backup()

            # 根据内存进行排序
            hid = self.choose_hid_to_try(hosts_to_try)
            if hid == 149:
                a = 1
            hosts_to_try.remove(hid)
            allowed_hids.remove(hid)
            # 选择的宿主机根据与required_nh向量的投影长度对虚机排序，选择虚机迁出
            # 因为在一开始已经使用flavor对宿主机进行了fill，所以如果可以正常迁移的话，是会有虚机迁移列表的
            vmids = self.select_eliminate_vms(hid)
            result = AttemptResult.FAIL
            if len(vmids) != 0:
                # 在可选择的目标宿主机上放置虚机
                place_bool = self.placer.place_vmids(vmids, allowed_hids)
                if place_bool:
                    env_new = copy.deepcopy(self.asg.env)
                    env_new.mapping.mapping = self.asg.mapping
                    env_new.mapping.numas = self.asg.numas
                    asg_new = Assignment(env_new, self.asg.settings,self.asg.flavor)
                    flavor_count_new = asg_new.fill(list(self.asg.hids))
                    if flavor_count_new <= flavor_count_old:
                        result = AttemptResult.WORSE
                    else:
                        flavor_count_old = flavor_count_new
                        result = AttemptResult.SUCCESS

            if result != AttemptResult.SUCCESS:
                allowed_hids.append(hid)
                self.asg.restore()

            self.log_result(result, hid)
            count += 1

        return self.asg.get_solution()
