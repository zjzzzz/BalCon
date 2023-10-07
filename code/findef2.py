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

class Prohibitor:
    """
    If one host is chosen more than two times in a row, another host is chosen.
    All possible hosts will be chosen in a long run.
    """

    def __init__(self, n_hosts):
        self.last_hid = -1
        self.last_hid_counter = 0
        self.tabu_score = np.full(n_hosts, 0, dtype=np.float)

    def forbid_long_repeats(self, hids, hid):
        if hid == self.last_hid:
            self.last_hid_counter += 1
            if self.last_hid_counter > 2:
                hid = hids[np.argmax(self.tabu_score[hids])]
                self.tabu_score[hid] -= 1
        else:
            self.last_hid = hid
            self.last_hid_counter = 0
        return hid


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
        self.vm_layers = [0 for _ in range(self.asg.n_vms)]
        self.add(vmids, 0)

    def add(self, vmids: Iterable, parent_layer: int) -> None:
        for vmid in vmids:
            self.add_vmid(vmid, parent_layer)

    def add_vmid(self, vmid, parent_layer: int) -> None:
        # 根据虚机得分加入最大堆
        heapq.heappush(self.vmids, (self.asg.size_nv[vmid], vmid))
        # stack中的总资源
        self.form += self.asg.required_nv_nr_all[vmid]
        self.vm_layers[vmid] = parent_layer+1

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
    def __init__(self, asg: Assignment, max_layer=4):
        self.asg = asg
        self.force_steps_counter = 0
        self.max_layer = max_layer
        self.alpha = 0.95
        self.induced_degree = 1

    def place_vmids(self, vmids: List[int], hids: List[int]) -> bool:
        stash = Stash(asg=self.asg, vmids=vmids)
        self.force_steps_counter = 0
        while not stash.is_empty():
            vmid = stash.pop()
            # 根据比值分类
            # 为放置目标VM，宿主机还缺少资源required_nh_nv
            required_nh_nv = self.asg.required_nv_nr[vmid] - self.asg.remained_nh_nr
            required_nh_nv[required_nh_nv <= 0] = 0

            situation = self.classify(hids, vmid)
            if situation == Situation.IMPOSSIBLE:
                return False
            elif situation == Situation.AMPLE:
                self.best_fit(vmid, hids)
            else:
                # return False

                self.induced_degree = self.asg.cal_induce_degree(hids, self.asg.env.vms[vmid], stash.get_form())

                # 可迁移虚机加起来符合要求的numa数目要置VM Host还需要的numa数目
                hids_filtered = self.filter_hosts(vmid, hids)
                if len(hids_filtered) == 0:
                    return False

                # 确定目标宿主机，其实可以选择多个，然后概率选择。
                hid = self.choose_hid_to_try(list(hids_filtered), self.asg.env.vms[vmid])

                normalize_required = self.asg.normalize(required_nh_nv)
                impt = np.sum(normalize_required[hid], axis=0) / np.sum(normalize_required[hid])
                cpu_impt, mem_impt = impt[0], impt[1]

                vmids_migrate = []
                # 对虚机规格筛选，大于vmid的不考虑
                vmids_hid = self.asg.get_vmids_on_hid(hid)
                vms_normalize = self.asg.normalize(self.asg.required_nv_nr_all[vmids_hid])
                vm_flavor = vms_normalize[:, Resources.CPU] * cpu_impt + \
                            vms_normalize[:, Resources.MEM] * mem_impt
                # smaller_flavor_ids = vm_flavor <= self.asg.env.vms[vmid].cpu * cpu_impt + self.asg.env.vms[vmid].mem * mem_impt
                # vmids_filter = vmids_hid[smaller_flavor_ids]
                vmids_filter = vmids_hid
                if len(vmids_filter) == 0:
                    return False

                # 虚机计算投影匹配度，计算虚机得分
                vm_local_metric = self.cal_vm_local_metric(vmids_filter, required_nh_nv[hid],
                                                             cpu_impt, mem_impt, self.asg.env.vms[vmid].numa)
                vm_global_metric = self.cal_vm_induce_metric(hid, vmids_filter, self.asg.env.vms[vmid], cpu_impt, mem_impt)
                vm_metric = self.induced_degree * vm_global_metric + (1 / self.induced_degree) * vm_local_metric

                flavor_key = vm_flavor
                migrate_key = (self.asg.init_mapping == self.asg.mapping)[vmids_filter]
                # smaller_flavor = np.any(self.asg.required_nv_nr_all[vmids_hid] > self.asg.required_nv_nr_all[vmid],
                #                             axis=1)
                # 对虚机进行逆序排序（优先选择迁过来的虚机）
                vm_sort_ids = np.lexsort((migrate_key, vm_metric))

                vmids_migrate = vmids_filter[vm_sort_ids]
                residue = self.push_vmid(vmid, hid, vmids_migrate)
                stash.add(residue, stash.vm_layers[vmid])
                for vmid in residue:
                    if stash.vm_layers[vmid] > self.max_layer:
                        return False
        return stash.is_empty()

    def choose_hid_to_try(self, hids: np.array, flavor: VM) -> int:
        # 计算需求资源
        required_nh_nr = self.asg.cal_required_nh(flavor)

        local_degree = self.asg.cal_host_required_degree(required_nh_nr[hids], flavor.numa)
        global_degree = -1*self.asg.cal_host_induced_degree(hids, flavor)
        degree = self.induced_degree*(global_degree) + (1/self.induced_degree) * local_degree
        # first_key = np.sum(self.asg.required_nh_nr, axis=1)[hids, Resources.MEM]
        # second_key = np.sum(self.asg.required_nh_nr, axis=1)[hids, Resources.CPU]
        return hids[np.argsort(degree)[0]]

    def nth_numa_index(self, hid: int, vm: VM, host_required: np.array) -> (List[int], List[int]):
        # 固定大规格虚机，判断哪些numa可以支持required_nh，然后根据required的内存和cpu排序
        nums_filter = self.filter_numas(hid, vm, host_required)
        num = vm.numa
        if len(nums_filter) < num:
            return []

        required_nh_temp = self.asg.normalize(host_required[hid])
        res_sum = np.sum(required_nh_temp[nums_filter, :], axis=0)
        degree = (res_sum[Resources.CPU] * required_nh_temp[nums_filter, Resources.CPU] +
                  res_sum[Resources.MEM] * required_nh_temp[nums_filter, Resources.MEM]) / sum(res_sum)

        # first_key = host_required[hid][nums_filter][:, less_res]
        # second_key = host_required[hid][nums_filter][:, more_res]
        numa_index = np.argsort(degree)
        first_numa_index = nums_filter[numa_index[:num]]

        latest_numa_index = []
        for numa_id in range(HOST_NUMA):
            if numa_id not in first_numa_index:
                latest_numa_index.append(numa_id)
        return first_numa_index, latest_numa_index

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

    def classify(self, hids: List[int], vmid: int) -> Situation:
        # 有宿主机可以覆盖虚机
        if np.any(self.asg.is_feasible_nh(vmid)[hids]):
            return Situation.AMPLE

        # 宿主机的规格要大于虚机才行
        hids_filtered = self.filter_hosts(vmid, hids)
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

    def filter_hosts(self, vmid: int, hids: np.array) -> np.array:
        """ Filter hosts that can't host the VM in any case """
        mask = np.all(
                self.asg.capacity_nh_nr[hids] >= self.asg.required_nv_nr[vmid],
                axis=(1, 2)
        )
        return np.array(hids)[mask]

    def push_vmid(self, in_vmid: int, hid: int, vmids: np.array) -> np.array:
        ejected_vmids = list()
        ejected_vm_nums = []
        for vmid in vmids:
            # 避免成环
            if self.asg.backup_mapping[vmid] != self.asg.mapping[vmid] or np.any(self.asg.backup_numas[vmid] != self.asg.numas[vmid]):
                continue
            ejected_vmids.append(vmid)
            ejected_vm_nums.append(self.asg.numas[vmid])
            self.asg.exclude(vmid)
            if self.asg.is_feasible(in_vmid, hid): break

        if self.asg.is_feasible(in_vmid, hid):
            self.asg.include(in_vmid, hid)
        else:
            for i, vmid in enumerate(ejected_vmids.copy()):
                self.asg.include(vmid, hid, ejected_vm_nums[i])
                ejected_vmids.remove(vmid)

        if len(ejected_vmids) == 0:
            return [in_vmid]

        residue = list()
        for vmid in ejected_vmids[::-1]:
            if self.asg.is_feasible(vmid, hid):
                self.asg.include(vmid, hid)
            else:
                residue.append(vmid)
        return residue

    def cal_vm_projection_metric(self, vmids: List, required_host_nh: np.array, first_numa_index: List,
                                    latest_numa_index: List) -> np.array:
        required_nv_nr_3d = self.asg.get_require_nv_3d(vmids)
        normalize_vm = self.asg.normalize(self.asg.required_nv_nr_all[vmids])

        # 虚机投影，匹配程度
        project_len_first, target_norm = self.asg.cal_projection_length(required_nv_nr_3d, required_host_nh,
                                                               first_numa_index)
        project_len_latest, _ = self.asg.cal_projection_length(required_nv_nr_3d, required_host_nh,
                                                      latest_numa_index)
        # 虚机投影得分
        # 虚机计算投影，计算虚机得分, 和前n个numa投影尽可能大，后面的numa投影尽可能小，两者相除
        vm_sort_metric = np.array((np.abs(project_len_first - target_norm) * (project_len_latest - project_len_first))) / np.sum(normalize_vm, axis=1)
        return vm_sort_metric / np.sum(vm_sort_metric)

    def cal_vm_local_metric(self, vmids: List, required_host_nh: np.array, cpu_impt: float, mem_impt: float, numa_num: int) -> np.array:
        required_nv_nr_3d = self.asg.get_require_nv_3d(vmids)
        required_after_migrate = required_host_nh-required_nv_nr_3d
        required_after_migrate[required_after_migrate < 0] = 0

        normalize_nh_before = self.asg.normalize(required_host_nh)
        normalize_before_degree = normalize_nh_before[:, Resources.CPU] * cpu_impt + normalize_nh_before[:, Resources.MEM] * mem_impt
        before_degree = np.sum(np.sort(normalize_before_degree)[:numa_num])

        normalize_nh_after = self.asg.normalize(required_after_migrate)
        normalize_after_degree = normalize_nh_after[:, :, Resources.CPU] * cpu_impt + normalize_nh_after[:, :, Resources.MEM] * mem_impt
        after_degree = np.sum(np.sort(normalize_after_degree, axis=1)[:, :numa_num], axis=1)
        vm_local_metric = np.exp(after_degree - before_degree)

        vm_migrate_cost = np.sum(self.asg.normalize(self.asg.required_nv_nr_all[vmids]) * np.array([cpu_impt, mem_impt]), axis=1)
        vm_local_metric = vm_local_metric*vm_migrate_cost

        return vm_local_metric / np.abs(np.sum(vm_local_metric)+0.0001)

    def cal_vm_induce_metric(self, hid: int, vmids: List, targetVM: VM, cpu_impt: float, mem_impt: float) -> np.array:
        remained_nh_nr_all = np.sum(self.asg.remained_nh_nr[hid], axis=0)
        rate = remained_nh_nr_all / np.array([targetVM.cpu, targetVM.mem])
        induced_degree_before = np.abs(rate[Resources.CPU] - rate[Resources.MEM])

        offset = remained_nh_nr_all + self.asg.required_nv_nr_all[vmids]
        rate_after = offset / np.array([targetVM.cpu, targetVM.mem])
        induced_degree_after = np.abs(rate_after[:, Resources.CPU] - rate_after[:, Resources.MEM])
        vm_induce_metric = np.exp(induced_degree_after - induced_degree_before)

        vm_migrate_cost = np.sum(self.asg.normalize(self.asg.required_nv_nr_all[vmids]) * np.array([cpu_impt, mem_impt]), axis=1)
        vm_induce_metric = vm_induce_metric*vm_migrate_cost

        return vm_induce_metric / np.abs(np.sum(vm_induce_metric)+0.0001)

    def cal_vm_impt(self, vmids: List, required_host_nh: np.array, first_numa_index: List) -> np.array:
        required_nv_nr_3d = self.asg.get_require_nv_3d(vmids)

        # 虚机重要性程度
        # 如果没有这台虚机，first_numa_index的numa能不能覆盖
        res_sum_without_vm = np.sum(required_nv_nr_3d[:, first_numa_index, :], axis=0)[np.newaxis, :, :] - required_nv_nr_3d[:, first_numa_index, :]
        vm_impt = ~np.all(res_sum_without_vm >= required_host_nh[first_numa_index], axis=(1, 2))

        # # 如果没有这台虚机，numa上可选虚机数目
        # np.sum(np.any(required_nv_nr_3d > 0, axis=2), axis=0)
        return vm_impt


class FinDf2(Algorithm):
    """Reference implementation of SerConFF algorithm """

    def __init__(self, env: Environment, settings: Settings, target_flavor: VM) -> None:
        super().__init__(env, settings)
        # 计算host和vm的资源
        self.asg = Assignment(env, settings)
        self.debug = False
        self.induced_degree = 1.0
        self.target_flavor = target_flavor
        self.placer = ProbForceFit(self.asg, max_force_steps=settings.max_force_steps)

    def choose_vms_to_try(self, hid: int) -> np.array:
        # 计算需求资源
        required_nh_nr = self.asg.cal_required_nh(self.target_flavor)
        normalize_required = self.asg.normalize(required_nh_nr)
        impt = np.sum(normalize_required[hid], axis=0) / (np.sum(normalize_required[hid]) + 0.0001)
        cpu_impt, mem_impt = impt[0], impt[1]

        # 对虚机规格筛选，大于flavor的不考虑
        vmids_hid = self.asg.get_vmids_on_hid(hid)
        vms_normalize = self.asg.normalize(self.asg.required_nv_nr_all[vmids_hid])
        vm_flavor = vms_normalize[:, Resources.CPU] * cpu_impt + \
                    vms_normalize[:, Resources.MEM] * mem_impt

        # smaller_flavor_ids = vm_flavor <= self.target_flavor.cpu*cpu_impt + self.target_flavor.mem*mem_impt
        # vmids_filter = vmids_hid[smaller_flavor_ids]
        vmids_filter = vmids_hid
        if len(vmids_filter) == 0:
            return []

        # 虚机计算投影，计算虚机得分
        # 除已经满足需求的numa，虚机对其余numa的影响，选择
        vm_local_metric = self.placer.cal_vm_local_metric(vmids_filter, required_nh_nr[hid], cpu_impt, mem_impt, self.target_flavor.numa)
        vm_global_metric = self.placer.cal_vm_induce_metric(hid, vmids_filter, self.target_flavor, cpu_impt, mem_impt)
        vm_metric = self.induced_degree * vm_global_metric + (1/self.induced_degree) * vm_local_metric

        flavor_key = vm_flavor
        migrate_key = (self.asg.init_mapping == self.asg.mapping)[vmids_filter]
        # 对虚机进行排序（优先选择迁过来的虚机）
        vm_sort_ids = np.lexsort((migrate_key, vm_metric))
        vmids_sort = vmids_filter[vm_sort_ids]

        # 按顺序选择虚机，直至覆盖required_nh_nr
        vm_eliminate_list = self.push_vmid(hid, vmids_sort)

        return vm_eliminate_list

    def push_vmid(self, hid: int, vmids: np.array) -> np.array:
        ejected_vmids = list()
        ejected_vm_nums = []
        for vmid in vmids:
            # if self.asg.env.vms[vmid].cpu > self.asg.flavor.cpu and self.asg.env.vms[vmid].mem > self.asg.flavor.mem:
            #     continue
            ejected_vmids.append(vmid)
            ejected_vm_nums.append(self.asg.numas[vmid])
            self.asg.exclude(vmid)
            if self.asg.is_feasible_flavor(hid, self.target_flavor):
                break

        if self.asg.is_feasible_flavor(hid, self.target_flavor):
            self.asg.fill_one(hid, self.target_flavor)
        else:
            for i, vmid in enumerate(ejected_vmids.copy()):
                self.asg.include(vmid, hid, ejected_vm_nums[i])
                ejected_vmids.remove(vmid)

        if len(ejected_vmids) == 0:
            return []

        residue = [ejected_vmids[-1]]
        for vmid in ejected_vmids[::-1][1:]:
            if self.asg.is_feasible_numa_index(vmid, hid, self.asg.backup_numas[vmid]):
                self.asg.include(vmid, hid, self.asg.backup_numas[vmid])
            else:
                residue.append(vmid)
        return residue

    def log_result(self, result: AttemptResult, hid: int) -> None:
        self.log(f'try host {hid:<5}\t'
                 f'force steps: {self.placer.force_steps_counter:<7}\t'
                 f'{result}')

    def clear_host(self, hid: int) -> np.array:
        vmids = self.asg.get_vmids_on_hid(hid)
        for vmid in vmids:
            # 将vm的mapping设置为-1，相当于加入stack。然后重新计算宿主机上的已占用和剩余资源
            self.asg.exclude(vmid)
        return vmids

    def solve_(self) -> Mapping:
        # 可选择的目标宿主机id
        allowed_hids = list(self.asg.hids)
        # 未尝试碎片整理的宿主机id
        hosts_to_try = list(self.asg.hids)

        # 使用指定flavor进行填充
        flavor_count_old = self.asg.fill(list(self.asg.hids), self.target_flavor)

        count = 0
        while hosts_to_try:
            # print(count, len(self.asg.hids))
            # 环境保存
            self.asg.backup()
            self.induced_degree = self.asg.cal_induce_degree(allowed_hids, self.target_flavor, np.array([0, 0]))
            # 根据内存进行排序
            hid = self.placer.choose_hid_to_try(hosts_to_try, self.target_flavor)
            hosts_to_try.remove(hid)
            allowed_hids.remove(hid)
            # 选择的宿主机根据与required_nh向量的投影长度对虚机排序，选择虚机迁出
            # 因为在一开始已经使用flavor对宿主机进行了fill，所以如果可以正常迁移的话，是会有虚机迁移列表的
            vmids = self.choose_vms_to_try(hid)
            # 在可选择的目标宿主机上放置虚机
            place_bool = self.placer.place_vmids(vmids, allowed_hids)
            if place_bool:
                env_new = copy.deepcopy(self.asg.env)
                env_new.mapping.mapping = self.asg.mapping
                env_new.mapping.numas = self.asg.numas
                asg_new = Assignment(env_new, self.asg.settings)
                flavor_count_new = asg_new.fill(list(self.asg.hids), self.target_flavor)
                if flavor_count_new <= flavor_count_old:
                    result = AttemptResult.WORSE
                else:
                    flavor_count_old = flavor_count_new
                    result = AttemptResult.SUCCESS
            else:
                result = AttemptResult.FAIL
            if result != AttemptResult.SUCCESS:
                allowed_hids.append(hid)
                self.asg.restore()

            self.log_result(result, hid)
            count += 1

        return self.asg.get_solution()
