from assignment import Assignment
from environment import Environment, Settings, Mapping
from algorithm import Algorithm
from utils import AttemptResult
from environment import VM
import copy
from global_variables import *
from environment import Resources

import numpy as np
import heapq

from collections.abc import Iterable
from enum import Enum
from typing import List, Optional


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


class Prohibitor:
    """
    If one host is chosen more than two times in a row, another host is chosen.
    All possible hosts will be chosen in a long run.
    """

    def __init__(self, n_hosts):
        self.last_hid = -1
        self.last_hid_counter = 0
        self.tabu_score = np.full(n_hosts, 0, dtype=np.float32)

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


class Coin:
    def __init__(self) -> None:
        self.state = 0

    def flip(self) -> int:
        self.state = (self.state + 1) % 2
        return self.state


def tan(v: np.array) -> float:
    return v[0] / v[1]


class ForceFit:
    def __init__(self, asg: Assignment, max_layer=5):
        self.asg = asg
        self.coin = Coin()
        self.force_steps_counter = 0
        self.alpha = 0.95
        self.induced_degree = 1
        self.max_layer = max_layer

    def place_vmids(self, vmids: List[int], hids: List[int]) -> bool:
        stash = Stash(asg=self.asg, vmids=vmids)
        prohibitor = Prohibitor(n_hosts=len(self.asg.env.hosts))
        self.force_steps_counter = 0
        while not stash.is_empty():
            vmid = stash.pop()
            # 根据比值分类
            situation = self.classify(hids, stash, vmid)
            # stack中有虚机就必须force迁移到下一台宿主机上，凭什么？不能迁移到其他宿主机上嘛？这和宿主机的排序关联关系很大，以及和宿主机上虚机的规格关联也很大
            if situation == Situation.IMPOSSIBLE:
                return False
            elif situation == Situation.AMPLE:
                self.best_fit(vmid, hids)
            else:
                # if situation == Situation.BALANCED:
                #     # 拥有最多的规格小于该虚机的宿主机
                #     # hid = self.choose_hid_balanced(vmid, hids_filtered)
                #     # hid = prohibitor.forbid_long_repeats(hids_filtered, hid)
                #     vmids = self.choose_vmids_balanced(hid)
                # else:
                #     # hid = self.choose_hid_lopsided(vmid, hids_filtered)
                #     # hid = prohibitor.forbid_long_repeats(hids_filtered, hid)
                #     vmids = self.choose_vmids_lopsided(hid, vmid)

                self.induced_degree = self.asg.cal_induce_degree(hids, self.asg.env.vms[vmid], stash.get_form())
                hids_filtered = self.filter_hosts(vmid, hids)
                hid = self.choose_hid_to_try(list(hids_filtered), self.asg.env.vms[vmid])
                required_nh_nv = self.asg.required_nv_nr[vmid] - self.asg.remained_nh_nr
                required_nh_nv[required_nh_nv <= 0] = 0

                normalize_required = self.asg.normalize(required_nh_nv)
                impt = np.sum(normalize_required[hid], axis=0) / np.sum(normalize_required[hid])
                cpu_impt, mem_impt = impt[0], impt[1]

                vmids_migrate = []
                # 对虚机规格筛选，大于vmid的不考虑
                vmids_hid = self.asg.get_vmids_on_hid(hid)
                vms_normalize = self.asg.normalize(self.asg.required_nv_nr_all[vmids_hid])
                vm_flavor = np.sum(vms_normalize * np.array([cpu_impt, mem_impt]), axis=1)
                    # vms_normalize[:, Resources.CPU] * cpu_impt + \
                    #         vms_normalize[:, Resources.MEM] * mem_impt
                # 虚机计算投影匹配度，计算虚机得分

                target_vm_norm = self.asg.normalize(self.asg.required_nv_nr_all[vmid])
                smaller_flavor_ids = vm_flavor < np.sum(target_vm_norm * np.array([cpu_impt, mem_impt]))
                vmids_filter = vmids_hid[smaller_flavor_ids]
                # vmids_filter = vmids_hid
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

                residue = self.push_vmid(vmid, hid, vmids_migrate, cpu_impt, mem_impt)
                stash.add(residue, stash.vm_layers[vmid])
                for vmid_res in residue:
                    if stash.vm_layers[vmid_res] > self.max_layer:
                        return False
        return stash.is_empty()

    def choose_hid_to_try(self, hids: np.array, flavor: VM) -> int:
        # 计算需求资源
        required_nh_nr = self.asg.cal_required_nh(flavor)

        local_degree = self.asg.cal_host_required_degree(required_nh_nr[hids], flavor.numa)
        global_degree = -1*self.asg.cal_host_induced_degree(hids, flavor)
        degree = self.induced_degree*(global_degree) + (1/self.induced_degree) * local_degree
        # first_key = np.sum(self.asg.required_nh_nr, axis=1)[hids, Resources.MEM]
        return hids[np.argsort(degree)[0]]

    def classify(self, hids: List[int], stash: Stash, vmid: int) -> Situation:
        # 有宿主机可以覆盖虚机
        if np.any(self.asg.is_feasible_nh(vmid)[hids]):
            return Situation.AMPLE

        # 宿主机的规格要大于虚机才行
        hids_filtered = self.filter_hosts(vmid, hids)
        if len(hids_filtered) == 0:
            return Situation.IMPOSSIBLE

        f_nr = stash.get_form() + self.asg.required_nv_nr_all[vmid]
        rem_nh_nr = np.sum(self.asg.remained_nh_nr[hids], axis=1)
        capacity = np.sum(np.amin(rem_nh_nr / f_nr, axis=1))
        potential_capacity = np.amin(np.sum(rem_nh_nr, axis=0) / f_nr)

        if potential_capacity < 1.0:
            return Situation.IMPOSSIBLE
        elif capacity >= 1.0 and capacity >= self.alpha * potential_capacity:
            return Situation.BALANCED
        else:
            return Situation.LOPSIDED

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

    def count_smaller_vms_nh(self, vmid: int, hids: np.array) -> np.array:
        size = self.asg.size_nv[vmid]
        # 找到得分比这台虚机小的
        smaller_vmids = self.asg.vmids[(self.asg.size_nv <= size) &
                                       (self.asg.mapping != -1)]
        # 计算这些虚机在哪台宿主机上数量最多
        count = np.bincount(self.asg.mapping[smaller_vmids])
        result = np.zeros(max(hids) + 1)
        result[:count.size] = count[:max(hids) + 1]
        return result[hids]

    def choose_hid_lopsided(self, vmid: int, hids: np.array) -> int:
        rid = self.coin.flip()
        hid = self.choose_hid_lopsided_rid(vmid, hids, rid)
        if hid is not None:
            return hid

        rid = (rid + 1) % 2
        hid = self.choose_hid_lopsided_rid(vmid, hids, rid)
        if hid is not None:
            return hid

        # all hosts have the same tan as VM
        free_nh_nr = self.asg.remained_nh_nr[hids] / self.asg.capacity_nh_nr[hids]
        free_nh = np.sum(free_nh_nr, axis=(1, 2))
        idx = np.argmax(free_nh)
        return hids[idx]

    def filter_hosts(self, vmid: int, hids: np.array) -> np.array:
        """ Filter hosts that can't host the VM in any case """
        mask = np.all(
                self.asg.capacity_nh_nr[hids] >= self.asg.required_nv_nr[vmid],
                axis=(1, 2)
        )
        return np.array(hids)[mask]

    def choose_hid_lopsided_rid(self, vmid: int, hids: np.array, rid: int) -> Optional[int]:
        arid = (rid + 1) % 2
        tan_vm = self.asg.required_nv_nr_all[vmid, rid] / self.asg.required_nv_nr_all[vmid, arid]
        tan_nh = np.sum(self.asg.occupied_nh_nr, axis=1)[hids, rid] / np.sum(self.asg.occupied_nh_nr, axis=1)[hids, arid]
        free_nh = np.sum(self.asg.remained_nh_nr, axis=1)[hids, rid] / np.sum(self.asg.capacity_nh_nr, axis=1)[hids, rid]
        # 需要宿主机的角度小于虚机角度，这样两者相加可以得到更偏中间的角度。（那如果这个宿主机已经是中间了呢？还需要迁移虚机？是不是可以吧这个作为不强制迁移的一个指标）
        free_nh[~(tan_nh < tan_vm)] = -1
        # 找到符合要求宿主机中，相对剩余目标资源最多的宿主机
        idx = np.argmax(free_nh)
        if free_nh[idx] >= 0:
            return hids[idx]
        return None

    def choose_vmids_lopsided(self, hid: int, in_vmid: int) -> np.array:
        vmids = self.asg.get_vmids_on_hid(hid)
        # 这些角度其实都需要先除以host的capacity，形成比例之后再进行对比
        tan_in_vm = tan(self.asg.required_nv_nr_all[in_vmid])
        tan_host = tan(np.sum(self.asg.occupied_nh_nr, axis=1)[hid])
        tan_nv = self.asg.required_nv_nr_all[:, 0] / self.asg.required_nv_nr_all[:, 1]

        # 先if判断in_vm和host之间的角度关系，这里的firstKey虚机选择好像和论文里面讲的是反的
        first_key = \
            tan_nv[vmids] >= tan_in_vm \
                if tan_in_vm > tan_host else \
                tan_nv[vmids] <= tan_in_vm
        second_key = (self.asg.init_mapping == self.asg.mapping)[vmids]
        third_key = self.asg.required_nv_nr_all[vmids, 1]
        return vmids[np.lexsort((third_key, second_key, first_key))]

    def choose_hid_balanced(self, vmid: int, hids: np.array) -> int:
        metric_nh = self.count_smaller_vms_nh(vmid, hids)
        idx = np.argmax(metric_nh)
        return hids[idx]

    def choose_vmids_balanced(self, hid: int) -> np.array:
        vmids = self.asg.get_vmids_on_hid(hid)
        first_key = (self.asg.init_mapping == self.asg.mapping)[vmids]
        second_key = self.asg.required_nv_nr_all[vmids, 1]
        return vmids[np.lexsort((second_key, first_key))]

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

    def push_vmid(self, in_vmid: int, hid: int, vmids: np.array, cpu_impt: float, mem_impt: float) -> np.array:
        ejected_vmids = list()
        ejected_vm_nums = []
        for vmid in vmids:
            # 避免成环
            # if self.asg.backup_mapping[vmid] != self.asg.mapping[vmid] or np.any(self.asg.backup_numas[vmid] != self.asg.numas[vmid]):
            #     continue
            ejected_vmids.append(vmid)
            ejected_vm_nums.append(copy.deepcopy(self.asg.numas[vmid]))
            self.asg.exclude(vmid)
            if self.asg.is_feasible(in_vmid, hid): break

        if self.asg.is_feasible(in_vmid, hid):
            self.asg.include(in_vmid, hid)
        else:
            for i, vmid in enumerate(ejected_vmids):
                self.asg.include(vmid, hid, ejected_vm_nums[i])

            return [in_vmid]

        residue = []
        residue_numa = []
        for vmid, numas in zip(ejected_vmids[::-1], ejected_vm_nums[::-1]):
            if self.asg.is_feasible_numa_index(vmid, hid, numas):
                self.asg.include(vmid, hid, numas)
            else:
                residue.append(vmid)
                residue_numa.append(numas)

        candi_degree = np.sum(self.asg.normalize(self.asg.required_nv_nr_all[residue]) * np.array([cpu_impt, mem_impt]))
        target_degree = np.sum(self.asg.normalize(self.asg.required_nv_nr_all[in_vmid] * np.array([cpu_impt, mem_impt])))

        if candi_degree > target_degree:
            self.asg.exclude(in_vmid)
            for vmid, numas in zip(residue, residue_numa):
                self.asg.include(vmid, hid, numas)
            return [in_vmid]
        else:
            return residue
        # return residue


class BalCon2(Algorithm):
    """Reference implementation of SerConFF algorithm """

    def __init__(self, env: Environment, settings: Settings, targetVM: VM) -> None:
        super().__init__(env, settings)
        # 计算host和vm的资源
        self.asg = Assignment(env, settings)
        self.placer = ForceFit(self.asg, max_layer=settings.max_layer)
        self.debug = False
        self.coin = Coin()
        self.target_flavor = targetVM
        self.induced_degree = 1
        # 宿主机已占用内存
        self.initial_memory_nh = np.sum(self.asg.occupied_nh_nr, axis=1)[:, 1].copy()

    def clear_host(self, hid: int) -> np.array:
        vmids = self.asg.get_vmids_on_hid(hid)
        for vmid in vmids:
            # 将vm的mapping设置为-1，相当于加入stack。然后重新计算宿主机上的已占用和剩余资源
            self.asg.exclude(vmid)
        return vmids

    def choose_hid_to_try2(self, hids: np.array, flavor: VM) -> int:
        # 计算需求资源
        required_nh_nr = self.asg.cal_required_nh(flavor)

        local_degree = self.asg.cal_host_required_degree(required_nh_nr[hids], flavor.numa)
        global_degree = -1*self.asg.cal_host_induced_degree(hids, flavor)
        degree = self.induced_degree*(global_degree) + (1/self.induced_degree) * local_degree
        # first_key = np.sum(self.asg.required_nh_nr, axis=1)[hids, Resources.MEM]
        # second_key = np.sum(self.asg.required_nh_nr, axis=1)[hids, Resources.CPU]
        return hids[np.argsort(degree)[0]]

    def choose_hid_to_try(self, hids: np.array, numa_num:int) -> int:
        first_key = self.initial_memory_nh[hids]
        # first_key = [1] * len(hids)

        # occupied_nh_temp = self.asg.normalize(self.asg.occupied_nh_nr)
        # res_sum = np.sum(occupied_nh_temp, axis=(0, 1))
        # weight_sum = (res_sum[Resources.CPU] * occupied_nh_temp[:, :, Resources.CPU] +
        #           res_sum[Resources.MEM] * occupied_nh_temp[:, :, Resources.MEM]) / sum(res_sum)
        # arr_sort = np.sort(weight_sum, axis=1)
        # if numa_num == 0:
        #     numa_num = arr_sort.shape[1]
        # degree = np.sum(arr_sort[:, :numa_num], axis=1)
        #
        # second_key = degree[hids]

        second_key = np.sum(self.asg.occupied_nh_nr, axis=1)[hids, 1]
        return hids[np.lexsort((second_key, first_key))[0]]

    def choose_vms_to_try(self, hid: int) -> np.array:
        # 计算需求资源
        required_nh_nr = self.asg.cal_required_nh(self.target_flavor)
        normalize_required = self.asg.normalize(required_nh_nr)
        impt = np.sum(normalize_required[hid], axis=0) / (np.sum(normalize_required[hid]) + 0.0001)
        cpu_impt, mem_impt = impt[0], impt[1]

        # 对虚机规格筛选，大于flavor的不考虑
        vmids_hid = self.asg.get_vmids_on_hid(hid)
        vms_normalize = self.asg.normalize(self.asg.required_nv_nr_all[vmids_hid])
        vm_flavor = np.sum(vms_normalize * np.array([cpu_impt, mem_impt]), axis=1)
        # vms_normalize[:, Resources.CPU] * cpu_impt + \
        #         vms_normalize[:, Resources.MEM] * mem_impt
        # 虚机计算投影匹配度，计算虚机得分

        target_flavor_norm = self.asg.normalize(np.array([self.target_flavor.cpu, self.target_flavor.mem]))
        smaller_flavor_ids = vm_flavor < np.sum(target_flavor_norm * np.array([cpu_impt, mem_impt]))
        vmids_filter = vmids_hid[smaller_flavor_ids]
        # vmids_filter = vmids_hid
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
        vm_eliminate_list = self.push_vmid(hid, vmids_sort, cpu_impt, mem_impt)

        return vm_eliminate_list

    def log_result(self, result: AttemptResult, hid: int) -> None:
        self.log(f'try host {hid:<5}\t'
                 f'force steps: {self.placer.force_steps_counter:<7}\t'
                 f'{result}')

    def push_vmid(self, hid: int, vmids: np.array, cpu_impt: float, mem_impt: float) -> np.array:
        ejected_vmids = list()
        ejected_vm_nums = []
        for vmid in vmids:
            # if self.asg.backup_mapping[vmid] != self.asg.mapping[vmid] or np.any(self.asg.backup_numas[vmid] != self.asg.numas[vmid]):
            #     continue
            ejected_vmids.append(vmid)
            ejected_vm_nums.append(copy.deepcopy(self.asg.numas[vmid]))
            self.asg.exclude(vmid)
            if self.asg.is_feasible_flavor(hid, self.target_flavor):
                break

        if self.asg.is_feasible_flavor(hid, self.target_flavor):
            self.asg.fill_one(hid, self.target_flavor)
        else:
            for i, vmid in enumerate(ejected_vmids.copy()):
                self.asg.include(vmid, hid, ejected_vm_nums[i])

            return []

        residue = []
        for vmid, numas in zip(ejected_vmids[::-1], ejected_vm_nums[::-1]):
            if self.asg.is_feasible_numa_index(vmid, hid, numas):
                self.asg.include(vmid, hid, numas)
            else:
                residue.append(vmid)

        candi_degree = np.sum(self.asg.normalize(self.asg.required_nv_nr_all[residue]) * np.array([cpu_impt, mem_impt]))
        target_degree = np.sum(self.asg.normalize(np.array([self.target_flavor.cpu, self.target_flavor.mem]) * np.array([cpu_impt, mem_impt])))

        # 如果虚机组合>flavor，则直接restore，因为从backup到host choose_vm这段里面没有对其他host执行include和exclude
        if candi_degree > target_degree:
            self.asg.restore()
            return []
        else:
            return residue

    def solve_(self) -> Mapping:
        # 可选择的目标宿主机id
        allowed_hids = list(self.asg.hids)
        # 未尝试关机的宿主机id
        hosts_to_try = list(self.asg.hids)
        # # 使用指定flavor进行填充
        flavor_count_old = self.asg.fill(list(self.asg.hids), self.target_flavor)

        # flavor_count_old = 0
        while hosts_to_try:
            self.asg.backup()
            self.induced_degree = self.asg.cal_induce_degree(allowed_hids, self.target_flavor, np.array([0, 0]))
            score = self.asg.compute_score()
            # 根据内存进行排序
            # hid = self.choose_hid_to_try(hosts_to_try, self.target_flavor.numa)
            hid = self.choose_hid_to_try2(hosts_to_try, self.target_flavor)
            hosts_to_try.remove(hid)
            allowed_hids.remove(hid)
            # 把选择的宿主机清空虚机，放到stack里面
            vmids = self.choose_vms_to_try(hid)
            # vmids = self.clear_host(hid)
            # 在可选择的目标宿主机上放置虚机
            if len(vmids) == 0:
                place_bool = False
            else:
                place_bool = self.placer.place_vmids(vmids, allowed_hids)

            if place_bool:
                env_new = Environment(self.asg.env.hosts, self.asg.env.vms, Mapping(copy.deepcopy(self.asg.mapping), copy.deepcopy(self.asg.numas)))
                # env_new = copy.deepcopy(self.asg.env)
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

        return self.asg.get_solution()
