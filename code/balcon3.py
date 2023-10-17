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
        heapq.heappush(self.vmids, (-1*self.asg.size_nv[vmid], vmid))
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
    def __init__(self, asg: Assignment, targetVM: VM, induced_degree: float, induced_bool: bool, max_layer=5):
        self.asg = asg
        self.coin = Coin()
        self.force_steps_counter = 0
        self.alpha = 0.95
        self.induced_degree = induced_degree
        self.induced_bool = induced_bool
        self.max_layer = max_layer
        self.target_flavor=targetVM

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
                if stash.vm_layers[vmid] >= self.max_layer:
                    return False

                # return False
                hids_filtered = self.filter_hosts(vmid, hids)
                # 对剩余宿主机按flavor需求资源排序，排序靠前的宿主机，需求资源更多，即剩余资源更少
                # hids_sort_index = self.choose_hid_to_try(list(hids_filtered))

                # 针对目标虚机，求宿主机上需求资源，按照flavor进行标准化，
                hids_sort_index = self.choose_hid_to_try2(list(hids_filtered), vmid)
                hids_sort = np.array(hids)[hids_sort_index]

                required_nh_nv = self.asg.required_nv_nr[vmid] - self.asg.remained_nh_nr
                required_nh_nv[required_nh_nv <= 0] = 0
                normalize_required = self.asg.normalize(required_nh_nv)

                # 通过虚机的迁移，把每次需要迁移的虚机规格逐渐减小，用更小的虚机去填充碎片资源
                # 比如放进去一个4c4g的虚机，但是换出来1c2g和2c2g的虚机，代表宿主机上新增了1c的资源使用。

                swap_migrate_bool = False
                for hid in hids_sort:
                    # self.asg.backup_temp()
                    # layers_temp = copy.deepcopy(stash.vm_layers)
                    # 有宿主机能通过swap把vmid放进去
                    if swap_migrate_bool:
                        break

                    impt = np.sum(normalize_required[hid], axis=0) / np.sum(normalize_required[hid])
                    cpu_impt, mem_impt = impt[0], impt[1]

                    # 迁移虚机筛选和排序
                    vmids_migrate = self.choose_vms(hid, vmid, required_nh_nv, cpu_impt, mem_impt)

                    # 迁出部分虚机以放入vmid，若无法实现则跳过这台宿主机
                    residue = self.push_vmid(vmid, hid, vmids_migrate, cpu_impt, mem_impt)
                    if len(residue) == 1 and residue[0] == vmid:
                        continue
                    else:
                        swap_migrate_bool = True
                        stash.add(residue, stash.vm_layers[vmid])
                        for vmid_res in residue:
                            if stash.vm_layers[vmid_res] > self.max_layer:
                                return False
                                # self.asg.restore_temp()
                                # stash.vm_layers = layers_temp
                                # continue

                # 如果所有宿主机上都放不下vmid
                if not swap_migrate_bool:
                    return False

        return stash.is_empty()

    def choose_vms(self, hid: int, in_vmid: int, required_nh_nv: np.array, cpu_impt: float, mem_impt: float) -> np.array:
        # 对虚机规格筛选，大于vmid的不考虑
        vmids_hid = self.asg.get_vmids_on_hid(hid)
        vms_normalize = self.asg.normalize(self.asg.required_nv_nr_all[vmids_hid])
        vm_flavor = np.sum(vms_normalize * np.array([cpu_impt, mem_impt]), axis=1)
        # vms_normalize[:, Resources.CPU] * cpu_impt + \
        #         vms_normalize[:, Resources.MEM] * mem_impt

        target_vm_norm = self.asg.normalize(self.asg.required_nv_nr_all[in_vmid])
        smaller_flavor_ids = vm_flavor < np.sum(target_vm_norm * np.array([cpu_impt, mem_impt]))
        vmids_filter = vmids_hid[smaller_flavor_ids]
        # vmids_filter = vmids_hid

        # 虚机numa指标排序
        vm_local_metric = self.cal_vm_local_metric(vmids_filter, required_nh_nv[hid],
                                                   cpu_impt, mem_impt, self.asg.env.vms[in_vmid].numa)

        flavor_key = vm_flavor[smaller_flavor_ids]
        migrate_key = (self.asg.init_mapping == self.asg.mapping)[vmids_filter]
        # smaller_flavor = np.any(self.asg.required_nv_nr_all[vmids_hid] > self.asg.required_nv_nr_all[vmid],
        #                             axis=1)
        # 对虚机进行逆序排序（优先选择迁过来的虚机）
        vm_sort_ids = np.lexsort((migrate_key, vm_local_metric, flavor_key))
        vmids_migrate = vmids_filter[vm_sort_ids]

        return vmids_migrate

    def choose_hid_to_try(self, hids: np.array) -> np.array:
        required_nh_nr = self.asg.cal_required_nh(self.target_flavor)
        required_nh_nr_temp = self.asg.normalize(required_nh_nr)
        res_sum = np.sum(required_nh_nr_temp, axis=(0, 1))
        weight_sum = (res_sum[Resources.CPU] * required_nh_nr_temp[:, :, Resources.CPU] +
                  res_sum[Resources.MEM] * required_nh_nr_temp[:, :, Resources.MEM]) / sum(res_sum)
        # arr_sort = np.sort(weight_sum, axis=1)
        degree = np.sum(weight_sum, axis=1)

        second_key = degree[hids]
        # 把vm优先放到需求资源多的宿主机上
        return np.argsort(second_key)[::-1]

    def choose_hid_to_try2(self, hids: np.array, in_vmid: int) -> np.array:
        required_nh_nr = self.asg.cal_required_nh(self.asg.env.vms[in_vmid])
        required_nh_nr_norm = np.sum(required_nh_nr[hids] / self.asg.required_nv_nr_all[in_vmid], axis=2)
        second_key = np.sum(np.sort(required_nh_nr_norm, axis=1)[:self.asg.env.vms[in_vmid].numa])

        # 把vm优先放到需求资源小的宿主机上
        return np.argsort(second_key)

    def classify(self, hids: List[int], stash: Stash, vmid: int) -> Situation:
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
        required_after_migrate = required_host_nh - required_nv_nr_3d
        required_after_migrate[required_after_migrate < 0] = 0

        normalize_nh_before = self.asg.normalize(required_host_nh)
        normalize_before_degree = normalize_nh_before[:, Resources.CPU] * cpu_impt + normalize_nh_before[:, Resources.MEM] * mem_impt
        before_degree = np.sum(np.sort(normalize_before_degree)[:numa_num])

        normalize_nh_after = self.asg.normalize(required_after_migrate)
        normalize_after_degree = normalize_nh_after[:, :, Resources.CPU] * cpu_impt + normalize_nh_after[:, :, Resources.MEM] * mem_impt
        after_degree = np.sum(np.sort(normalize_after_degree, axis=1)[:, :numa_num], axis=1)
        vm_local_metric = np.exp(after_degree - before_degree)

        # vm_migrate_cost = np.sum(self.asg.normalize(self.asg.required_nv_nr_all[vmids]) * np.array([cpu_impt, mem_impt]), axis=1)
        # vm_local_metric = vm_local_metric*vm_migrate_cost

        return vm_local_metric

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
        for vmid, numas in zip(ejected_vmids, ejected_vm_nums):
            if self.asg.is_feasible_numa_index(vmid, hid, numas):
                self.asg.include(vmid, hid, numas)
            else:
                residue.append(vmid)
                residue_numa.append(numas)

        candi_degree = np.sum(self.asg.normalize(self.asg.required_nv_nr_all[residue]) * np.array([cpu_impt, mem_impt]))
        target_degree = np.sum(self.asg.normalize(self.asg.required_nv_nr_all[in_vmid] * np.array([cpu_impt, mem_impt])))

        residue_cpu = np.sum(self.asg.required_nv_nr_all[residue, Resources.CPU])
        residue_mem = np.sum(self.asg.required_nv_nr_all[residue, Resources.MEM])
        target_cpu = self.asg.required_nv_nr_all[in_vmid][Resources.CPU]
        target_mem = self.asg.required_nv_nr_all[in_vmid][Resources.MEM]
        # if residue_cpu > target_cpu or residue_mem > target_mem or (residue_cpu == target_cpu and residue_mem == target_mem):
        #     self.asg.exclude(in_vmid)

        if candi_degree > target_degree:
            self.asg.exclude(in_vmid)
            for vmid, numas in zip(residue, residue_numa):
                self.asg.include(vmid, hid, numas)
            return [in_vmid]
        else:
            return residue
        # return residue


class BalCon3(Algorithm):
    """Reference implementation of SerConFF algorithm """

    def __init__(self, env: Environment, settings: Settings, targetVM: VM, induced_degree: float = 1, induced_bool: bool=True) -> None:
        super().__init__(env, settings)
        # 计算host和vm的资源
        self.asg = Assignment(env, settings)
        self.placer = ForceFit(self.asg, targetVM, induced_degree, induced_bool, max_layer=settings.max_layer)
        self.debug = False
        self.target_flavor = targetVM
        self.induced_degree = induced_degree
        self.induced_bool = induced_bool
        # 宿主机已占用内存
        self.initial_memory_nh = np.sum(self.asg.occupied_nh_nr, axis=1)[:, 1].copy()

    def choose_hid_to_try(self, hids: np.array) -> int:
        required_nh_nr = self.asg.cal_required_nh(self.target_flavor)
        required_nh_nr_temp = self.asg.normalize(required_nh_nr)
        res_sum = np.sum(required_nh_nr_temp, axis=(0, 1))
        weight_sum = (res_sum[Resources.CPU] * required_nh_nr_temp[:, :, Resources.CPU] +
                  res_sum[Resources.MEM] * required_nh_nr_temp[:, :, Resources.MEM]) / sum(res_sum)
        # arr_sort = np.sort(weight_sum, axis=1)
        degree = np.sum(weight_sum, axis=1)

        second_key = degree[hids]
        # 需求资源最少的host
        return hids[np.argsort(second_key)[0]]

    def choose_vms_to_try(self, hid: int, flavor_num: int) -> np.array:
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

        target_flavor_norm = self.asg.normalize(np.array([flavor_num * self.target_flavor.cpu, flavor_num * self.target_flavor.mem]))
        smaller_flavor_ids = vm_flavor < np.sum(target_flavor_norm * np.array([cpu_impt, mem_impt]))
        vmids_filter = vmids_hid[smaller_flavor_ids]
        # vmids_filter = vmids_hid
        if len(vmids_filter) == 0:
            return []

        # 虚机迁移后，判断宿主机上需求资源变化，加权求和，考虑numa
        vm_local_metric = self.placer.cal_vm_local_metric(vmids_filter, required_nh_nr[hid], cpu_impt, mem_impt, self.target_flavor.numa)

        flavor_key = vm_flavor[smaller_flavor_ids]
        migrate_key = (self.asg.init_mapping == self.asg.mapping)[vmids_filter]
        # 对虚机进行排序（优先选择迁过来的虚机）
        vm_sort_ids = np.lexsort((migrate_key, vm_local_metric, flavor_key))
        vmids_sort = vmids_filter[vm_sort_ids]

        # 按顺序选择虚机，直至覆盖required_nh_nr
        vm_eliminate_list = self.push_vmid(hid, vmids_sort, cpu_impt, mem_impt, flavor_num)

        return vm_eliminate_list

    def log_result(self, result: AttemptResult, hid: int) -> None:
        self.log(f'try host {hid:<5}\t'
                 f'force steps: {self.placer.force_steps_counter:<7}\t'
                 f'{result}')

    def push_vmid(self, hid: int, vmids: np.array, cpu_impt: float, mem_impt: float, flavor_num: int) -> np.array:
        ejected_vmids = list()
        ejected_vm_numas = []
        vmids_temp = []
        vmids_numas_temp = []

        flavor_count = 0
        for vmid in vmids:
            if flavor_count >= flavor_num:
                break
            # if self.asg.backup_mapping[vmid] != self.asg.mapping[vmid] or np.any(self.asg.backup_numas[vmid] != self.asg.numas[vmid]):
            #     continue
            vmids_temp.append(vmid)
            vmids_numas_temp.append(copy.deepcopy(self.asg.numas[vmid]))
            self.asg.exclude(vmid)
            if self.asg.is_feasible_flavor(hid, self.target_flavor):
                ejected_vmids.append(vmids_temp.copy())
                ejected_vm_numas.append(vmids_numas_temp.copy())
                vmids_temp = []
                vmids_numas_temp = []

                self.asg.fill_one(hid, self.target_flavor)
                flavor_count += 1

        ejected_vmids.append(vmids_temp)
        ejected_vm_numas.append(vmids_numas_temp)

        for i, vmid in enumerate(ejected_vmids[flavor_count].copy()):
            self.asg.include(vmid, hid, ejected_vm_numas[flavor_count][i])

        if flavor_count == 0:
            return []

        ejected_vmids_flatten = [vmid for temp_vmids in ejected_vmids[:flavor_count] for vmid in temp_vmids]
        ejected_vm_numas_flatten = [numa for temp_numas in ejected_vm_numas[:flavor_count] for numa in temp_numas]

        residue = []
        residue_numa = []
        # for count in range(flavor_count):

        #  改成正序，
        for vmid, numas in zip(ejected_vmids_flatten, ejected_vm_numas_flatten):
            if self.asg.is_feasible_numa_index(vmid, hid, numas):
                self.asg.include(vmid, hid, numas)
            else:
                residue.append(vmid)
                residue_numa.append(numas)

        # 选择的虚机组合要求不超过flavor，相当于剪枝
        # 去除试验下
        candi_degree = np.sum(self.asg.normalize(self.asg.required_nv_nr_all[residue]) * np.array([cpu_impt, mem_impt]))
        target_degree = np.sum(self.asg.normalize(np.array([flavor_count*self.target_flavor.cpu, flavor_count*self.target_flavor.mem]) * np.array([cpu_impt, mem_impt])))

        residue_cpu = np.sum(self.asg.required_nv_nr_all[residue, Resources.CPU])
        residue_mem = np.sum(self.asg.required_nv_nr_all[residue, Resources.MEM])
        target_cpu = self.target_flavor.cpu
        target_mem = self.target_flavor.mem

        # 如果虚机组合>flavor，则直接restore，因为从backup到host choose_vm这段里面没有对其他host执行include和exclude
        # if residue_cpu > target_cpu or residue_mem > target_mem or (
        #         residue_cpu == target_cpu and residue_mem == target_mem):

        # if candi_degree > target_degree:
        #     self.asg.restore()
        #     return []
        # else:
        #     return residue

        return residue

    def clear_host(self, hid: int) -> np.array:
        vmids = self.asg.get_vmids_on_hid(hid)
        for vmid in vmids:
            # 将vm的mapping设置为-1，相当于加入stack。然后重新计算宿主机上的已占用和剩余资源
            self.asg.exclude(vmid)
        return vmids


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
            if self.induced_bool:
                self.induced_degree = self.asg.cal_induce_degree(allowed_hids, self.target_flavor, np.array([0, 0]))
            # score = self.asg.compute_score()
            # 根据内存进行排序
            # hid = self.choose_hid_to_try(hosts_to_try, self.target_flavor.numa)
            hid = self.choose_hid_to_try(hosts_to_try)
            hosts_to_try.remove(hid)
            allowed_hids.remove(hid)
            # 把选择的宿主机清空虚机，放到stack里面
            # opt_flavor_num = np.max(np.sum(self.asg.remained_nh_nr[hid], axis=0) // np.array([self.target_flavor.cpu,
            #                                                        self.target_flavor.mem]))
            #
            # opt_flavor_num += 1
            opt_flavor_num = 1
            vmids = self.choose_vms_to_try(hid, opt_flavor_num)
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
                    hosts_to_try.append(hid)
                    allowed_hids.append(hid)
            else:
                result = AttemptResult.FAIL

            if result != AttemptResult.SUCCESS:
                allowed_hids.append(hid)
                self.asg.restore()
            self.log_result(result, hid)

        return self.asg.get_solution()
