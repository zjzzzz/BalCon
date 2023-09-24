import numpy as np
import copy
from typing import List, Optional


def sort_numa_projection_metric(required_nv_nr_3d: np.array, target_host_nh: np.array, first_numa_index: List, latest_numa_index: List) -> list:
    project_len_first, target_norm = cal_projection_length(required_nv_nr_3d, target_host_nh,
                                              first_numa_index)
    project_len_latest, _ = cal_projection_length(required_nv_nr_3d, target_host_nh,
                                               latest_numa_index)
    # 虚机投影得分
    # 均衡度，不均衡的时候需要优先考虑缓解不均衡的VM，均衡的时候固定大于等于规格虚机，优先考虑小规格虚机，优先迁移之前迁过来的虚机
    # 紧缺资源
    vm_sort_metric = np.abs(project_len_first - target_norm) / (project_len_latest+0.1)
    return vm_sort_metric


def cal_projection_length(required_nv_nr_3d: np.array, target_host_nh: np.array, numa_index: List):
    required_nv_nr_3d_temp, target_host_nh_temp = normalize(required_nv_nr_3d, target_host_nh)
    dot_product = np.sum(required_nv_nr_3d_temp[:, numa_index, :] * target_host_nh_temp[numa_index], axis=(1, 2))
    target_norm = np.linalg.norm(target_host_nh_temp[numa_index])
    project_lengths = dot_product / (target_norm + 1)
    return project_lengths, target_norm


def normalize(required_nv_nr_3d: np.array, target_host_nh: np.array) -> (np.array, np.array):
    required_nv_nr_3d_temp = copy.deepcopy(required_nv_nr_3d)
    target_host_nh_temp = copy.deepcopy(target_host_nh)

    max_value = np.max(required_nv_nr_3d_temp, axis=(0, 1))
    min_value = np.min(required_nv_nr_3d_temp, axis=(0, 1))
    required_nv_nr_3d_temp = (required_nv_nr_3d_temp - min_value) / (max_value - min_value + 0.1)
    target_host_nh_temp = (target_host_nh_temp - min_value) / (max_value - min_value + 0.1)

    return required_nv_nr_3d_temp, target_host_nh_temp