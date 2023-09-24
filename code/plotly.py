import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from environment import Environment, Settings, Mapping, VM
from global_variables import *


def ploy_host(env: Environment, mapping: Mapping):
    # host容量
    capacity_nh_nr = np.array(
        [[[h.cpu // HOST_NUMA, h.mem // HOST_NUMA] for _ in range(HOST_NUMA)] for h in env.hosts])
    # host已使用
    n_hosts = len(env.hosts)
    occupied_nh_nr = np.zeros((n_hosts, HOST_NUMA, 2))

    required_nv_nr_numa = np.array([[[v.cpu // v.numa, v.mem // v.numa] if numa_id in env.mapping.numas[i] else [0, 0]
                                     for numa_id in range(HOST_NUMA)] for i, v in enumerate(env.vms)])
    np.add.at(occupied_nh_nr, mapping, required_nv_nr_numa)

    # 数据
    data = occupied_nh_nr

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
