import numpy as np
import math


def set_host_num(numa_bool):
    if numa_bool:
        return 4
    else:
        return 1

NUMA_BOOL = True
HOST_SIZE = (152*3, 1008 * 1024)
FLAVORS_CPU = (1, 2, 4, 6, 8, 12, 16, 24, 32)
FLAVORS_RATIO = (1, 2, 4)
HOST_NUMA = set_host_num(NUMA_BOOL)
HOST_NUM = 100
RESOURCE_NUM = 2
# 判断是2的幂
assert (HOST_NUMA > 0) and ((HOST_NUMA & (HOST_NUMA - 1)) == 0)

HOST_NUMAS = list(np.arange(1, HOST_NUMA+1))
NUMA = [2 ** i for i in range(int(math.log2(HOST_NUMA)) + 1)]
N_VMS_RANGE = (500, 5000)

FST_RES = 1
SEC_RES = 0
ALPHA = 0.95
