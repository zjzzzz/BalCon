import numpy as np
import pathlib
from typing import List, Tuple
from numpy.random import Generator
from itertools import product
from tqdm import trange

from dataclasses import replace
from environment import Environment, Host, VM, Mapping, Settings, Resources
from firstfit import FirstFit
from global_variables import *


def predefined_flavors() -> List[VM]:
    """ Predefined set of flavors used in synthetic problem instances """
    flavor_list = []
    for cpu, ratio, numa in product(FLAVORS_CPU, FLAVORS_RATIO, NUMA):
        if cpu % numa == 0:
            flavor_list.append(VM(cpu, cpu * ratio * 1024, numa))
    return flavor_list


def remove_inactive_vms(env: Environment) -> Environment:
    """ Removes vms mapping == -1 """
    active_vmids = np.flatnonzero([x != -1 for x in env.mapping.mapping])
    # old2new = {old_vmid: new_vmid for new_vmid, old_vmid in enumerate(active_vmids)}
    return Environment(
            hosts=env.hosts,
            vms=[replace(env.vms[old_hid]) for old_hid in active_vmids],
            mapping=Mapping([x for x in env.mapping.mapping if x != -1], [env.mapping.numas[i] for i, x in enumerate(env.mapping.mapping) if x != -1])
    )


def remove_inactive_hosts(env: Environment) -> Environment:
    """ Removes hosts with no VMs from the problem instance """
    active_hids = list(set(env.mapping.mapping))
    old2new = {old_hid: new_hid for new_hid, old_hid in enumerate(active_hids)}
    return Environment(
            hosts=[replace(env.hosts[old_hid]) for old_hid in active_hids],
            vms=env.vms,
            mapping=Mapping([old2new[hid] for hid in env.mapping.mapping], env.mapping.numas)
    )


def determine_host_size(vms: List[VM]) -> Tuple[int, int]:
    """
    Starting from predefined host size adjust host size so that none of
    resources are dominated
    """
    required_nr = np.sum([[vm.cpu, vm.mem] for vm in vms], axis=0)
    n_nr = required_nr / np.array(HOST_SIZE)
    resource = Resources.CPU if n_nr[Resources.CPU] < n_nr[Resources.MEM] else Resources.MEM
    return (np.round(required_nr / n_nr[resource]) // HOST_NUMA + 1) * HOST_NUMA


def generate_vms(n_vms: int, rng: Generator) -> List[VM]:
    """
    Generate random distribution on the set of predefined flavors
    such that small flavors are more likely
    than sample <n_vms> VMs from the distribution
    """
    flavors = predefined_flavors()
    cpu_nf = np.array([flavor.cpu for flavor in flavors])
    weights = -np.log(rng.random(len(flavors))) / cpu_nf
    weights /= np.sum(weights)
    return rng.choice(flavors, n_vms, p=weights)


def generate_instance(n_vms: int, rng: Generator) -> Environment:
    """ Generate problem instances with heavy resource imbalance """
    vms = generate_vms(n_vms, rng)
    cpu, mem = determine_host_size(vms)
    # cpu, mem = HOST_SIZE[0], HOST_SIZE[1]
    hosts = [Host(cpu, mem, HOST_NUMA) for _ in range(n_vms)]

    env = Environment(hosts=hosts, vms=vms, mapping=Mapping.emtpy(n_vms, [vm.numa for vm in vms]))
    scheduler = FirstFit(env, Settings(), sorting=FirstFit.SortingRule.RATIO)
    env.mapping = scheduler.solve().mapping
    assert np.all(np.all(scheduler.asg.occupied_nh_nr >= 0, axis=(1,2)))
    # 因为设定numa后，部分vm单numa上资源需求高，因此无法分配
    env = remove_inactive_vms(env)
    env = remove_inactive_hosts(env)

    assert env.validate_mapping(env.mapping)
    return env


def generate_synthetic_dataset(n_instances: int, path: str) -> None:
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(0)
    for i in trange(n_instances):
        n_vms = rng.integers(N_VMS_RANGE[0], N_VMS_RANGE[1] + 1)
        env = generate_instance(n_vms=n_vms, rng=rng)
        n_vms = len(env.vms)
        problem_path = path / f'{i}th-numa.json'
        env.save(problem_path)


if __name__ == '__main__':
    generate_synthetic_dataset(100, './data/synthetic')
