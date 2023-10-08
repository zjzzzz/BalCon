#!/usr/bin/env python3

# Forbid numpy to use more than one thread
# !!! Place before import numpy !!!
import copy
import os
import time
import json
import asyncio
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import pathlib
from datetime import datetime
from time import perf_counter
import numpy as np
import random
import pandas as pd
import threading

from environment import Environment, Settings
from balcon import BalCon
from balcon2 import BalCon2
from findf import FinDf
from findef2 import FinDf2
from firstfit import FirstFit
from algorithm import Solution, Score
from solver import FlowModel, AllocationModel, FlowModelRelaxed
from sercon import SerCon, SerConOriginal
from environment import VM
from assignment  import Assignment
from generate import predefined_flavors, generate_vms


def get_time():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def process_input_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True,
                        help='path to the .json with problem instance')
    parser.add_argument('--output', type=str, required=False,
                        help='path to store results')
    parser.add_argument('--algorithm', type=str, nargs='+', required=True,
                        help=f'algorithms to run, possible algorithms: '
                             f'{[alg.__name__ for alg in registry.values()]}')

    parser.add_argument('--wa', type=float, required=False, default=10,
                        help='weight for savings score')
    parser.add_argument('--wm', type=float, required=False, default=1,
                        help='weight for migration score')
    parser.add_argument('--f', type=float, required=False, default=4000,
                        help='max number of force steps for BalCon algorithm')
    parser.add_argument('--tl', type=float, required=True,
                        help='time limit for each algorithm')
    return parser.parse_args()


def setup_enviroment(args: argparse.Namespace, output_dir: str) -> Environment:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(output_dir + '/log.txt'),
            logging.StreamHandler()
        ]
    )


    env = Environment.load(args.problem)
    assert env.validate_mapping(env.mapping), "initial solution must be feasible"
    env.save(output_dir + '/problem.json')

    return env 


def run_algorithms(algorithms: list, settings: Settings, env: Environment, outputDir: str) -> None:
    solutions = {}
    
    for algorithm in algorithms:
        initial_solution = Solution(mapping=env.mapping)
        logging.info(f'Run {algorithm}...')
        solver = registry[algorithm](env, settings)

        start_time = perf_counter()
        solution = solver.solve()
        solution.elapsed = perf_counter() - start_time
        solutions[algorithm] = solution

    for algorithm, solution in solutions.items():
        score = Score(env, settings)
        logging.info(f'[{algorithm}]')
        if solution.mapping is None or env.validate_mapping(solution.mapping):
            logging.info(f'  Objective: {score.repr_objective(solution)}')
            logging.info(f'  Active: {score.repr_savings_score(solution)}')
            logging.info(f'  Migration: {score.repr_migration_score(solution)}')
            logging.info(f'  Elapsed: {solution.elapsed:.4f}s')
            if solution.status is not None:
                logging.info(f'  Status: {solution.status}')
        else:
            logging.error(f'  INFEASIBLE SOLUTION')
        solution.save(outputDir + f'/{algorithm}-result.json')




def run_from_command_line() -> None:
    args = process_input_arguments()
    
    output_dir = f'runs/run-{get_time()}/'
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)   

    env = setup_enviroment(args, output_dir)
    settings = Settings(
            wa=args.wa,
            wm=args.wm,
            tl=args.tl,
            max_force_steps=args.f,
    )
    
    run_algorithms(args.algorithm, settings, env,  output_dir)
    

def run_example(problem_path: str, algorithm: str, targetVM: VM=None, time_limit:int =60, wa:float = 1, wm:int = 2, max_layer: int = 5):
    env = Environment.load(problem_path)

    settings = Settings(tl=time_limit, wa=wa, wm=wm, max_layer=max_layer)

    t_start = perf_counter()
    if targetVM:
        solution = algorithm(env,settings,targetVM).solve()
    else:
        solution = algorithm(env, settings).solve()
    t = perf_counter() - t_start
    
    score = Score(env, settings)

    new_env = copy.deepcopy(env)
    new_env.mapping = solution.mapping
    if not new_env.validate_mapping(new_env.mapping):
        print("false")

    asg_new = Assignment(new_env, settings)

    init_flavor_num = score.flavor_nums(solution, Assignment(env=env, settings=settings), targetVM)
    final_flavor_num = score.flavor_nums(solution, asg_new, targetVM)

    results = f"""
            Running time:                       {t} sec
            Objective function:                 {score.objective(solution)}
            Number of hosts in initial mapping: {score.savings_score(Solution(mapping=score.env.mapping))}
            Number of hosts in final mapping:   {score.savings_score(solution)}
            Number of released hosts:           {score.savings_score(Solution(mapping=score.env.mapping))-score.savings_score(solution)}
            Number of flavors in initial mapping:   {init_flavor_num}
            Number of flavors in final mapping:   {final_flavor_num}
            Number of increased flavors:   {final_flavor_num - init_flavor_num}
            Amount of migrated memory:          {score.migration_score(solution)} TiB
            Amount of migrated VM: {score.migration_count(solution)}\n"""

    # print(results)
    return init_flavor_num, final_flavor_num, score.migration_score(solution), score.migration_count(solution)


# 定义一个函数，作为线程的目标函数
def task(layer, result_list):
    flavors = predefined_flavors()
    for j, flavor in enumerate(flavors):
        for i in range(100):
            # i = 7
            print("-----------------{}th-layer-{}th-flavor-{}th-num------------------".format(layer, j, i))
            # layer = 1
            # flavor = candi_flavors[1]
            # i = 42
            init_flavor_num, after_flavor_num, migrated_memory, migrated_count = run_example(
                problem_path='./data/synthetic/{}th-numa.json'.format(i), algorithm=registry['balcon2'],
                targetVM=flavor, time_limit=30, wa=1, wm=2, max_layer=layer)
            result_list[layer-1].append(
                [i, layer, flavor.cpu, flavor.mem, flavor.numa, init_flavor_num, after_flavor_num, migrated_memory,
                 migrated_count])


if __name__ == '__main__':
    registry = {
            'balcon': BalCon,
            'balcon2': BalCon2,
            'findf': FinDf,
            'findf2': FinDf2,
            'sercon-modified': SerCon,
            'sercon-original': SerConOriginal,
            'firstfit': FirstFit,
            'flowmodel': FlowModel,
            'flowmodel-relaxed': FlowModelRelaxed,
            'allocationmodel': AllocationModel,
    }

    # flavor = VM
    # flavor.cpu = 8
    # flavor.mem = 16 * 1024
    # flavor.numa = 4
    # run_from_command_line()
    rng = np.random.default_rng(0)

    flavors = predefined_flavors()
    candi_flavor_id = random.sample(range(len(flavors)), 10)
    candi_flavor = [flavors[id] for id in candi_flavor_id]
    candi_exp = random.sample(range(100), 20)
    begin = time.time()
    result = []
    # results = [[] for _ in range(5)]
    # threads = []
    #
    # for layer in range(1, 6):
    #     thread = threading.Thread(target=task, args=(layer, results))
    #     thread.start()
    #     threads.append(thread)
    #
    # # 等待所有线程执行完成
    # for thread in threads:
    #     thread.join()
    #
    # for layer_result in results:
    #     result.extend(layer_result)

    for layer in range(3, 6):
        # layer = 2
        # flavor = VM
        # flavor.cpu = 8
        # flavor.mem = 16 * 1024
        # flavor.numa = 4
        for j, flavor in enumerate(flavors):
            for i in range(100):
            # i = 7
                print("-----------------{}th-layer-{}th-flavor-{}th-num------------------".format(layer, j, i))
                # layer = 1
                # flavor = candi_flavors[1]
                # i = 42
                init_flavor_num, after_flavor_num, migrated_memory, migrated_count = run_example(problem_path='./data/synthetic/{}th-numa.json'.format(i), algorithm=registry['balcon2'], targetVM=flavor, time_limit=30, wa=1, wm=2, max_layer=layer)
                result.append([i, layer, flavor.cpu, flavor.mem, flavor.numa, init_flavor_num, after_flavor_num, migrated_memory, migrated_count])
    end = time.time()
    print("spend {}".format(end-begin))
    #
    result_df = pd.DataFrame(result)
    columns = ["exp_id", "max_layer", "cpu", "mem", "numa", "init_flavor_num", "after_flavor_num", "migrated_memory", "migrated_count"]
    result_df.columns = columns

    result_df["add_flavor_num"] = result_df["after_flavor_num"] - result_df["init_flavor_num"]
    grouped = result_df.groupby("max_layer").mean()

    result_df.to_excel("max_layer_data.xlsx")
    print(grouped)



