import os
import ctypes
import torch
import random
import argparse
import numpy as np
from pprint import pprint
from datetime import datetime
import multiprocessing as mp
from collections import defaultdict, Counter
from agent.DQN.agent import DQN_Agent
from env.env import JSP_Env

seed = 1000
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

instance_dir = "./JSPLIB/instances/"


def eval_ta_worker(agent, env, dqn_args, total_gap, total_case_num, tasks):
    for task in tasks:
        instance, size, op_ms = task
        avai_ops = env.load_instance(os.path.join(instance_dir, instance))
        state = env.get_graph_data(dqn_args.device)
        while True:
            action = agent.select_action(
                state, random=False, test_only=True)
            state, reward, done, info = env.step(action)
            if done:
                makespan = env.get_makespan()
                gap = (makespan - op_ms) / op_ms
                with total_gap.get_lock():
                    total_gap.value += gap
                    total_case_num.value += 1
                print(f"{instance}\t{size}\t{makespan}\t{op_ms}\t{round(gap, 3)}")
                break


def load_task(ta_dir):
    size_list = os.listdir(ta_dir)
    all_task = defaultdict(list)
    for size in size_list:
        size_gap = 0
        case_num = 0
        lines = open(os.path.join(ta_dir, size)).readlines()
        for id, line in enumerate(lines):
            case_num += 1
            line = line.rstrip('\n').split(',')
            instance, num_job, num_machine, op_ms = line[0], int(
                line[1]), int(
                line[2]), int(
                line[3])
            all_task[id].append(tuple([instance, size, op_ms]))
    return all_task


def eval_ta_main(weight_path, all_task):
    num_worker = len(all_task)
    queue = mp.Queue()
    env = JSP_Env(dqn_args)
    agent = DQN_Agent(dqn_args, out_dim=len(env.rules))
    agent.load(weight_path)
    total_gap = mp.Value(ctypes.c_double, 0.0)
    total_case_num = mp.Value(ctypes.c_int, 0)
    workers = [
        mp.Process(
            target=eval_ta_worker,
            args=(
                agent,
                env,
                dqn_args,
                total_gap,
                total_case_num,
                all_task[id])) for id in range(num_worker)]
    tic = datetime.now()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    toc = datetime.now()
    print(f"{round((toc - tic).total_seconds(), 2)}\t")
    return total_gap.value / total_case_num.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    # arguments for DQN
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=100000, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=.01, type=float)
    parser.add_argument('--eps', default=0.0, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    parser.add_argument('--double', action='store_true')
    parser.add_argument(
        '--max_process_time',
        type=int,
        default=100,
        help='Maximum Process Time of an Operation')
    dqn_args = parser.parse_args()

    # weight_path = "agent/SoftDQN/weight/20230215_211557/DQN_ep2040"
    # weight_path = "agent/SoftDQN/weight/20230212_181516/DQN_ep2300"
    # weight_path = "agent/SoftDQN/weight/20230215_211557/DQN_ep2040"
    mp.set_start_method('spawn')
    weight_path = "agent/SoftDQN/weight/20230215_225054/DQN_ep4020"
    ta_dir = "./JSPLIB/TA"
    
    all_task = load_task(ta_dir)
    gap = eval_ta_main(weight_path, all_task)
    print(gap)
