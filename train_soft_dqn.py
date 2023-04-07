import os
import ctypes
import argparse
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
from agent.SoftDQN.agent import SoftDQNAgent
from env.env import JSP_Env

instance_dir = "./JSPLIB/instances/"

def train_dqn(args):
    env = JSP_Env(args)
    agent = SoftDQNAgent(args, out_dim=len(env.rules))

    for episode in range(1, args.episode + 1):
        avai_ops = env.reset()
        state = env.get_graph_data(args.device)
        while True:
            if agent.total_steps < args.warmup:
                action = agent.select_action(state, random=True)
            else:
                action = agent.select_action(state, random=False)
            next_state, reward, done, info = env.step(action)
            agent.add_transition((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
        if episode % 10 == 0:
            gap = eval_ta(agent, episode)
            # gap = para_eval_ta(agent, episode)
            print(
                f"Episode: {episode}\t"
                f"Step: {agent.total_steps}\t"
                f"epsilon: {round(agent.epsilon, 3)}\t"
                f"gap: {round(gap, 3)}\t")
            save_thr = 0.19
            if gap <= save_thr:
                agent.save(os.path.join(weight_dir, f"SoftDQN_ep{episode}"))
            eval_dqn(agent, episode, "JSPLIB/instances/ta01")


def eval_dqn(agent, episode, instance_path):
    env = JSP_Env(dqn_args)
    avai_ops = env.load_instance(instance_path)
    state = env.get_graph_data(dqn_args.device)
    while True:
        action = agent.select_action(state, random=False, test_only=True)
        state, reward, done, info = env.step(action)
        if done:
            break
    makespan = env.get_makespan()
    tardiness = env.get_tardiness()
    setup_count = env.jsp_instance.setup_count
    print(
        f"Episode: {episode}\t"
        f"{instance_path}\t"
        f"{makespan}\t"
        f"{tardiness}\t"
        f"{setup_count}\t"
        f"{env.rules_count}\t")
    writer.add_scalar("Eval/Makespan", makespan, episode)
    writer.add_scalar("Eval/Tardiness", tardiness, episode)
    writer.add_scalars("Eval/Rules count", env.rules_count, episode)

def eval_ta(agent, episode):
    total_gap = 0
    total_case_num = 0
    size_list = os.listdir(ta_dir)
    # size_list = ['15x15', '20x15', '20x20', '30x15', '30x20', '50x15', '50x20', '100x20',]
    size_list = ['15x15', '20x15',]
    for size in size_list:
        size_gap = 0
        case_num = 0
        lines = open(os.path.join(ta_dir, size)).readlines()
        for line in lines:
            case_num += 1
            line = line.rstrip('\n').split(',')
            instance, op_ms = line[0], int(line[3])
            env = JSP_Env(dqn_args)
            avai_ops = env.load_instance("./JSPLIB/instances/" + instance)
            state = env.get_graph_data(dqn_args.device)
            while True:
                action = agent.select_action(state, random=False, test_only=True)
                state, reward, done, info = env.step(action)
                if done:
                    size_gap += (env.get_makespan() - op_ms) / op_ms
                    break
        writer.add_scalar(size, size_gap / case_num, episode)
        total_gap += size_gap
        total_case_num += case_num
        # print(f"\tsize: {size}\tcase_num: {case_num}")
    writer.add_scalar("total_TA", total_gap / total_case_num, episode)
    return total_gap / total_case_num


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
                # print(f"{instance}\t{size}\t{makespan}\t{op_ms}\t{round(gap, 3)}")
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


def para_eval_ta(agent, episode):
    all_task = load_task(ta_dir)
    num_worker = len(all_task)
    queue = mp.Queue()
    env = JSP_Env(dqn_args)
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
    writer.add_scalar("total_TA", total_gap.value / total_case_num.value, episode)
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
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--eps_decay', default=0.99, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    # arguments for SoftDQN
    parser.add_argument('--alpha', default=0.5, type=float)
    # arguments for env
    parser.add_argument('--data_size', type=int, default=10)
    parser.add_argument(
        '--max_process_time',
        type=int,
        default=100,
        help='Maximum Process Time of an Operation')
    dqn_args = parser.parse_args()
    print(dqn_args)
    root_dir = "agent/SoftDQN"
    now_str = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    result_dir = os.path.join(root_dir, "result", now_str)
    weight_dir = os.path.join(root_dir, "weight", now_str)
    logdir = os.path.join(root_dir, "log", now_str)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)
    print(logdir)
    ta_dir = "./JSPLIB/TA"
    mp.set_start_method('spawn')
    train_dqn(dqn_args)

