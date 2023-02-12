import os
import torch
import argparse
from datetime import datetime
from collections import defaultdict
from agent.SoftDQN.agent import SoftDQNAgent
from env.env import JSP_Env


def eval_dqn(weight_path, instance_path):
    env = JSP_Env(args)
    agent = SoftDQNAgent(args, out_dim=len(env.rules))
    agent.load(weight_path)
    avai_ops = env.load_instance(instance_path)
    job_num = env.jsp_instance.initial_job_num
    machine_num = env.jsp_instance.machine_num
    state = env.get_graph_data(args.device)
    tic = datetime.now()
    while True:
        action = agent.select_action(state, random=False, test_only=True)
        state, reward, done, info = env.step(action)
        if done:
            break

    makespan = env.get_makespan()
    tardiness = env.get_tardiness()
    setup_count = env.jsp_instance.setup_count
    toc = datetime.now()
    print(
        f"{instance_path}\t"
        f"{job_num}\t"
        f"{machine_num}\t"
        f"{weight_path}\t"
        f"{makespan}\t"
        f"{tardiness}\t"
        f"{setup_count}\t"
        f"{env.rules_count}\t"
        f"{round((toc - tic).total_seconds(), 2)}\t")


def eval_ta(weight_path):
    total_gap = 0
    total_case_num = 0
    size_list = os.listdir("./JSPLIB/TA")
    size_list = [
        '15x15',
        '20x15',
        '20x20',
        '30x15',
        '30x20',
        '50x15',
        '50x20',
        '100x20',
    ]
    # size_list = ['15x15', '20x15',]
    for size in size_list:
        size_gap = 0
        case_num = 0
        lines = open("./JSPLIB/TA/" + size).readlines()
        for line in lines:
            case_num += 1
            line = line.rstrip('\n').split(',')
            instance, op_ms = line[0], int(line[3])
            env = JSP_Env(args)
            agent = SoftDQNAgent(args, out_dim=len(env.rules))
            agent.load(weight_path)
            avai_ops = env.load_instance("./JSPLIB/instances/" + instance)
            state = env.get_graph_data(args.device)
            while True:
                action = agent.select_action(
                    state, random=False, test_only=True)
                state, reward, done, info = env.step(action)
                if done:
                    makespan = env.get_makespan()
                    size_gap += (makespan - op_ms) / op_ms
                    break
        total_gap += size_gap
        total_case_num += case_num
        print(f"size: {size}\tgap: {round(size_gap / case_num, 3)}")
    print(f"total gap: {round(total_gap / total_case_num, 3)}")


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
    args = parser.parse_args()

    # weight_path = "agent/SoftDQN/weight/20230212_124705/DQN_ep50"
    # weight_path = "agent/SoftDQN/weight/20230212_124705/DQN_ep230"
    # weight_path = "agent/SoftDQN/weight/20230212_124705/DQN_ep350"
    weight_path = "agent/SoftDQN/weight/20230212_162152/DQN_ep40"
    print(weight_path)
    eval_ta(weight_path)

    # instance_dir = "JSPLIB/instances"
    # for instance_name in os.listdir(instance_dir):
    #     instance_path = os.path.join(instance_dir, instance_name)
    #     eval_dqn(weight_path, instance_path)
    # print()
