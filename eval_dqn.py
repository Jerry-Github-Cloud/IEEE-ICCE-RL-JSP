import os
import torch
import argparse
from datetime import datetime
from collections import defaultdict
from agent.DQN.agent import DQN_Agent
from env.env import JSP_Env

def eval_dqn(weight_path, instance_path):
    agent = DQN_Agent(args)
    agent.load(weight_path)
    env = JSP_Env(args)
    avai_ops = env.load_instance(instance_path)
    state = env.get_graph_data(args.device)
    tic = datetime.now()
    with torch.no_grad():
        while True:
            action = agent.select_action(state, random=False)
            state, reward, done, info = env.step(action)
            if done:
                break
    makespan = env.get_makespan()
    toc = datetime.now()
    print(
        f"{instance_path}\t"
        f"{weight_path}\t"
        f"{makespan}\t"
        f"{env.rules_count}\t"
        f"{round((toc - tic).total_seconds(), 2)}\t")

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
    # weight_path = "agent/DQN/weight/20230129_142414/DQN_ep5000"
    # instance_dir = "JSPLIB/instances"
    # for instance_name in os.listdir(instance_dir):
    #     instance_path = os.path.join(instance_dir, instance_name)
    #     eval_dqn(weight_path, instance_path)

    weight_dir = "agent/DQN/weight/20230129_142414"
    instance_dir = "JSPLIB/instances"
    for weight_name in os.listdir(weight_dir):
        for instance_name in os.listdir(instance_dir):
            instance_path = os.path.join(instance_dir, instance_name)
            weight_path = os.path.join(weight_dir, weight_name)
            eval_dqn(weight_path, instance_path)
        print()
        