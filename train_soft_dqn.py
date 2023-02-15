import os
import argparse
from datetime import datetime
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
from agent.SoftDQN.agent import SoftDQNAgent
from env.env import JSP_Env

# torch.manual_seed(0)

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
            print(
                f"Episode: {episode}\t"
                f"Step: {agent.total_steps}\t"
                f"epsilon: {round(agent.epsilon, 3)}\t"
                f"gap: {round(gap, 3)}\t")
            save_thr = 0.2
            if gap <= save_thr:
                agent.save(os.path.join(weight_dir, f"DQN_ep{episode}"))
            eval_dqn(agent, episode, "JSPLIB/instances/ta01")
            # print(f"Episode {episode}\tEpsilon:{agent.epsilon}")


def eval_dqn(agent, episode, instance_path):
    env = JSP_Env(args)
    avai_ops = env.load_instance(instance_path)
    state = env.get_graph_data(args.device)
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
    size_list = os.listdir("./JSPLIB/TA")
    # size_list = ['15x15', '20x15', '20x20', '30x15', '30x20', '50x15', '50x20', '100x20',]
    size_list = ['15x15', '20x15',]
    for size in size_list:
        size_gap = 0
        case_num = 0
        lines = open("./JSPLIB/TA/" + size).readlines()
        for line in lines:
            case_num += 1
            line = line.rstrip('\n').split(',')
            instance, op_ms = line[0], int(line[3])
            env = JSP_Env(args)
            avai_ops = env.load_instance("./JSPLIB/instances/" + instance)
            state = env.get_graph_data(args.device)
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
    args = parser.parse_args()
    print(args)
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
    train_dqn(args)
