import os
import argparse
from datetime import datetime
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from agent.DQN.agent import DQN_Agent
from env.env import JSP_Env


def train_dqn(args):
    env = JSP_Env(args)
    agent = DQN_Agent(args, out_dim=len(env.rules))
    total_steps = 0

    for episode in range(1, args.episode + 1):
        avai_ops = env.reset()
        state = env.get_graph_data(args.device)
        while True:
            if total_steps < args.warmup:
                action = agent.select_action(state, random=True)
            else:
                action = agent.select_action(state, random=False)
            next_state, reward, done, info = env.step(action)
            agent.add_transition((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
        if episode % 100 == 0:
            eval_dqn(agent, episode, "JSPLIB/instances/abz5")
        if episode % 10000 == 0:
            agent.save(os.path.join(weight_dir, f"DQN_ep{episode}"))
        # print(f"makespan: {env.get_makespan()}")


def eval_dqn(agent, episode, instance_path):
    env = JSP_Env(args)
    avai_ops = env.load_instance(instance_path)
    state = env.get_graph_data(args.device)
    while True:
        action = agent.select_action(state, random=False)
        state, reward, done, info = env.step(action)
        if done:
            break
    makespan = env.get_makespan()
    tardiness = env.get_tardiness()
    print(
        f"Episode: {episode}\t"
        f"{instance_path}\t"
        f"{makespan}\t"
        f"{tardiness}\t"
        f"{env.rules_count}\t")
    writer.add_scalar("Eval/Makespan", makespan, episode)
    writer.add_scalar("Eval/Tardiness", tardiness, episode)
    writer.add_scalars("Eval/Rules count", env.rules_count, episode)


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
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    parser.add_argument('--double', action='store_true')
    # arguments for env
    parser.add_argument('--data_size', type=int, default=10)
    parser.add_argument(
        '--max_process_time',
        type=int,
        default=100,
        help='Maximum Process Time of an Operation')
    args = parser.parse_args()
    print(args)
    root_dir = "agent/DQN"
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
