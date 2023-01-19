import argparse
from collections import defaultdict
from agent.DQN.agent import DQN_Agent
from env.env import JSP_Env


def train_dqn(args):
    agent = DQN_Agent(args)
    total_steps = 0

    for episode in range(1, args.episode + 1):
        env = JSP_Env(args)
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
        if episode % 10 == 0:
            eval_dqn(agent, episode, "JSPLIB/instances/abz5")
        # print(f"makespan: {env.get_makespan()}")


def eval_dqn(agent, episode, instance_path):
    env = JSP_Env(args)
    avai_ops = env.load_instance(instance_path)
    state = env.get_graph_data(args.device)
    rule_count = defaultdict(int)
    while True:
        action = agent.select_action(state, random=False)
        rule_count[action] += 1
        state, reward, done, info = env.step(action)
        if done:
            break
    print(
        f"Episode: {episode}\t"
        f"{instance_path}\t"
        f"makespan: {env.get_makespan()}\t"
        f"{rule_count}\t")


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
    train_dqn(args)
