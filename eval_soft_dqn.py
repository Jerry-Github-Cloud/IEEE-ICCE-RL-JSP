import os
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from agent.SoftDQN.agent import SoftDQNAgent
from env.env import JSP_Env

seed = 1000
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
    total_rules_count = Counter()
    ta_dir = "./JSPLIB/TA"
    size_list = os.listdir(ta_dir)
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
    tic = datetime.now()
    for size in size_list:
        size_gap = 0
        case_num = 0
        lines = open(os.path.join(ta_dir, size)).readlines()
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
                    # print(f"\t{instance}\t{makespan}\t{op_ms}\t{env.rules_count}")
                    size_gap += (makespan - op_ms) / op_ms
                    break
            total_rules_count += Counter(env.rules_count)
            env.jsp_instance.logger.save(os.path.join(result_dir, f"{instance}.json"))
        total_gap += size_gap
        total_case_num += case_num
        print(f"size: {size}\tgap: {round(size_gap / case_num, 3)}")
    toc = datetime.now()
    print(f"total gap: {round(total_gap / total_case_num, 3)}")
    print(f"total rules count: {total_rules_count}")
    print(f"{round((toc - tic).total_seconds(), 2)}")


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
    parser.add_argument(
        '--max_process_time',
        type=int,
        default=100,
        help='Maximum Process Time of an Operation')
    args = parser.parse_args()

    # weight_path = "agent/SoftDQN/weight/20230212_124705/DQN_ep50"
    # weight_path = "agent/SoftDQN/weight/20230212_124705/DQN_ep230"
    # weight_path = "agent/SoftDQN/weight/20230212_124705/DQN_ep350"
    # weight_path = "agent/SoftDQN/weight/20230212_162152/DQN_ep40"
    # weight_path = "agent/SoftDQN/weight/20230212_162152/DQN_ep920"
    
    # eval one case
    # weight_path = "agent/SoftDQN/weight/20230212_181516/DQN_ep600"
    weight_path = "agent/SoftDQN/weight/20230212_181516/DQN_ep2300"
    # weight_path = "agent/SoftDQN/weight/20230212_181516/DQN_ep2320"
    # weight_path = "agent/SoftDQN/weight/20230212_181516/DQN_ep1250"
    # weight_path = "agent/SoftDQN/weight/20230212_181516/DQN_ep3350"
    # weight_path = "agent/SoftDQN/weight/20230213_022608/DQN_ep4280"
    # weight_path = "agent/SoftDQN/weight/20230213_022608/DQN_ep2250"
    # weight_path = "agent/SoftDQN/weight/20230213_010847/DQN_ep4230"
    result_dir = "agent/SoftDQN/result/20230212_181516"
    print(weight_path)
    eval_ta(weight_path)

    # eval all cases
    # weight_dir = "agent/SoftDQN/weight/20230212_181516"
    # weight_dir = "agent/SoftDQN/weight/20230213_010034"
    # weight_dir = "agent/SoftDQN/weight/20230213_010847"
    # weight_dir = "agent/SoftDQN/weight/20230213_022608"
    # weight_dir = "agent/SoftDQN/weight/20230212_172350"
    # weight_dir = "agent/SoftDQN/weight/20230213_174023"
    # weight_dir = "agent/SoftDQN/weight/20230213_174028"
    # weight_dir = "agent/SoftDQN/weight/20230214_103834"
    # for weight_name in os.listdir(weight_dir):
    #     weight_path = os.path.join(weight_dir, weight_name)
    #     print(weight_path)
    #     eval_ta(weight_path)
    #     print()
