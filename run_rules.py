import os
import numpy as np
import random
from random import sample
from params import get_args
from env.env import JSP_Env
from datetime import datetime
from heuristic import *


def ta_gap():
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
    env = JSP_Env(args)
    for action, rule in env.rules:
        total_gap = 0
        total_case_num = 0
        for size in size_list:
            size_gap = 0
            case_num = 0
            lines = open("./JSPLIB/TA/" + size).readlines()
            for line in lines:
                case_num += 1
                line = line.rstrip('\n').split(',')
                instance, op_ms = line[0], int(line[3])
                env.reset()
                avai_ops = env.load_instance("./JSPLIB/instances/" + instance)
                state = env.get_graph_data(args.device)
                while True:
                    state, reward, done, info = env.step(action)
                    if done:
                        makespan = env.get_makespan()
                        size_gap += (makespan - op_ms) / op_ms
                        break
            total_gap += size_gap
            total_case_num += case_num
            print(f"{rule.name}\tsize: {size}\tgap: {round(size_gap / case_num, 3)}")
        print(f"{rule.name}\ttotal gap: {round(total_gap / total_case_num, 3)}")

if __name__ == "__main__":
    args = get_args()
    # ta_gap()
    # # all cases
    args = get_args()
    instance_dir = "JSPLIB/instances"
    result_dir = "agent/Rule/result/instances"
    # instance_dir = "JSPLIB/small_case"
    # # result_dir = "agent/Rule/result/small_case"
    
    # # rule_names = ["CR", "EDD", "FIFO", "LPT", "LS", "MOR", "MRPT", "SPT", "SRPT"]
    rule_names = ["MOR", "FIFO", "SPT"]
    # rule_names = ["MOR"]
    for instance_name in os.listdir(instance_dir):
        for rule_name in rule_names:
            env = JSP_Env(args)
            avai_ops = env.load_instance(os.path.join(instance_dir, instance_name))
            job_num = env.jsp_instance.initial_job_num
            machine_num = env.jsp_instance.machine_num
            tic = datetime.now()
            makespan, tardiness = heuristic_metric(env, avai_ops, rule_name)
            setup_count = env.jsp_instance.setup_count
            toc = datetime.now()
            print(f"{instance_name}\t"
                  f"{job_num}\t"
                  f"{machine_num}\t"
                  f"{rule_name:10}\t"
                  f"{makespan}\t"
                  f"{tardiness}\t"
                  f"{setup_count}\t"
                  f"{round((toc - tic).total_seconds(), 5)}")
            # result_path = os.path.join(
            #     result_dir, f"{instance_name}_{rule_name}.json")
            # env.jsp_instance.logger.save(result_path)