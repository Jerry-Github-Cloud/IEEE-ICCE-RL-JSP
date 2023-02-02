import os
import numpy as np
import random
from random import sample
from params import get_args
from env.env import JSP_Env
from datetime import datetime
from heuristic import heuristic_metric

if __name__ == "__main__":
    # all cases
    args = get_args()
    instance_dir = "JSPLIB/instances"
    result_dir = "agent/Rule/result/instances"
    rule_names = ["CR", "EDD", "FIFO", "LPT", "LS", "MOR", "SPT", "SRPT"]
    for instance_name in os.listdir(instance_dir):
        for rule_name in rule_names:
            env = JSP_Env(args)
            avai_ops = env.load_instance(os.path.join(instance_dir, instance_name))
            tic = datetime.now()
            makespan, tardiness = heuristic_metric(env, avai_ops, rule_name)
            toc = datetime.now()
            print(f"{instance_name}\t"
                  f"{rule_name:10}\t"
                  f"{makespan}\t"
                  f"{tardiness}\t"
                  f"{round((toc - tic).total_seconds(), 2)}")
            result_path = os.path.join(
                result_dir, f"{instance_name}_{rule_name}.json")
            env.jsp_instance.logger.save(result_path)