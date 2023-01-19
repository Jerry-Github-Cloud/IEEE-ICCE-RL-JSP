import os
import numpy as np
import random
from random import sample
from params import get_args
from env.env import JSP_Env
from heuristic import heuristic_makespan

if __name__ == "__main__":
    args = get_args()
    instance_dir = "JSPLIB/instances"
    instance_name = "abz5"
    result_dir = "result/instances"
    for rule_name in ["MOR", "FIFO", "SPT"]:
        env = JSP_Env(args)
        avai_ops = env.load_instance(os.path.join(instance_dir, instance_name))
        makespan = heuristic_makespan(env, avai_ops, rule_name)
        print(f"{rule_name}\t{makespan}")
        result_path = os.path.join(
            result_dir, f"{instance_name}_{rule_name}.json")
        env.jsp_instance.logger.save(result_path)