import os
import numpy as np
import random
from random import sample
from params import get_args

MAX = 1e6


def heuristic_makespan(env, avai_ops, rule):
    while True:
        if rule == "Random":
            action_idx = Random(avai_ops)
        elif rule == "MOR":
            action_idx = MOR(avai_ops, env.jsp_instance.jobs)
        elif rule == "FIFO":
            action_idx = FIFO(avai_ops, env.jsp_instance.jobs)
        elif rule == "SPT":
            action_idx = SPT(avai_ops, env.jsp_instance.jobs)
        avai_ops, done = env.step(avai_ops, action_idx)
        if done:
            return env.get_makespan()


def rollout(env, avai_ops):
    epsilon = 0.1
    while True:
        magic_num = random.random()
        if magic_num < epsilon:
            action_idx = Random(avai_ops)
        else:
            action_idx = MOR(avai_ops, env.jsp_instance.jobs)
        avai_ops, done = env.step(avai_ops, action_idx)
        if done:
            return env.get_makespan()


def Random(avai_ops):
    return np.random.choice(len(avai_ops), size=1)[0]


def MOR(avai_ops, jobs):
    max_remaining_op = -1
    action_idx = -1
    for i in range(len(avai_ops)):
        op_info = avai_ops[i]
        job = jobs[op_info['job_id']]
        op = job.operations[op_info['op_id']]
        if len(job.operations) - op.op_id > max_remaining_op:
            max_remaining_op = len(job.operations) - op.op_id
            action_idx = i

    return action_idx


def FIFO(avai_ops, jobs):
    min_avai_time = MAX
    for i in range(len(avai_ops)):
        op_info = avai_ops[i]
        op = jobs[op_info['job_id']].operations[op_info['op_id']]
        if op.avai_time < min_avai_time:
            min_avai_time = op.avai_time
            action_idx = i
    return action_idx


def SPT(avai_ops, jobs):
    min_process_time = MAX
    action_idx = -1
    for i in range(len(avai_ops)):
        op_info = avai_ops[i]
        op = jobs[op_info['job_id']].operations[op_info['op_id']]
        if op.process_time < min_process_time:
            min_process_time = op.process_time
            action_idx = i
    return action_idx
