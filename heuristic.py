import os
import numpy as np
import random
from random import sample
from params import get_args

MAX = 1e6


def heuristic_metric(env, avai_ops, rule_name):
    rules = {
        "CR":   CR(),
        "EDD":  EDD(),
        "FIFO": FIFO(),
        "LPT":  LPT(),
        "LS":   LS(),
        "MOR":  MOR(),
        "LRPT": MWKR(),
        "SPT":  SPT(),
        "SRPT": SRPT(),
    }
    while True:
        if rule_name == "Random":
            action_idx = Random(avai_ops)
        elif rule_name == "CR":
            action_idx = rules[rule_name](
                avai_ops,
                env.jsp_instance.jobs,
                env.jsp_instance.current_time)
        else:
            action_idx = rules[rule_name](avai_ops, env.jsp_instance.jobs)
        avai_ops, done = env.step(avai_ops, action_idx, rule_name=rule_name)
        if done:
            return env.get_makespan(), env.get_tardiness()


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


class CR:
    def __init__(self):
        self.name = "CR"

    def __call__(self, avai_ops, jobs, current_time):
        action_idx = -1
        cr = MAX
        for i, op_info in enumerate(avai_ops):
            job = jobs[op_info['job_id']]
            if job.done():
                continue
            ratio = (job.due_date - current_time) / job.remain_process_time()
            if ratio > cr:
                cr = ratio
                action_idx = i
        return action_idx


class EDD:
    def __init__(self):
        self.name = "EDD"

    def __call__(self, avai_ops, jobs):
        action_idx = -1
        edd = MAX
        for i, op_info in enumerate(avai_ops):
            job = jobs[op_info['job_id']]
            if job.done():
                continue
            if edd > job.due_date:
                edd = job.due_date
                action_idx = i
        return action_idx


class MOR:
    def __init__(self):
        self.name = "MOR"

    def __call__(self, avai_ops, jobs):
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


class LOR:
    def __init__(self):
        self.name = "LOR"

    def __call__(self, avai_ops, jobs):
        min_remaining_op = MAX
        action_idx = -1
        for i in range(len(avai_ops)):
            op_info = avai_ops[i]
            job = jobs[op_info['job_id']]
            op = job.operations[op_info['op_id']]
            if len(job.operations) - op.op_id < min_remaining_op:
                min_remaining_op = len(job.operations) - op.op_id
                action_idx = i
        return action_idx


class MWKR:
    def __init__(self):
        self.name = "MWKR"

    def __call__(self, avai_ops, jobs):
        action_idx = -1
        lrpt = -1
        for i, op_info in enumerate(avai_ops):
            job = jobs[op_info['job_id']]
            if job.done():
                continue
            rpt = job.remain_process_time()
            if rpt > lrpt:
                lrpt = rpt
                action_idx = i
        return action_idx


class FIFO:
    def __init__(self):
        self.name = "FIFO"

    def __call__(self, avai_ops, jobs):
        min_avai_time = MAX
        for i in range(len(avai_ops)):
            op_info = avai_ops[i]
            op = jobs[op_info['job_id']].operations[op_info['op_id']]
            if op.avai_time < min_avai_time:
                min_avai_time = op.avai_time
                action_idx = i
        return action_idx


class SPT:
    def __init__(self):
        self.name = "SPT"

    def __call__(self, avai_ops, jobs):
        min_process_time = MAX
        action_idx = -1
        for i in range(len(avai_ops)):
            op_info = avai_ops[i]
            op = jobs[op_info['job_id']].operations[op_info['op_id']]
            if op.process_time < min_process_time:
                min_process_time = op.process_time
                action_idx = i
        return action_idx


class LPT:
    def __init__(self):
        self.name = "LPT"

    def __call__(self, avai_ops, jobs):
        max_process_time = -1
        action_idx = -1
        for i in range(len(avai_ops)):
            op_info = avai_ops[i]
            op = jobs[op_info['job_id']].operations[op_info['op_id']]
            if op.process_time > max_process_time:
                max_process_time = op.process_time
                action_idx = i
        return action_idx


class SRPT:
    def __init__(self):
        self.name = "SRPT"

    def __call__(self, avai_ops, jobs):
        action_idx = -1
        srpt = MAX
        for i, op_info in enumerate(avai_ops):
            job = jobs[op_info['job_id']]
            if job.done():
                continue
            rpt = job.remain_process_time()
            if rpt < srpt:
                srpt = rpt
                action_idx = i
        return action_idx


class LS:
    def __init__(self):
        self.name = "LS"

    def __call__(self, avai_ops, jobs):
        action_idx = -1
        least_slack = MAX
        for i, op_info in enumerate(avai_ops):
            job = jobs[op_info['job_id']]
            if job.done():
                continue
            slack = job.due_date - job.remain_process_time()
            if slack < least_slack:
                least_slack = slack
                action_idx = i
        return action_idx
