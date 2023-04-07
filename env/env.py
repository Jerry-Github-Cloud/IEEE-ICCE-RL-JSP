import gym
from collections import defaultdict
from multipledispatch import dispatch
from env.utils.instance import JSP_Instance
# from env.utils.instance_kuo_hao import JSP_Instance
from heuristic import *

class JSP_Env(gym.Env):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.jsp_instance = JSP_Instance(args)
        # self.rules = [MOR(), FIFO(), SPT()]
        # self.rules = [EDD(), FIFO(), LOR(), LPT(), LS(), MOR(), MWKR(), SPT(), SRPT()]
        # self.rules = [FIFO(), LS(), MOR(), MWKR(), SPT(), ]
        self.rules = [FIFO(), MOR(), MWKR(), SPT(), ]
        self.rules_count = dict.fromkeys([rule.name for rule in self.rules], 0)

    @dispatch(list, int)
    def step(self, avai_ops, action_idx, rule_name=None):
        self.jsp_instance.assign(avai_ops, action_idx, rule_name)
        avai_ops = self.jsp_instance.current_avai_ops()
        return avai_ops, self.done()

    @dispatch(int)
    def step(self, action):
        avai_ops = self.jsp_instance.current_avai_ops()
        jobs = self.jsp_instance.jobs
        rule = self.rules[action]
        action_idx = rule(avai_ops, jobs)
        self.rules_count[rule.name] += 1
        prev_makespan = self.get_makespan()
        avai_ops, done = self.step(avai_ops, action_idx, rule_name=rule.name)
        state = self.get_graph_data(self.device)
        # reward = -(self.get_makespan() - prev_makespan)
        reward = self.get_reward()
        return state, reward, done, {}
    
    def reset(self):
        self.jsp_instance.reset()
        avai_ops = self.jsp_instance.current_avai_ops()
        return avai_ops
       
    def done(self):
        return self.jsp_instance.done()

    def get_makespan(self):
        return max(m.avai_time() for m in self.jsp_instance.machines)

    def get_mean_flow_time(self):
        pass

    def get_tardiness(self):
        tardiness = 0
        for job in self.jsp_instance.jobs:
            assert job.done(), f"Job{job.job_id} not done, current_op_id: {job.current_op_id}"
            tardiness += max(job.operations[-1].finish_time - job.due_date, 0)
            # print(f"\tJob{job.job_id}"
            #       f"\tDue date: {job.due_date}"
            #       f"\tFinish time: {job.operations[-1].finish_time}")
        return tardiness

    def get_reward(self):
        return -self.get_makespan()
        # return -self.get_tardiness() 
    
    def get_graph_data(self, device):
        return self.jsp_instance.get_graph_data(device)
        
    def load_instance(self, filename):
        self.jsp_instance.load_instance(filename)
        avai_ops = self.jsp_instance.current_avai_ops()
        return avai_ops
