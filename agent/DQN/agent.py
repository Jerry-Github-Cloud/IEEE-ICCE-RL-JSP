import json
import yaml
from gym.spaces import Box, Dict, Discrete
from datetime import datetime, timedelta
from collections import defaultdict
import torch

from agent.DQN.dqn import DQN


class DQN_Agent:
    def __init__(self, args):
        self.total_steps = 0
        self.epsilon = args.eps
        self.args = args
        self.dqn = DQN(args)
        self.action_space = Discrete(3)

    def select_action(self, state, random=True, test_only=False):
        if random:
            action = self.action_space.sample()
        else:
            if test_only:
                self.epsilon = 0
            else:
                self.epsilon = max(
                    self.epsilon *
                    self.args.eps_decay,
                    self.args.eps_min)
            action = self.dqn.select_action(
                state, self.epsilon, self.action_space)
            # action = self.dqn.select_action(state, 0, self.action_space)
        return action

    def add_transition(self, transition):
        state, action, reward, next_state, done = transition
        # print(f"state: {state}")
        self.dqn.append(state, action, reward, next_state, done)
        if self.total_steps >= self.args.warmup:
            self.dqn.update(self.total_steps)

    def save(self, path):
        self.dqn.save(path)

    def load(self, path):
        self.dqn.load(path)