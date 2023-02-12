import json
import yaml
from gym.spaces import Box, Dict, Discrete
from datetime import datetime, timedelta
from collections import defaultdict
import torch

from agent.SoftDQN.soft_dqn import DQN


class SoftDQNAgent:
    def __init__(self, args, out_dim):
        self.total_steps = 0
        self.epsilon = args.eps
        self.args = args
        self.dqn = DQN(args, out_dim)
        self.action_space = Discrete(out_dim)

    def select_action(self, state, random=True, test_only=False):
        self.total_steps += 1
        if random:
            action = self.action_space.sample()
        elif test_only:
            with torch.no_grad():
                action = self.dqn.select_action(state, 0, self.action_space)
        else:
            self.epsilon = max(
                self.epsilon * self.args.eps_decay,
                self.args.eps_min)
            action = self.dqn.select_action(
                state, self.epsilon, self.action_space)
        return action

    def add_transition(self, transition):
        state, action, reward, next_state, done = transition
        # print(f"state: {state}")
        self.dqn.append(state, action, reward, next_state, done)
        # print(f"len(self.dqn._memory): {len(self.dqn._memory)}")
        if self.total_steps >= self.args.warmup:
            self.dqn.update(self.total_steps)

    def save(self, path):
        self.dqn.save(path)

    def load(self, path):
        self.dqn.load(path)
