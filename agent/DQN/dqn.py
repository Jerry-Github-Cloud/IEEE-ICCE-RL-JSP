import argparse
from collections import deque
import itertools
import random
import time
import json

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import HeteroConv, Linear, MLP, GINConv


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size, device):
        transitions = random.sample(self.buffer, batch_size)
        return (x for x in zip(*transitions))

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.num_layers = 3
        self.hidden_dim = 128
        self.convs = torch.nn.ModuleList()
        self.m_trans_fc = Linear(3, 7)
        in_dim = 7
        for _ in range(self.num_layers):
            nn1 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            nn2 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            nn3 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            nn4 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            conv = HeteroConv({
                ('op', 'to', 'op'): GINConv(nn=nn1),
                ('op', 'to', 'm'): GINConv(nn=nn2),
                ('m', 'to', 'op'): GINConv(nn=nn3),
                ('m', 'to', 'm'): GINConv(nn=nn4)
            }, aggr='sum')
            self.convs.append(conv)
            in_dim = self.hidden_dim
        self.job_fc = Linear(self.hidden_dim, 16)
        self.m_fc = Linear(self.hidden_dim, 16)
    
    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict['m'] = self.m_trans_fc(x_dict['m'])
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict['op'] = self.job_fc(x_dict['op'])
        x_dict['m'] = self.m_fc(x_dict['m'])
        x_dict['op'] = global_add_pool(x_dict['op'], None)
        x_dict['m'] = global_add_pool(x_dict['m'], None)
        x = torch.cat((x_dict['op'], x_dict['m']), dim=1)
        return x

class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=30, hidden_dim=128, num_layer=3):
        super(Net, self).__init__()
        self.layer_list = []
        self.gnn = GNN()
        for i in range(num_layer):
            if i == 0:
                self.layer_list.append(nn.Linear(state_dim, hidden_dim))
            elif i == num_layer-1:
                self.layer_list.append(nn.Linear(hidden_dim, action_dim))
            else:
                self.layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.layer_list.append(nn.ReLU())
        self.model = nn.Sequential(*self.layer_list)
        for m in self.model:
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                # torch.nn.init.normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, state):
        embedding = self.gnn(state)
        x = self.model(embedding)
        return x

class DQN:
    def __init__(self, args, out_dim):
        self._behavior_net = Net(state_dim=32, action_dim=out_dim).to(args.device)
        self._target_net = Net(state_dim=32, action_dim=out_dim).to(args.device)

        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr)
        # self._optimizer = torch.optim.RMSprop(self._behavior_net.parameters(), lr=args.lr)
        # self._optimizer = torch.optim.SGD(self._behavior_net.parameters(), lr=args.lr)

        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        self.args = args
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq
        
        self.loss = 0

    def select_action(self, state, epsilon, action_space):
        random_number = random.random()
        if random_number < epsilon:
            action = action_space.sample()
        else:
            actions_pdf = self._behavior_net(state)
            # print(f"actions_pdf: {actions_pdf}")
            action = torch.argmax(actions_pdf)
            action = action.item()
            # print('action:', action)
        return action

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward], next_state,
                            [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        losses = []
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._memory.sample(self.batch_size, self.device)
        for state, action, reward, next_state, done in zip(state_batch, action_batch, reward_batch, next_state_batch, done_batch):
            state, action, reward, next_state, done = state, action[0], reward[0], next_state, done[0]
            data, jsp_feat = state
            q_value = self._behavior_net(state)
            q_value = q_value[:,action]
            
            with torch.no_grad():
                if self.args.double:
                    # print("Double DQN")
                    _, a_next = torch.max(self._behavior_net(next_state), dim=1, keepdim=False)
                    q_next = self._target_net(next_state)[:,a_next][0]
                else:
                    q_next, _ = torch.max(self._target_net(next_state), dim=1, keepdim=False)
                q_target = reward + gamma * q_next * (1 - done)
                q_target = q_target.detach()
            criterion = nn.MSELoss()
            loss = criterion(q_target, q_value)
            losses.append(loss) 
        # gradient clipping
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5) 
        self._optimizer.zero_grad()
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        self.loss = total_loss.item()
        self._optimizer.step()

    def _update_target_network(self):
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=True):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=True):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])

