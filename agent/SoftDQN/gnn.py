import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, MLP, GINConv
from torch_geometric.nn import global_mean_pool, avg_pool


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.num_layers = 3
        self.hidden_dim = 128
        self.convs = torch.nn.ModuleList()
        self.m_trans_fc = Linear(4, 7)
        in_dim = 7
        for _ in range(self.num_layers):
            nn1 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            nn2 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            nn3 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            nn4 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            conv = HeteroConv({
                ('job', 'to', 'job'): GINConv(nn=nn1),
                ('job', 'to', 'm'): GINConv(nn=nn2),
                ('m', 'to', 'job'): GINConv(nn=nn3),
                ('m', 'to', 'm'): GINConv(nn=nn4)
            }, aggr='sum')
            self.convs.append(conv)
            in_dim = self.hidden_dim
        self.op_fc = Linear(self.hidden_dim, 8)
        self.m_fc = Linear(self.hidden_dim, 8)

    def select_action(self, state, epsilon, action_space):
        random_number = random.random()
        state = torch.tensor(state, device=self.device)
        if random_number < epsilon:
            action = action_space.sample()
        else:
            actions_pdf = self._behavior_net(state)
            action = torch.argmax(actions_pdf)
            action = action.item()
            # print('action:', action)
        return action
    
    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict['m'] = self.m_trans_fc(x_dict['m'])
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict['job'] = self.op_fc(x_dict['job'])
        x_dict['m'] = self.m_fc(x_dict['m'])
        x_dict['job'] = global_mean_pool(x_dict['job'], None)
        x_dict['m'] = global_mean_pool(x_dict['m'], None)
        x = torch.cat((x_dict['job'], x_dict['m']), dim=1)
        
        return x
    