import numpy as np
import torch
from torch_geometric.data import HeteroData
AVAILABLE = 0
PROCESSED = 1
COMPLETE = 2
FUTURE = 3

class Graph:
    def __init__(self, machine_num):
        self.op_num = 0
        self.op_x = []
        self.m_x = []
        self.op_op_edge_idx = []
        self.op_m_edge_idx = []
        self.m_op_edge_idx = []
        self.m_m_edge_idx = self.fully_connect(0, machine_num - 1)
    
    def get_data(self):
        data = HeteroData()
        data['op'].x = torch.FloatTensor(self.op_x)
        data['m'].x = torch.FloatTensor(self.m_x)
        data['op', 'to', 'op'].edge_index = torch.LongTensor(self.op_op_edge_idx).t().contiguous()
        data['op', 'to', 'm'].edge_index = torch.LongTensor(self.op_m_edge_idx).t().contiguous()
        data['m', 'to', 'op'].edge_index = torch.LongTensor(self.m_op_edge_idx).t().contiguous()
        data['m', 'to', 'm'].edge_index = torch.LongTensor(self.m_m_edge_idx).t().contiguous()
        return data
       
    def add_job(self, job):
        # op to op edge
        self.op_op_edge_idx += self.fully_connect(self.op_num, self.op_num + job.op_num - 1)
        for i in range(job.op_num):
            job.operations[i].node_id = self.op_num
            # for op to machine edge
            self.op_m_edge_idx.append([self.op_num, job.operations[i].machine_id])
            self.m_op_edge_idx.append([job.operations[i].machine_id, self.op_num])
            self.op_num += 1
            
    def update_feature(self, jobs, machines, current_time, max_process_time):
        self.op_x, self.m_x = [], []
        # op feature
        for job in jobs:
            for op in job.operations:
                feat = [0] * 4
                # status
                status = op.get_status(current_time)
                feat[status] = 1
                # time to complete
                if status == PROCESSED:
                    feat.append((op.finish_time - current_time) / max_process_time)
                else:
                    feat.append(0)
                # process time
                feat.append(op.process_time / max_process_time)
                # remaining operations
                feat.append((len(job.operations) - op.op_id) / len(job.operations))
                self.op_x.append(feat) 
                
        # machine feature
        for m in machines:
            feat = [0] * 2
            # status
            status = m.get_status(current_time)
            feat[status] = 1
            # available time
            if status == AVAILABLE:
                feat.append(0)
            else:
                feat.append((m.avai_time() - current_time) / max_process_time)
            
            self.m_x.append(feat)
        
    def fully_connect(self, begin, end):
        edge_idx = []
        for i in range(begin, end + 1):
            for j in range(begin, end + 1):
                if i != j:
                    edge_idx.append([i, j])
        return edge_idx
    
    def print_edge(self, edges):
        for edge in edges:
            print("from {} to {}".format(edge[0], edge[1]))
        print()
    
    def print_structure(self):
        print("op_op_edge_idx:")
        self.print_edge(self.op_op_edge_idx)
        print("op_m_edge_idx:")
        self.print_edge(self.op_m_edge_idx)
        print("m_op_edge_idx:")
        self.print_edge(self.m_op_edge_idx)
        print("m_m_edge_idx:")
        self.print_edge(self.m_m_edge_idx)
      
    def print_feature(self):
        print("\noperation feature:")
        for i in range(len(self.op_x)):
            feat = self.op_x[i]
            print("operation id: {}".format(i), end='\t\t')
            if feat[AVAILABLE] == 1:
                print("status: {}".format("available"), end='\t\t')
            elif feat[PROCESSED] == 1:
                print("status: {}".format("processed"), end='\t\t')
            elif feat[COMPLETE] == 1:
                print("status: {}".format("completed"), end='\t\t')
            else:
                print("status: {}".format("future   "), end='\t\t')
            print("time to complete: {}".format(feat[4]), end='\t\t')
            print("process time: {}".format(feat[5]), end='\t\t')
            print("remaining operations: {}".format(feat[6]))
        print("\nmachine feature:")
        for i in range(len(self.m_x)):
            feat = self.m_x[i]
            print("machine id: {}".format(i), end='\t\t')
            if feat[AVAILABLE] == 1:
                print("status: {}".format("idle     "), end='\t\t')
            elif feat[PROCESSED] == 1:
                print("status: {}".format("processed"), end='\t\t')
            print("available time: {}".format(feat[2]))
        print("\n################################################################")
        