import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for RL_GNN_JSP')
    parser.add_argument('--name', type=str, default='Dummy')
    # args for normal setting
    parser.add_argument('--device', type=str, default='cuda:0')
    # args for env
    parser.add_argument('--data_size', type=int, default=10)
    parser.add_argument('--max_process_time', type=int, default=100, help='Maximum Process Time of an Operation')
    # args for RL
    parser.add_argument('--episode', type=int, default=10000000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    # args for network
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_policy_layers', type=int, default=2)
    parser.add_argument('--num_value_layers', type=int, default=3)
    # args for GNN
    parser.add_argument('--GNN_model', type=str, default="GIN")
    parser.add_argument('--GNN_num_layers', type=int, default=3)
    # args for MCTS
    parser.add_argument('--num_simulations', type=int, default=200)
    #parser.add_argument('--c_PUCT', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=5.0)
    parser.add_argument('--worker_num', type=int, default=2)
    
    args = parser.parse_args()
    return args
