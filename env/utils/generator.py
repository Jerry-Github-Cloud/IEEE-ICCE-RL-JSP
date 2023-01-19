import numpy as np
import random
#random.seed(6404)
#np.random.seed(6404)


def gen_operations(op_num, machine_num, op_process_time_range):
    op = []
    m_seq = [i for i in range(machine_num)]
    random.shuffle(m_seq)
    for op_id in range(op_num):
        #m_id = np.random.choice(machine_num, size=1)[0]
        m_id = m_seq[op_id]
        process_time = np.random.randint(*op_process_time_range)
        op.append({"id": op_id, "machine_id": m_id, "process_time": process_time})
    return op
