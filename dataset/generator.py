import os
import numpy as np
import random

target_root = "./dataset/validation"
# sizes = [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20), (50, 15), (50, 20), (100, 20)]
sizes = [(3, 3)]
case_num = 100
op_process_time_range = [1, 100]

def main():
    for n, m in sizes:
        # fp = open(target_root + "/details/" + str(n) + "x" + str(m), 'w')
        for case in range(1, case_num + 1):
            # fp.write(str(n) + "x" + str(m) + "_" + str(case) + "\n")
            fn = f"{n}x{m}_{case:05}"
            instance_fp = open(os.path.join(target_root, "instances", fn), 'w')
            instance_fp.write(str(n) + " " + str(m) + "\n")
            for job in range(1, n + 1):
                m_seq = [i for i in range(m)]
                random.shuffle(m_seq)
                for op_id in range(m):
                    m_id = m_seq[op_id]
                    process_time = np.random.randint(*op_process_time_range)
                    instance_fp.write(str(m_id) + " " + str(process_time) + " ")
                instance_fp.write("\n")

if __name__ == '__main__':
    main()
    