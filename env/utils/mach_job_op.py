import numpy as np
MAX = 1e6
AVAILABLE = 0
PROCESSED = 1
COMPLETED = 2
FUTURE = 3


class Machine:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.processed_op_history = []
        self.current_time = 0

    def process_op(self, op_info, setup_time=0):
        machine_avai_time = self.avai_time(setup_time)
        start_time = max(op_info["current_time"], machine_avai_time)
        # assert start_time == op_info["current_time"]
        op_info["start_time"] = start_time
        finish_time = start_time + op_info["process_time"]
        self.processed_op_history.append(op_info)
        self.current_time += op_info["process_time"]
        return start_time, finish_time

    def get_setup_time(self, op, active=False):
        # if not active or len(self.processed_op_history) == 0:
        #     return 0
        # return 10
        if not active or len(self.processed_op_history) == 0:
            return 0
        prev_op_info = self.processed_op_history[-1]
        if prev_op_info["job_type"] == op.job_type:
            return 0
        else:
            return 10

    def avai_time(self, setup_time=0):
        if len(self.processed_op_history) == 0:
            return 0
        else:
            return self.processed_op_history[-1]["start_time"] + \
                self.processed_op_history[-1]["process_time"] + setup_time

    def get_status(self, current_time):
        if current_time >= self.avai_time():
            return AVAILABLE
        else:
            return PROCESSED

    def last_job_id(self):
        return self.processed_op_history[-1]["job_id"]


class Job:
    def __init__(self, job_id, arrival_time, job_type, op_config):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.operations = [
            Operation(
                self.job_id,
                self.arrival_time,
                job_type,
                config) for config in op_config]
        self.due_date = self.arrival_time + 2 * sum([op.process_time for op in self.operations])
        self.op_num = len(op_config)
        self.current_op_id = 0  # ready to be processed
        self.job_type = job_type

    def reset(self):
        self.current_op_id = 0
        for op in self.operations:
            op.reset()

    def current_op(self):
        if self.current_op_id == -1:
            return None
        else:
            return self.operations[self.current_op_id]

    def update_current_op(self, avai_time):
        self.operations[self.current_op_id].avai_time = avai_time

    def next_op(self):
        if self.current_op_id + 1 < len(self.operations):
            self.current_op_id += 1
        else:
            self.current_op_id = -1
        return self.current_op_id

    def done(self):
        if self.current_op_id == -1:
            return True
        else:
            return False

    def remain_process_time(self):
        if self.done():
            return 0
        else:
            rpt = 0
            for i in range(self.current_op_id, len(self.operations)):
                rpt += self.operations[i].process_time
            return rpt


class Operation:
    def __init__(self, job_id, job_arrival_time, job_type, config):
        self.job_id = job_id
        self.op_id = config['id']
        self.machine_id = config['machine_id']
        self.node_id = -1
        self.process_time = config['process_time']
        # the time when the operation is ready
        self.job_arrival_time = job_arrival_time
        if self.op_id == 0:
            self.avai_time = self.job_arrival_time
        else:
            self.avai_time = MAX
        self.start_time = -1  # the time when op processed on machine
        self.finish_time = -1
        self.job_type = job_type
        self.rule_name = None

    def reset(self):
        if self.op_id == 0:
            self.avai_time = self.job_arrival_time
        else:
            self.avai_time = MAX
        self.start_time = -1
        self.finish_time = -1

    def update(self, start_time):
        self.start_time = start_time
        self.finish_time = start_time + self.process_time

    def get_status(self, current_time):
        if self.start_time == -1:
            if current_time >= self.avai_time:
                return AVAILABLE
            else:
                return FUTURE
        else:
            if current_time >= self.finish_time:
                return COMPLETED
            else:
                return PROCESSED
