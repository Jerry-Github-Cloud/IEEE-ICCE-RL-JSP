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

    def process_op(self, op_info):
        machine_avai_time = self.avai_time()
        start_time = max(op_info["current_time"], machine_avai_time)
        assert start_time == op_info["current_time"]
        op_info["start_time"] = start_time
        finished_time = start_time + op_info["process_time"]
        self.processed_op_history.append(op_info)
        return finished_time

    def avai_time(self):
        if len(self.processed_op_history) == 0:
            return 0
        else:
            return self.processed_op_history[-1]["start_time"] + \
                self.processed_op_history[-1]["process_time"]

    def get_status(self, current_time):
        if current_time >= self.avai_time():
            return AVAILABLE
        else:
            return PROCESSED


class Job:
    def __init__(self, job_id, arrival_time, op_config):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.operations = [
            Operation(
                self.job_id,
                self.arrival_time,
                config) for config in op_config]
        self.due_date = self.arrival_time + 2 * sum([op.process_time for op in self.operations])
        self.op_num = len(op_config)
        self.current_op_id = 0  # ready to be processed

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
            return sum(op.process_time for op in self.operations)


class Operation:
    def __init__(self, job_id, job_arrival_time, config):
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
