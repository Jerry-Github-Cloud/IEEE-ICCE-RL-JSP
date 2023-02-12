import os
import json
import collections
from ortools.sat.python import cp_model


def load_instance(path):
    data = []
    # f = open(os.path.join(instance_dir, filename))
    path = os.path.join(instance_dir, file_name)
    f = open(path)
    line = f.readline()
    while line[0] == '#':
        line = f.readline()
    line = line.split()
    num_job, num_machine = int(line[0]), int(line[1])
    for i in range(num_job):
        job_data = []
        line = f.readline().split()
        for j in range(num_machine):
            machine_id, processing_time = int(
                line[j * 2]), int(line[j * 2 + 1])
            # print('Machine', machine_id, 'processing_time:', processing_time)
            job_data.append((machine_id, processing_time))
        data.append(job_data)
    return data


def solve(file_name, time_limit, num_thread=8, export=False):
    result = []
    # jobs_data = [  # task = (machine_id, processing_time).
    #     [(0, 3), (1, 2), (2, 2)],  # Job0
    #     [(0, 2), (2, 1), (1, 4)],  # Job1
    #     [(1, 4), (2, 3)]  # Job2
    # ]
    jobs_data = load_instance(file_name)
    jobs_due = []
    for job in jobs_data:
        due_date = 0
        for item in job:
            due_date += item[1]
        due_date *= 2
        jobs_due.append(due_date)

    machines_count = 1 + max(item[0] for job in jobs_data for item in job)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(item[1] for job in jobs_data for item in job)

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    makespan_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])

    # Tardiness objective
    tardiness_var = model.NewIntVar(0, horizon, "tardiness")
    for job_id, job in enumerate(jobs_data):
        due = jobs_due[job_id]
        complete = all_tasks[job_id, len(job) - 1].end
        delay_var = model.NewIntVar(0, horizon, f"delay_{job_id}")
        model.AddMaxEquality(delay_var, [(complete - due), 0])
        tardiness_var += delay_var

    model.Minimize(tardiness_var)

    # Creates the solver.
    solver = cp_model.CpSolver()
    # time limit
    solver.parameters.max_time_in_seconds = time_limit
    # number of threads
    solver.parameters.num_workers = num_thread
    # solve
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"{file_name}\t"
              f"{solver.ObjectiveValue()}\t"
              f"{round(solver.WallTime(), 2)}\t"
              f"{bool(status == cp_model.OPTIMAL)}")
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(
                        all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1]))

        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            for assigned_task in assigned_jobs[machine]:
                start = assigned_task.start
                duration = assigned_task.duration
                # Add spaces to output to align columns.
                op_info = {
                    'Order': None,
                    'job_id': assigned_task.job,
                    'op_id': assigned_task.index,
                    'machine_id': machine,
                    'start_time': start,
                    'process_time': duration,
                    'finish_time': start + duration,
                    'job_type': None,
                }
                result.append(op_info)
    else:
        print('No solution found.')
    if export:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        with open(os.path.join(result_dir, f"{file_name}.json"), 'w') as f:
            json.dump(result, f, indent=4)

if __name__ == '__main__':
    # instance_dir = "../JSPLIB/small_case"
    # result_dir = "result"
    # for file_name in os.listdir(instance_dir):
    #     path = os.path.join(instance_dir, file_name)
    #     if not os.path.isfile(path):
    #         continue
    #     time_limit = 6000
    #     solve(file_name, time_limit, export=True)

    instance_dir = "../JSPLIB/instances"
    file_name = "abz5"
    time_limit = 6000
    solve(file_name, time_limit, export=False)
