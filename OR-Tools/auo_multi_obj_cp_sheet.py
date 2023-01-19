import json
import yaml
import os
import copy
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from ortools.sat.python import cp_model
from pprint import pprint


def solve(wait_wips_path, run_wips_path, booking_path, result_dir, 
          system_start_time, task_name,
          time_limit, num_thread=8, add_setup_time=True, 
          verbose=False, export=True):
    
    system_start_time_dt = datetime.strptime(system_start_time, '%Y-%m-%d %H:%M:%S')
    result = []
    
    # load AUO job data
    with open(wait_wips_path, 'r') as f:
        wait_data = json.load(f)
    data = wait_data
    num_jobs = len(data)
    num_sheets = 0
    for item in data:
        num_sheets += item['size']

    # global variables
    intervals_per_machines = defaultdict(list)
    presences_per_machines = defaultdict(list)
    starts_per_machines = defaultdict(list)
    ends_per_machines = defaultdict(list)
    model_abbr_per_machines = defaultdict(list)
    ranks_per_machines = defaultdict(list)

    job_starts = {}  # indexed by (job_id).
    job_presences = {}  # indexed by (job_id, alt_id).
    job_ranks = {}  # indexed by (job_id, alt_id).
    job_ends = {}  # indexed by job_id
    
    machine_id_list = ['M0', 'M1', 'M2', 'M3']
    base_time_dt = system_start_time_dt
    if verbose:
        print(f'Start scheduling at {base_time_dt}.')

    with open(booking_path, 'r') as f:
        bookings = json.load(f)
    result += copy.deepcopy(bookings)

    # booking
    for booking in bookings:
        start = datetime.strptime(booking['start_time'], '%Y-%m-%d %H:%M:%S')
        finish = datetime.strptime(booking['finish_time'], '%Y-%m-%d %H:%M:%S')
        booking['start_time'] = int((start - base_time_dt).total_seconds())
        booking['finish_time'] = int((finish - base_time_dt).total_seconds())
    horizon = max([item['max_qtime'] for item in data])
    
    # set horizon
    for item in data:
        max_capacity = max(item['process_time'].values())
        horizon += int(max_capacity)
    for booking in bookings:
        horizon += booking['finish_time'] - booking['start_time']
  
    # set solver
    model = cp_model.CpModel()
    
    # FJSP
    for job_id, item in enumerate(data):
        # capacity
        capacity = item['process_time']
        min_duration = horizon
        max_duration = 0
        num_alternatives = len(capacity)
        for m_id in item['process_time'].keys():
            alt_duration = capacity[m_id]
            min_duration = min(min_duration, alt_duration)
            max_duration = max(max_duration, alt_duration)
        # creat interval, start, end
        suffix = f'_{job_id}'
        min_q = item['min_qtime']
        max_q = item['max_qtime']
        avai_time = min_q
        start = model.NewIntVar(avai_time, horizon, 'start' + suffix)
        duration = model.NewIntVar(int(min_duration), int(max_duration), 'duration' + suffix)
        finish = model.NewIntVar(0, horizon, 'end' + suffix)
        interval = model.NewIntervalVar(start, duration, finish, 'interval' + suffix)
        if num_alternatives > 1:
            alt_presences = []
            for m_id in item['process_time'].keys():
                alt_capacity = int(capacity[m_id])
                alt_suffix = f'_{job_id}_{m_id}'
                presence = model.NewBoolVar('presence' + alt_suffix)
                alt_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                alt_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                alt_interval = model.NewOptionalIntervalVar(
                    alt_start, alt_capacity, alt_end, presence, 'interval' + alt_suffix)
                alt_rank = model.NewIntVar(0, num_jobs, 'rank' + alt_suffix)
                alt_presences.append(presence)
                model.Add(start == alt_start).OnlyEnforceIf(presence)
                model.Add(duration == alt_capacity).OnlyEnforceIf(presence)
                model.Add(finish == alt_end).OnlyEnforceIf(presence)
                
                model.Add(alt_start == 0).OnlyEnforceIf(presence.Not())
                model.Add(alt_end == 0).OnlyEnforceIf(presence.Not())
                # Store alternative variables
                intervals_per_machines[m_id].append(alt_interval)
                starts_per_machines[m_id].append(alt_start)
                ends_per_machines[m_id].append(alt_end)
                presences_per_machines[m_id].append(presence)
                model_abbr_per_machines[m_id].append((model_no, abbr_no))
                ranks_per_machines[m_id].append(alt_rank)
                job_ranks[job_id, m_id] = alt_rank
                job_presences[job_id, m_id] = presence
            model.AddExactlyOne(alt_presences)
        else:
            m_id, _ = capacity[0]
            intervals_per_machines[m_id].append(interval)
            job_presences[job_id, 0] = model.NewConstant(1)
        # Store job's start, end variables
        job_starts[job_id] = start
        job_ends[job_id] = finish
    # Booking the unavailable time.
    for book_id, booking in enumerate(bookings):
        suffix = f'_{book_id}'
        # machine = eqp_id_list.index(booking['selected_eqp_id'])
        m_id = booking['machine_id']
        start = booking['start_time']
        finish = booking['finish_time']
        # duration = model.NewIntVar(0, horizon, 'book_duration' + suffix)
        # interval = model.NewIntervalVar(
        #     start, duration, finish, 'book_interval' + suffix)
        # intervals_per_machines[machine_id].append(interval)
        if verbose:
            print(f'book machine {m_id} on [{start}, {finish}]')
    # no overlap constraint
    for m_id in machine_id_list:
        model.AddNoOverlap(intervals_per_machines[m_id])

    # setup constraint
    setup_list = []
    for m_id in machine_id_list:
        arcs = []
        intervals = intervals_per_machines[m_id]
        machine_starts = starts_per_machines[m_id]
        machine_ends = ends_per_machines[m_id]
        machine_presences = presences_per_machines[m_id]
        machine_ranks = ranks_per_machines[m_id]
        arcs.append([0, 0, model.NewBoolVar('')])
        for i, start in enumerate(machine_starts):
            depot = model.NewBoolVar(f"depot_{m_id}")
            # depot_rank = model.NewIntVar(-1, num_jobs, f"depot_rank_{eqp_id}")
            depot_start = model.NewIntVar(0, horizon, f"depot_start_{m_id}")
            arcs.append([0, i + 1, depot])
            # model.Add(depot_rank == 0).OnlyEnforceIf(depot)
            model.Add(depot_start == 0).OnlyEnforceIf(depot)
            # Final arc from an arc to the dummy node.
            arcs.append([i + 1, 0, model.NewBoolVar('')])
            arcs.append([i + 1, i + 1, machine_presences[i].Not()])
            model.Add(machine_ranks[i] == 0).OnlyEnforceIf(
                machine_presences[i].Not())
            model.Add(machine_ranks[i] != 0).OnlyEnforceIf(
                machine_presences[i])
            for j, _ in enumerate(machine_starts):
                if i == j:
                    continue
                # follow
                follow = model.NewBoolVar(f"{j} follows {i} on {m_id}")
                arcs.append([i + 1, j + 1, follow])
                model.AddImplication(follow, machine_presences[i])
                model.AddImplication(follow, machine_presences[j])
                model.Add(machine_ranks[j] == machine_ranks[i] + 1).OnlyEnforceIf(follow)
                
                # setup time
                setup_time = 0
                model_no_i, abbr_no_i = model_abbr_per_machines[m_id][i]
                model_no_j, abbr_no_j = model_abbr_per_machines[m_id][j]
                if abbr_no_i != abbr_no_j:
                    setup_list.append(follow)
                    if model_no_i != model_no_j:
                        setup_time = 3 * 3600
                    else:
                        setup_time = 1800
                model.Add(machine_ends[i] + setup_time <= machine_starts[j]).OnlyEnforceIf(follow)
        if arcs:
            model.AddCircuit(arcs)
    # makespan
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, list(job_ends.values()))

    # Objective
    obj = makespan
    model.Minimize(obj)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_thread
    printer = cp_model.VarArrayAndObjectiveSolutionPrinter([makespan, over_qtime_sheet_count, min_tact_time_sheet_count, setup_count])

    if verbose:
        status = solver.Solve(model, printer)
    else:
        status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if verbose:
            print('Solution:')
            if status == cp_model.OPTIMAL:
                print('Optimal ', end='')
            else:
                print('Feasible ', end='')
            print(f'Schedule Objective: {solver.ObjectiveValue()}')

        for job_id, item in enumerate(data):
            job_line = f'Job {job_id}: '
            start = solver.Value(job_starts[job_id])
            num_alternatives = len(item['process_time'])
            num_presences = 0
            for m_id in item['process_time'].keys():
                # print(f"\t{job_id} {eqp_id}")
                if solver.Value(job_presences[job_id, m_id]):
                    num_presences += 1
                    item['model_abbr'] = item['model_no'] + '-' + item['abbr_no']
                    alt_capacity = item['process_time'][m_id]
                    item['start_time'] = start
                    item['finish_time'] = start + alt_capacity
                    item['selected_eqp_id'] = m_id
                    item['rank'] = solver.Value(job_ranks[job_id, m_id])
                    min_capacity = min(item['process_time'].values())
                    # if alt_capacity != min_capacity:
                    #     print(f"\tNot min capacity, job_id: {job_id}\teqp_id: {eqp_id}\tcassette_id: {item['cassette_id']}\trank: {item['rank']}")
                    job_line += f"Machine {m_id} "
                    job_line += f"[{start}, {start + alt_capacity}]"
                if verbose:
                    print(job_line)
            assert num_presences == 1, f"job {job_id} num_presences: {num_presences}"
        for job_id, item in enumerate(data):
            int2datetime(base_time_dt, item)
            datetime2str(item)
        result += copy.deepcopy(data)
        result_name_dir = os.path.join(result_dir, task_name)
        if export:
            if not os.path.exists(result_name_dir):
                os.mkdir(result_name_dir)
            with open(os.path.join(result_name_dir, f'or-tools_{args.w0}-{args.w1}-{args.w2}.json'), 'w') as f:
                json.dump(result, f, indent=4)
    else:
        print("Not found")
    print(f"{task_name}\t"
          f"({args.w0},{args.w1},{args.w2})\t"
          f"{len(data_model_abbr)}\t"
          f"{num_jobs}\t"
          f"{num_sheets}\t"
          f"{round(solver.Value(makespan) / 3600, 2)}\t"
          f"{solver.Value(over_qtime_sheet_count)}\t"
          f"{solver.Value(min_tact_time_sheet_count)}\t"
          f"{solver.Value(setup_count)}\t"
          f"{round(solver.WallTime(), 2)}\t"
          f"{status == cp_model.OPTIMAL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w0', default=0, type=float)    # over_qtime
    parser.add_argument('--w1', default=0, type=float)    # select_min_tact
    parser.add_argument('--w2', default=0, type=float)    # has_setup
    parser.add_argument('--w3', default=0, type=float)    # has_setup
    args = parser.parse_args()
    print(args)

    tasks = [
        # {"name": "2022-07-11", "start": "2022-07-11 08:05:56"},
        # {"name": "2022-07-12", "start": "2022-07-12 08:05:56"},
        # {"name": "2022-07-13", "start": "2022-07-13 08:05:56"},
        # {"name": "2022-07-14", "start": "2022-07-14 08:05:56"},
        # {"name": "2022-07-16", "start": "2022-07-16 08:05:57"},
        # {"name": "2022-07-17", "start": "2022-07-17 08:05:57"},
        # {"name": "2022-07-18", "start": "2022-07-18 08:05:58"},
        # {"name": "2022-07-19", "start": "2022-07-19 08:05:58"},
        # {"name": "2022-07-20", "start": "2022-07-20 08:05:58"},
        # {"name": "2022-07-21", "start": "2022-07-21 08:05:59"},
        # {"name": "2022-07-22", "start": "2022-07-22 08:05:58"},
        # {"name": "2022-07-23", "start": "2022-07-23 08:05:59"},
        # {"name": "2022-07-24", "start": "2022-07-24 08:05:59"},
        # {"name": "2022-07-25", "start": "2022-07-25 08:05:59"},
        # {"name": "2022-07-26", "start": "2022-07-26 08:05:59"},
        # {"name": "2022-07-27", "start": "2022-07-27 08:06:00"},
        # {"name": "2022-07-28", "start": "2022-07-28 08:05:00"},
        # {"name": "2022-07-29", "start": "2022-07-29 08:05:00"},
        # {"name": "2022-07-30", "start": "2022-07-30 08:05:00"},
        # {"name": "2022-07-31", "start": "2022-07-31 08:05:00"},
        # {"name": "2022-08-01", "start": "2022-08-01 08:05:01"},
        # {"name": "2022-08-02", "start": "2022-08-02 08:05:01"},
        # {"name": "2022-08-03", "start": "2022-08-03 08:05:01"},
        # {"name": "2022-08-05", "start": "2022-08-05 08:05:02"},
        # {"name": "2022-08-06", "start": "2022-08-06 08:05:02"},
        # {"name": "2022-08-07", "start": "2022-08-07 08:05:02"},
        # {"name": "2022-08-08", "start": "2022-08-08 08:05:02"},
        # {"name": "2022-08-09", "start": "2022-08-09 08:05:03"},
        # {"name": "2022-08-10", "start": "2022-08-10 08:05:03"},
        # {"name": "2022-08-11", "start": "2022-08-11 08:05:03"},
        # {"name": "2022-08-12", "start": "2022-08-12 08:05:04"},
        # {"name": "2022-08-13", "start": "2022-08-13 08:05:04"},
        # {"name": "2022-08-14", "start": "2022-08-14 08:05:04"},
        # # {"name": "2022-08-15", "start": "2022-08-15 08:05:04"},
        # {"name": "2022-08-16", "start": "2022-08-16 08:05:04"},
        # {"name": "2022-08-17", "start": "2022-08-17 08:05:05"},
        # {"name": "2022-08-18", "start": "2022-08-18 08:05:05"},
        # {"name": "2022-08-19", "start": "2022-08-19 08:05:05"},
        # {"name": "2022-08-20", "start": "2022-08-20 08:05:05"},
        # {"name": "2022-08-21", "start": "2022-08-21 08:05:06"},
        {"name": "2022-08-22", "start": "2022-08-22 08:05:06"},
        {"name": "2022-08-23", "start": "2022-08-23 08:05:06"},
        {"name": "2022-08-24", "start": "2022-08-24 08:05:06"},
        {"name": "2022-08-25", "start": "2022-08-25 08:05:06"},
        {"name": "2022-08-26", "start": "2022-08-26 08:05:06"},
        {"name": "2022-08-27", "start": "2022-08-27 08:05:07"},
        {"name": "2022-08-28", "start": "2022-08-28 08:05:07"},
    ]
    # result_dir = "./result_multi_obj"
    # result_dir = "./result"
    # result_dir = "./result_debug"
    result_dir = "./result_mtz_cp"
    # time_limit = 3600 * 24 * 7
    time_limit = 3600 * 24
    # time_limit = 600
    wip_data_dir = "../wip_data"
    booking_dir = "../booking"
    plan_dir = "../plan"
    for item in tasks:
        start, task_name = item['start'], item['name']
        system_start_time_dt = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        wait_wips_path = os.path.join(wip_data_dir, task_name, "odf_wait_wip.json")
        run_wips_path = os.path.join(wip_data_dir, task_name, "odf_run_wip.json")
        booking_path = os.path.join(booking_dir, task_name, "booking.json")
        solve(wait_wips_path, run_wips_path, booking_path, result_dir, 
                start, task_name, 
                time_limit, num_thread=8,
                verbose=False, export=True)
