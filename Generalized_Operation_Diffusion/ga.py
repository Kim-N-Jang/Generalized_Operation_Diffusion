import numpy as np
import geatpy as ea

# 디코딩 함수: GA 염색체(random_keys)를 기반으로 schedule 생성 
# 반환되는 schedule: (job, op_idx, machine, start_time, end_time)
def decode_readylist(
    random_keys,
    proc_time,
    mach_order,
    job_ids,
    num_jobs,
    num_machines,
    ops_per_job,
    return_schedule=False
):
    sorted_idx = np.argsort(random_keys)  
    sorted_job_ids = job_ids[sorted_idx]

    job_counters = [0] * num_jobs          
    machine_end = [0] * num_machines       
    job_end = [0] * num_jobs              
    schedule = []

    for job in sorted_job_ids:
        op_idx = job_counters[job]         
        mach_row = mach_order[job]        
        proc_row = proc_time[job]          
        machine = mach_row[op_idx] - 1     
        duration = proc_row[op_idx]        

        start = max(machine_end[machine], job_end[job])  
        end = start + duration                             

        schedule.append((job, op_idx, machine, start, end))

        job_counters[job] += 1
        machine_end[machine] = end
        job_end[job] = end

    makespan = max(end for *_, end in schedule) if schedule else 0
    return (makespan, schedule) if return_schedule else makespan

# 각 기계 내에서 작업 순서를 기록해주는 함수 (order_on_machine 추가)
def add_order_on_machine(schedule, num_machines):
    schedule = list(schedule)
    machine_counters = [0] * num_machines
    updated_schedule = []
    for entry in schedule:
        job, op_idx, machine, start, end = entry
        order = machine_counters[machine]
        updated_schedule.append((job, op_idx, machine, start, end, order))
        machine_counters[machine] += 1
    return updated_schedule

# GEATpy용 JSSP 문제 정의 클래스
class JSSPProblem(ea.Problem):
    def __init__(self, proc_time, mach_order):
        self.proc_time = proc_time
        self.mach_order = mach_order
        self.num_jobs, self.ops_per_job = proc_time.shape
        self.total_ops = self.num_jobs * self.ops_per_job
        self.num_machines = np.max(mach_order)  
        self.job_ids = np.repeat(np.arange(self.num_jobs), self.ops_per_job)

        name = 'JSSP_GA'
        M = 1
        maxormins = [1]  # 최소화 문제
        Dim = self.total_ops
        varTypes = [0] * self.total_ops
        lb = [0] * self.total_ops
        ub = [1] * self.total_ops
        lbin = [1] * self.total_ops
        ubin = [1] * self.total_ops
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        X = pop.Phen 
        ObjV = [
            decode_readylist(x, self.proc_time, self.mach_order,
                             self.job_ids, self.num_jobs,
                             self.num_machines, self.ops_per_job)
            for x in X
        ]
        pop.ObjV = np.array(ObjV).reshape(-1, 1)

def solve_jssp_instance(proc_time, mach_order, max_gen=300, pop_size=300):
    problem = JSSPProblem(proc_time, mach_order)
    algorithm = ea.soea_DE_best_1_L_templet(
        problem,
        ea.Population(Encoding='RI', NIND=pop_size),
        MAXGEN=max_gen,
        logTras=0,
        trappedValue=1e-6,
        maxTrappedCount=100
    )
    res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False)

    best_keys = res['Vars'][0]
    makespan, schedule = decode_readylist(
        best_keys, proc_time, mach_order,
        problem.job_ids, problem.num_jobs,
        problem.num_machines, problem.ops_per_job,
        return_schedule=True
    )
    updated_schedule = add_order_on_machine(schedule, problem.num_machines)
    return makespan, res['executeTime'], updated_schedule, best_keys

# 전체 dataset에 대해 GA 실행
# 문제 (proc_time, mach_order) -> 정답 (proc_time, mach_order, makespan, schedule)
def run_all_instances(dataset, save_path):
    dataset_GA = []
    for idx, (proc_time, mach_order) in enumerate(dataset):
        print("="*50)
        print(f"🗆️ Instance {idx+1}/{len(dataset)}")

        makespan, exetime, schedule, best_keys = solve_jssp_instance(proc_time, mach_order)
        print(f"✅ Makespan: {makespan}")
        print(f"⏱️  Time: {exetime:.2f}s\n")
        print(schedule)
        print(best_keys)

        schedule_np = np.array(schedule, dtype=np.int32)
        instance_data = (proc_time, mach_order, makespan, schedule_np)
        dataset_GA.append(instance_data)

    np.save(save_path, np.array(dataset_GA, dtype=object))
    print(f"✅ 저장 완료: {save_path}")


# Job마다 모든 Machine에 할당 되야 된다 가정
def build_adjs(sample):
    proc_time, mach_order, makespan, schedule = sample

    n_jobs, n_machines = proc_time.shape
    n_tasks = n_jobs * n_machines

    job_adj = np.eye(n_tasks, dtype=np.float32)
    machine_adj = np.zeros((n_tasks, n_tasks), dtype=np.float32)

    def get_task_id(job, op):
        return job * n_machines + op

    # Job 간 연결
    for job in range(n_jobs):
        for op in range(n_machines - 1):
            job_adj[get_task_id(job, op + 1), get_task_id(job, op)] = 1

    # Machine 간 연결
    schedule = np.array(schedule)
    job_op_to_task = {
        (int(row[0]), int(row[1])): idx for idx, row in enumerate(schedule)
    }

    for machine in np.unique(schedule[:, 2]).astype(int):
        machine_ops = schedule[schedule[:, 2] == machine]
        machine_ops = machine_ops[np.argsort(machine_ops[:, 3])]
        for i in range(len(machine_ops) - 1):
            cur_job, cur_op = int(machine_ops[i, 0]), int(machine_ops[i, 1])
            next_job, next_op = int(machine_ops[i+1, 0]), int(machine_ops[i+1, 1])
            from_id = job_op_to_task[(cur_job, cur_op)]
            to_id = job_op_to_task[(next_job, next_op)]
            machine_adj[to_id, from_id] = 1

    # 최종 adj = job_adj + machine_adj
    combined_adj = np.logical_or(job_adj, machine_adj).astype(np.float32)
    np.fill_diagonal(combined_adj, 1)

    return job_adj, machine_adj, combined_adj

import numpy as np

def generate_adj(dataset, save_path):
    full_dataset = []

    for i, sample in enumerate(dataset):
        print(f"🔧 Processing instance {i+1}/{len(dataset)}")
        job_adj, machine_adj, combined_adj = build_adjs(sample)

        full_data = sample + (job_adj, machine_adj, combined_adj)
        full_dataset.append(full_data)

    np.save(save_path, np.array(full_dataset, dtype=object))
    print(f"✅ 저장 완료: {save_path}")


if __name__ == '__main__':
    N_JOBS_P = 5
    N_MACHINES_P = 3
    SEED = 200
    SIZE = 10

    # dataset[i] = (proc_time, mach_order)
    dataLoaded = np.load(f'./Data/TrainData/GeneratedData_Job{N_JOBS_P}Machine{N_MACHINES_P}Seed{SEED}Size{SIZE}.npy')
    dataset = [(dataLoaded[i][0], dataLoaded[i][1]) for i in range(dataLoaded.shape[0])]
    run_all_instances(dataset,save_path='dataset_GA.npy')

    # dataset_GA[i] = (proc_time, mach_order, makespan, schedule)
    dataLoaded = np.load('dataset_GA.npy', allow_pickle=True)
    dataset = [(dataLoaded[i][0], dataLoaded[i][1],dataLoaded[i][2],dataLoaded[i][3] ) for i in range(dataLoaded.shape[0])]
    generate_adj(dataset, save_path='dataset_GA_adj.npy')





