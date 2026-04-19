import numpy as np
import geatpy as ea

# 디코딩 함수: GA(RandomKeys) -> (Makespanm, Schedule) 
def DecodeReadylist(
    RandomKeys,
    ProcTime,
    MachOrder,
    JobIds,
    NumJobs,
    NumMachines,
    OpsPerJob,
    ReturnSchedule=False
):
    SortedIdx = np.argsort(RandomKeys)  
    SortedJobIds = JobIds[SortedIdx]

    JobCounters = [0] * NumJobs          
    MachineEnd = [0] * NumMachines       
    JobEnd = [0] * NumJobs             
    Schedule = []

    for Job in SortedJobIds:
        OpIdx = JobCounters[Job]       
        MachRow = MachOrder[Job]         
        ProcRow = ProcTime[Job]          
        Machine = MachRow[OpIdx] - 1     
        Duration = ProcRow[OpIdx]        

        Start = max(MachineEnd[Machine], JobEnd[Job])  
        End = Start + Duration                           
        # 반환되는 Schedule: (Job, OpIdx, Machine, StartTime, EndTime)
        Schedule.append((Job, OpIdx, Machine, Start, End))

        JobCounters[Job] += 1
        MachineEnd[Machine] = End
        JobEnd[Job] = End

    Makespan = max(End for *_, End in Schedule) if Schedule else 0
    return Makespan, Schedule 

# UpdatedSchedule : 기계 내 작업 순서를 Schedule에 추가
def AddOrderOnMachine(Schedule, NumMachines):
    Schedule = list(Schedule)
    MachineCounters = [0] * NumMachines
    UpdatedSchedule = []
    for Entry in Schedule:
        Job, OpIdx, Machine, Start, End = Entry
        Order = MachineCounters[Machine]  
        UpdatedSchedule.append((Job, OpIdx, Machine, Start, End, Order))
        MachineCounters[Machine] += 1
    return UpdatedSchedule

# GEATpy 기반 JSSP 문제 정의 클래스
class JSSPProblem(ea.Problem):
    def __init__(self, ProcTime, MachOrder):
        self.ProcTime = ProcTime
        self.MachOrder = MachOrder
        self.NumJobs, self.OpsPerJob = ProcTime.shape
        self.TotalOps = self.NumJobs * self.OpsPerJob
        self.NumMachines = np.max(MachOrder)  
        self.JobIds = np.repeat(np.arange(self.NumJobs), self.OpsPerJob)

        Name = 'JSSP_GA'
        M = 1
        MaxOrMins = [1]  # 최소화 문제
        Dim = self.TotalOps
        VarTypes = [0] * self.TotalOps
        Lb = [0] * self.TotalOps
        Ub = [1] * self.TotalOps
        LBin = [1] * self.TotalOps
        UBin = [1] * self.TotalOps
        super().__init__(Name, M, MaxOrMins, Dim, VarTypes, Lb, Ub, LBin, UBin)

    def aimFunc(self, Pop):
        X = Pop.Phen 
        # ObjV = Makespan
        ObjV = [
            DecodeReadylist(Xi, self.ProcTime, self.MachOrder,
                            self.JobIds, self.NumJobs,
                            self.NumMachines, self.OpsPerJob)[0]
            for Xi in X
        ] 
        Pop.ObjV = np.array(ObjV).reshape(-1, 1)

# 주어진 인스턴스에 대해 JSSP GA로 해결하고 스케줄 및 해를 반환
def SolveJSSPInstance(ProcTime, MachOrder, MaxGen=300, PopSize=300):
    Problem = JSSPProblem(ProcTime, MachOrder)
    Algorithm = ea.soea_DE_best_1_L_templet(
        Problem,
        ea.Population(Encoding='RI', NIND=PopSize),
        MAXGEN=MaxGen,
        logTras=0,
        trappedValue=1e-6,
        maxTrappedCount=100
    )
    Res = ea.optimize(Algorithm, verbose=False, drawing=0, outputMsg=False, saveFlag=False)

    BestKeys = Res['Vars'][0]
    Makespan, Schedule = DecodeReadylist(
        BestKeys, ProcTime, MachOrder,
        Problem.JobIds, Problem.NumJobs,
        Problem.NumMachines, Problem.OpsPerJob
    )
    UpdatedSchedule = AddOrderOnMachine(Schedule, Problem.NumMachines) #Schedule에 Machine 내 순서 추가

    return Makespan, Res['executeTime'], UpdatedSchedule, BestKeys

# 하나의 인스턴스를 디코딩 후 adj 행렬 생성
def RunAllInstances(ProcTime, Eligiblity):
    Makespan, _, Schedule, _ = SolveJSSPInstance(ProcTime, Eligiblity)
    Schedule = np.array(Schedule, dtype=np.int32)
    InstanceData = (ProcTime, Eligiblity, Schedule)
    Jobadj, MachineAdj = BuildAdjs(InstanceData)
    return Jobadj, MachineAdj, Makespan

# 전체 작업 수 기준으로 Job-Operation을 Task ID로 변환
def GetTaskId(JobNum, Op, MachineNum):
    return JobNum * MachineNum + Op

# Adj(인접행렬)
# Adj[i, j] = 1 은 j 작업이 선행되어야 i 작업이 수행 가능하다는 의미
def BuildAdjs(InstanceData):
    ProcTime, _, Schedule = InstanceData
    NJobs, NMachines = ProcTime.shape
    NTasks = NJobs * NMachines

    JobAdj = np.eye(NTasks, dtype=np.float32)
    MachineAdj = np.zeros((NTasks, NTasks), dtype=np.float32)

    # JobAdj : 같은 Job 간 선행 연결
    for Job in range(NJobs):
        for Op in range(NMachines - 1):
            JobAdj[GetTaskId(Job, Op + 1, NMachines), GetTaskId(Job, Op, NMachines)] = 1

    # MachineAdj : 같은 M내에서 선행 연결
    Schedule = np.array(Schedule)
    JobOpToTask = {
        (Job, OpIdx): GetTaskId(Job, OpIdx, NMachines)
        for Job in range(NJobs)
        for OpIdx in range(NMachines)
    }

    for Machine in np.unique(Schedule[:, 2]).astype(int):
        MachineOps = Schedule[Schedule[:, 2] == Machine]
        MachineOps = MachineOps[np.argsort(MachineOps[:, 3])]
        for i in range(len(MachineOps) - 1):
            CurJob, CurOp = int(MachineOps[i, 0]), int(MachineOps[i, 1])
            NextJob, NextOp = int(MachineOps[i+1, 0]), int(MachineOps[i+1, 1])
            FromId = JobOpToTask[(CurJob, CurOp)]
            ToId = JobOpToTask[(NextJob, NextOp)]
            MachineAdj[ToId, FromId] = 1

    # CombinedAdj = np.logical_or(JobAdj, MachineAdj).astype(np.float32)
    # np.fill_diagonal(CombinedAdj, 1)
    # CombinedAdj = JobAdj(같은 Job의 선후행 연결) + MachineAdj(같은 Machine내에서의 연결)
    return JobAdj, MachineAdj

