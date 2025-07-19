import numpy as np
from UniformInstanceGen import UniformInstanceGen

Job = 20
Machine = 10
Low = 1
High = 99
Size = 100
Seed = 200

def main():
    np.random.seed(Seed)
    data = np.array([UniformInstanceGen(n_j=Job, n_m=Machine, low=Low, high= High) for _ in range(Size)])
    np.save(f'TrainData/GeneratedData_Job{Job}Machine{Machine}Seed{Seed}Size{Size}.npy', data)

if __name__ == "__main__":
    main()