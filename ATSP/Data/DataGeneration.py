import numpy as np
from UniformInstanceGen import UniformInstanceGen

Job = 3
Machine = 5
Low = 1
High = 99
Size = 10
Seed = 42

def main():
    np.random.seed(Seed)
    data = np.array([UniformInstanceGen(Nj=Job, Nm=Machine, low=Low, high= High) for _ in range(Size)], dtype=object)
    np.save(f'TrainData/GeneratedData_Job{Job}Machine{Machine}Seed{Seed}Size{Size}.npy', data, allow_pickle=True)

if __name__ == "__main__":
    main()