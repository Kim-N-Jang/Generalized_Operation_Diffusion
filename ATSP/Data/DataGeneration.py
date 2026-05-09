import numpy as np
import DataGeneration as DataGenerationModel

num_nodes = 50 #도시 수
batch_size = 10 # 배치 수
seed = 42

def main():
    np.random.seed(seed)
    Data = []
    for _ in range(batch_size):
        Data.append(DataGenerationModel.ATSPGeneration(num_nodes))

    SavePath = f'TrainData/ATSPData_NumNodes{num_nodes}Seed{seed}Size{batch_size}.npy'
    np.save(SavePath, np.array(Data, dtype=object), allow_pickle=True)

if __name__ == "__main__":
    main()