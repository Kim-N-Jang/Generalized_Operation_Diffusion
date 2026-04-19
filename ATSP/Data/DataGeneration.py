import numpy as np
from ATSPGeneration import ATSPGeneration

NumNodes = 10
Size = 10
Seed = 42

def main():
    np.random.seed(Seed)
    Data = []
    for _ in range(Size):
        Data.append(ATSPGeneration(NumNodes))

    SavePath = f'TrainData/ATSPData_NumNodes{NumNodes}Seed{Seed}Size{Size}.npy'
    np.save(SavePath, np.array(Data, dtype=object), allow_pickle=True)

if __name__ == "__main__":
    main()