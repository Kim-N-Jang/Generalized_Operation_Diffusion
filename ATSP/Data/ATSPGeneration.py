import numpy as np
from ATSPSolver import SolveATSP

def ATSPGeneration(NumNodes):
    # To do : 삼각 부등식 기반의 EdgeFeature 생성으로 변경. NodeFeature는 필수 아님.
    Low, High = 0, 1
    NoiseScale = 0.1

    # NodeFeature : 각 노드의 (x, y) 좌표
    NodeFeature = np.random.uniform(Low, High, size=(NumNodes, 2))  

    # EdgeFeature : Node Feature 기반의 유클리드 거리 계산 후 노이즈 추가하여 비대칭성 생성
    # 유클리드 거리 행렬
    EdgeFeature = np.sqrt(((NodeFeature[:, None] - NodeFeature[None]) ** 2).sum(-1))

    # 비대칭 노이즈 적용
    EdgeFeature += np.random.uniform(-NoiseScale, NoiseScale, size=(NumNodes, NumNodes)) * EdgeFeature
    np.fill_diagonal(EdgeFeature, 0)

    SolutionAdj, Objective = SolveATSP(EdgeFeature)  

    return NumNodes, NodeFeature, EdgeFeature, SolutionAdj, Objective
