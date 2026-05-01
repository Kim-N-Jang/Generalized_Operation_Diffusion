import numpy as np
# import torch
import subprocess
import tempfile
import shutil
from pathlib import Path


def ATSPGeneration(node_cnt, lkh_path="/home/inuai_11/Generalized_Operation_Diffusion/ATSP/Data/DataGeneration/LKH-3.0.14/LKH"):
    int_min = 0
    int_max = 1000 * 1000
    scaler = 1000 * 1000

    # 1. 2차원 배열 생성 (node, node)
    problem = np.random.randint(low=int_min, high=int_max, size=(node_cnt, node_cnt)).astype(np.float64)
    
    # 2. 대각 성분 0
    np.fill_diagonal(problem, 0)

    while True:
        old_problem = problem.copy()

        # 원본 로직 투영:
        # torch의 problem[:, None, :] -> problem[:, np.newaxis, :] (N, 1, N)
        # torch의 problem[None, :, :].transpose(1, 2) -> np.transpose(problem[None, :, :], (0, 2, 1)) (1, N, N)
        
        term1 = problem[:, np.newaxis, :]  # (N, 1, N)
        term2 = np.transpose(problem[None, :, :], (0, 2, 1))  # (1, N, N)
        
        # np.min(..., axis=2)는 torch의 .min(dim=2)와 동일합니다.
        problem = np.min(term1 + term2, axis=2)

        if np.array_equal(problem, old_problem):
            break

    # 5. LKH3 실행하여 최적해 구하기
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        prob_file = tmp_path / "problem.atsp"
        tour_file = tmp_path / "result.tour"
        par_file = tmp_path / "params.par"

        # TSPLIB 파일 쓰기
        _write_tsplib_atsp_full_matrix(problem, prob_file, name="temp")

        # PAR 파일 쓰기
        lines = [
            f"PROBLEM_FILE = {prob_file}",
            f"TOUR_FILE = {tour_file}",
            f"RUNS = 1",
            f"TRACE_LEVEL = 0"
        ]
        with open(par_file, "w") as f:
            f.write("\n".join(lines))

        # LKH 실행
        subprocess.run([lkh_path, str(par_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 결과 파싱
        tour_raw = _parse_tour_file(tour_file)
        tour0 = _normalize_tour_to_permutation(tour_raw, node_cnt)

    # 6. solution_adj (Adjacency Matrix) 구성
    solution_adj = np.zeros((node_cnt, node_cnt), dtype=np.float32)
    for i in range(node_cnt):
        u = tour0[i]
        v = tour0[(i + 1) % node_cnt]
        solution_adj[u, v] = 1.0

    # objective는 스케일링된 거리 행렬 기준 경로 합

    edge_feature = problem / scaler
    objective = _cycle_cost(edge_feature, tour0)

    return node_cnt, None, edge_feature, solution_adj, objective


# --- 내부 보조 함수들 (기존 코드 활용) ---

def _write_tsplib_atsp_full_matrix(dist_2d, out_path, name):
    n = dist_2d.shape[0]
    with open(out_path, "w") as f:
        f.write(f"NAME: {name}\nTYPE: ATSP\nDIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in dist_2d:
            f.write(" ".join(map(str, row.tolist())) + "\n")
        f.write("EOF\n")


def _parse_tour_file(tour_path):
    with open(tour_path, "r") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if "TOUR_SECTION" in l) + 1
    tour = []
    for l in lines[start:]:
        val = int(l.strip())
        if val == -1: break
        tour.append(val)
    return tour


def _normalize_tour_to_permutation(tour_raw, n):
    seen = set()
    tour0 = []
    for v in tour_raw:
        m = (v - 1) % n
        if m not in seen:
            seen.add(m)
            tour0.append(m)
    return tour0


def _cycle_cost(dist_2d, tour0):
    t = np.array(tour0)
    nxt = np.roll(t, -1)
    return float(dist_2d[t, nxt].sum())