import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch

import torch



def ATSPEvaluator(adj_mtrx, edge_feature):

    """

    adj_mtrx: [B, S, N, N]  (A_ij: i -> j 확률)

    edge_feature: [B, N, N] (c_ij: cost)

    return: total_costs [B, S], tours [B, S, N+1]

    """

    B, S, N, _ = adj_mtrx.shape

    device = adj_mtrx.device



    # 거리 broadcast

    dist = edge_feature.unsqueeze(1).expand(B, S, N, N)

    # priority 정의 (log-space 추천)

    priority = torch.log(adj_mtrx + 1e-8)

    # flatten

    flat_priority = priority.reshape(B * S, -1)

    flat_dist = dist.reshape(B * S, -1)

    # sort edges by priority (descending)

    _, sorted_indices = torch.sort(flat_priority, dim=-1, descending=True)



    total_costs = torch.zeros(B * S, device=device)

    all_tours = torch.full((B * S, N + 1), -1, dtype=torch.long, device=device)  # -1로 초기화



    for b_s in range(B * S):

        out_degree = torch.zeros(N, device=device)

        in_degree  = torch.zeros(N, device=device)

        parent = torch.arange(N, device=device)

        next_node = torch.full((N,), -1, dtype=torch.long, device=device)  # next_node[u] = v



        def find(i):

            while parent[i] != i:

                parent[i] = parent[parent[i]]

                i = parent[i]

            return i



        def union(i, j):

            ri, rj = find(i), find(j)

            if ri != rj:

                parent[ri] = rj

                return True

            return False



        edge_count  = 0

        current_cost = 0.0



        for idx in sorted_indices[b_s]:

            u = (idx // N).item()

            v = (idx  % N).item()



            if u == v:

                continue

            if out_degree[u] >= 1 or in_degree[v] >= 1:

                continue



            root_u = find(u)

            root_v = find(v)



            if root_u == root_v and edge_count < N - 1:

                continue



            union(u, v)

            out_degree[u] += 1

            in_degree[v]  += 1

            next_node[u]   = v                          # 엣지 기록

            current_cost  += flat_dist[b_s, idx].item()

            edge_count    += 1



            if edge_count == N:

                break



        total_costs[b_s] = current_cost



        # ── 투어 시퀀스 복원 ──────────────────────────────────────────

        # in_degree == 0 인 노드가 시작점

        start = (in_degree == 0).nonzero(as_tuple=True)[0]

        if len(start) == 1:

            node = start[0].item()

            for step in range(N):

                all_tours[b_s, step] = node

                nxt = next_node[node].item()

                if nxt == -1:

                    break

                node = nxt

            all_tours[b_s, N] = all_tours[b_s, 0]     # 출발점으로 복귀

        # else: 투어 구성 실패 → -1 유지

    tours = all_tours.view(B, S, N + 1)

    return total_costs.view(B, S), tours 
  

def JSSPEvaluator(adj_mat, np_pt, np_machine, num_jobs):
    
    """
    JSSP 스케줄러 (batch 지원, ready set 유지)
    - 머신/Job ID: 1-based 유지
    - adj[i, j]가 클수록 'j 선행 → i 후행' 선호가 큼
    - 콜드스타트(해당 머신의 첫 작업) 점수: adj[i, i] (대각 성분)
    - 각 Job의 작업 수 = 머신 수(num_m), op는 0-based 인덱스
      (Job j의 op 인덱스 범위: [(j-1)*num_m, j*num_m - 1])

    입력
      adj_mat     : (S, n, n)
      np_pt       : (n, 1) 처리시간 (float)
      np_machine  : (n,)  각 op의 머신 ID (항상 1-based)
      num_jobs    : Job 수

    출력
      results = [
        {
          "sample": s,
          "makespan": float,
          "schedule": [{"op":..,"job":..,"machine":..,"start":..,"end":..}, ...]
        }, ...
      ]
    """
    S, n, _ = adj_mat.shape

    pt = np_pt[:, 0]                 
    Machine_ids = np_machine.tolist() 
    num_m = n // num_jobs

    job_ids = [j+1 for j in range(num_jobs) for _ in range(num_m)]

    results = []

    for s in range(S):
        adj = adj_mat[s]

        machine_fin     = [0] * (num_m + 1)
        last_on_machine = [None] * (num_m + 1)
        job_fin         = [0] * (num_jobs + 1)

        ready = set(j * num_m for j in range(num_jobs))
        sched = [None] * n
        scheduled = 0
        while scheduled < n:
            candidates = []
            for op in ready:
                j = job_ids[op]
                m = Machine_ids[op]
                last = last_on_machine[m]

                score = float(adj[op, op] if last is None else adj[op, last])
                est   = max(job_fin[j], machine_fin[m])

                key = (-score, est, op)  
                candidates.append((key, op, score, est, j, m))

             # Op 선택 기준 : 높은 점수 > 낮은 EST > 작은 op
            _, op, score, est, j, m = min(candidates, key=lambda x: x[0])

            start = est
            end   = start + pt[op]
            sched[op] = {"op": op, "job": j, "machine": m, "start": start, "end": end}

            scheduled += 1
            ready.discard(op)
            machine_fin[m] = end
            job_fin[j] = end
            last_on_machine[m] = op

            if (op + 1) % num_m != 0:
                ready.add(op + 1)

        makespan = max(x["end"] for x in sched)
        results.append({
            "sample": s,
            "makespan": makespan,
            "schedule": sorted(sched, key=lambda x: x["start"])
        })

    # Best(makespan 최소) 선택
    best = min(results, key=lambda r: (r["makespan"], r["sample"]))
    return best["makespan"], best["schedule"]