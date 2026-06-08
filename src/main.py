import heapq
import math
import random
from typing import Dict, List, Optional
import numpy as np


def read_csv(f: str) -> np.ndarray:
    return np.loadtxt(f, dtype=np.float32, delimiter=',')


class Node:
    def __init__(self, node_id: int, vector: np.ndarray, level: int):
        self.id = node_id
        self.vector = vector
        self.level = level


class HNSW:
    def __init__(self, M: int = 16, ef_construction: int = 200, mL: float = None):
        self.M = M                              # 每个节点在 layer>0 的最大邻居数
        self.M_max = M                          # layer>0 的邻居上限
        self.M_max0 = 2 * M                     # layer 0 的邻居上限（论文建议 2M）
        self.ef_construction = ef_construction  # 构建时 beam search 的候选集大小
        self.mL = mL if mL is not None else 1.0 / math.log(M)  # 层级采样归一化因子

        self.nodes: List[Node] = []                      # 所有节点
        self.graphs: List[Dict[int, List[int]]] = []     # graphs[layer][node_id] = [邻居id列表]
        self.ep: Optional[int] = None                    # 全局入口节点 id（始终在最高层）
        self.max_layer: int = -1                         # 当前图的最高层编号

    # ── 内部工具方法 ──────────────────────────────────────────────

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _random_level(self) -> int:
        # 论文 Alg.1 line 4：l ← ⌊-ln(unif(0,1)) × mL⌋
        return int(-math.log(random.uniform(0, 1)) * self.mL)

    def _get_neighbors(self, node_id: int, layer: int) -> List[int]:
        if layer >= len(self.graphs):
            return []
        return list(self.graphs[layer].get(node_id, []))

    def _set_neighbors(self, node_id: int, layer: int, neighbors: List[int]):
        while len(self.graphs) <= layer:
            self.graphs.append({})
        self.graphs[layer][node_id] = neighbors

    # ── Algorithm 1：INSERT ──────────────────────────────────────
    def insert(self, vector: np.ndarray) -> int:
        """论文 Algorithm 1 INSERT：把新向量插入 HNSW 图。"""
        node_id = len(self.nodes)
        l = self._random_level()                 # line 4：随机生成新节点的层级
        self.nodes.append(Node(node_id, vector, l))

        # 第一个节点：直接作为全局入口，无需搜索
        if self.ep is None:
            for layer in range(l + 1):
                self._set_neighbors(node_id, layer, [])
            self.ep = node_id
            self.max_layer = l
            return node_id

        ep = [self.ep]                           # line 2：从全局入口开始
        L = self.max_layer                       # line 3：当前最高层

        # ── Phase 1（line 5-7）：从顶层 L 到 l+1 层，ef=1 贪心下降 ──
        # 这几层比新节点层级高，只需快速找到离 q 最近的入口，不建立连接
        for lc in range(L, l, -1):              # lc = L, L-1, ..., l+1
            W = self.search_layer(vector, ep, ef=1, layer=lc)
            # 只保留 W 中离 q 最近的那个节点作为下一层入口
            ep = [min(W, key=lambda nid: self._distance(vector, self.nodes[nid].vector))]

        # ── Phase 2（line 8-17）：从 min(L,l) 到 0 层，正式建立连接 ──
        for lc in range(min(L, l), -1, -1):     # lc = min(L,l), ..., 1, 0 选择Min 是为了防止新节点层级比当前最高层还高的情况
            # L = 当前图里最高层（插入前已有的）
            # l = 新节点随机抽到的层级
            W = self.search_layer(vector, ep, ef=self.ef_construction, layer=lc)

            # line 10：从候选集 W 中选出要连接的 M 个邻居
            M_lc = self.M_max0 if lc == 0 else self.M
            neighbors = self.select_neighbors(node_id, W, M_lc, lc)

            # line 11：在 lc 层建立双向连接（新节点→邻居 和 邻居→新节点）
            self._set_neighbors(node_id, lc, neighbors) # 新节点 → 邻居（单向）
            for e in neighbors: # 邻居 → 新节点（反向）
                e_neighbors = self._get_neighbors(e, lc)  # line 13：取 e 现有邻居
                if node_id not in e_neighbors:
                    e_neighbors.append(node_id)

                # line 14-16：若 e 的邻居数超过上限，裁剪保留最近的
                M_max = self.M_max0 if lc == 0 else self.M_max
                if len(e_neighbors) > M_max:
                    e_neighbors = self.select_neighbors(e, e_neighbors, M_max, lc)
                self._set_neighbors(e, lc, e_neighbors)

            ep = W  # line 17：把本层搜索结果作为下一层的入口

        # line 18-19：若新节点层级比当前最高层还高，更新全局入口
        if l > L:
            self.ep = node_id
            self.max_layer = l

        return node_id

    # ── Algorithm 2：SEARCH-LAYER ────────────────────────────────
    def search_layer(self, q: np.ndarray, ep: List[int], ef: int, layer: int) -> List[int]:
        """论文 Algorithm 2 SEARCH-LAYER：在第 layer 层从入口 ep 出发，找 q 的 ef 个最近邻。"""
        visited = set(ep)

        # C：待探索的候选节点，每次取最近的扩展（最小堆）
        # W：当前找到的最近 ef 个结果，存负距离方便取最远的（最大堆）
        C = []
        W = []

        # 用入口节点初始化 C 和 W
        for node in ep:
            d = self._distance(q, self.nodes[node].vector)
            heapq.heappush(C, (d, node))        # C 存正距离，堆顶最近
            heapq.heappush(W, (-d, node))       # W 存负距离，堆顶最远

        while C:
            c_dist, c = heapq.heappop(C)        # 取候选集中离 q 最近的节点 c
            f_dist = -W[0][0]                   # W 堆顶是最远的，取出它的距离 f

            if c_dist > f_dist:
                break   # c 比 W 里最差结果还远，继续探索不会有改善，终止

            for e in self._get_neighbors(c, layer):
                if e in visited:
                    continue
                visited.add(e)

                e_dist = self._distance(q, self.nodes[e].vector)
                f_dist = -W[0][0]

                if e_dist < f_dist or len(W) < ef:  # e 更近，或 W 还没装满 ef 个
                    heapq.heappush(C, (e_dist, e))
                    heapq.heappush(W, (-e_dist, e))
                    if len(W) > ef:
                        heapq.heappop(W)            # W 超过 ef 个，踢掉最远的

        # W 里存的是 (-距离, node_id)，只返回 node_id
        result = []
        for _, nid in W:
            result.append(nid)
        return result

    # ── Algorithm 3：SELECT-NEIGHBORS（简单版）───────────────────
    def select_neighbors(self, q_id: int, candidates: List[int], M: int, _layer: int) -> List[int]:
        """论文 Algorithm 3：从候选集中选出离 q 最近的 M 个节点（排序取前 M）。"""
        q_vec = self.nodes[q_id].vector

        # 算每个候选节点到 q 的距离
        scored = []
        for nid in candidates:
            if nid == q_id:
                continue              # 跳过自己
            d = self._distance(q_vec, self.nodes[nid].vector)
            scored.append((d, nid))

        # 按距离从近到远排序，取前 M 个
        scored.sort()
        return [nid for _, nid in scored[:M]]

    # ── Algorithm 4：SELECT-NEIGHBORS-HEURISTIC ──────
    def select_neighbors_heuristic(
        self,
        q_id: int,
        candidates: List[int],
        M: int,
        layer: int,
        extend_candidates: bool = False,    # 是否扩展候选集（极端聚类数据才需要）
        keep_pruned: bool = False,          # 是否补充被丢弃的节点凑满 M 个
    ) -> List[int]:
        """
        论文 Algorithm 4：启发式选邻居。
        与 Algorithm 3 的区别：不只选最近的，而是优先选"能覆盖不同方向"的节点，
        避免所有邻居都挤在同一个方向，让图的连通性更好。
        """
        q_vec = self.nodes[q_id].vector

        # line 2：W 是工作候选集，初始等于输入候选集
        W = list(candidates)

        # line 3-7：可选，把候选集里每个节点的邻居也加进来（扩大搜索范围）
        if extend_candidates:
            for e in list(candidates):
                for e_adj in self._get_neighbors(e, layer):
                    if e_adj not in W:
                        W.append(e_adj)

        R = []      # line 1：最终结果集（已选中的邻居）
        Wd = []     # line 8：被丢弃的候选节点

        # 把 W 转成最小堆，方便每次取出离 q 最近的节点
        W_heap = [(self._distance(q_vec, self.nodes[nid].vector), nid)
                  for nid in W if nid != q_id]
        heapq.heapify(W_heap)

        # line 9：W 非空 且 R 还没装满 M 个
        while W_heap and len(R) < M:
            e_dist, e = heapq.heappop(W_heap)   # line 10：取出离 q 最近的候选 e
            e_vec = self.nodes[e].vector

            # line 11：判断 e 是否比 R 里所有节点都更靠近 q
            # 核心思想：如果 e 比 R 里某个节点更靠近 e 自己，说明方向重叠，丢弃
            closer_to_q_than_to_r = True
            for r in R:
                r_vec = self.nodes[r].vector
                if self._distance(e_vec, r_vec) < e_dist:
                    closer_to_q_than_to_r = False
                    break

            if closer_to_q_than_to_r:
                R.append(e)             # line 12：e 方向独特，加入结果集
            else:
                Wd.append((e_dist, e))  # line 14：e 方向重叠，暂时丢弃

        # line 15-17：如果开启 keep_pruned，把丢弃的节点按距离补进来凑满 M 个
        if keep_pruned:
            Wd.sort()
            for e_dist, e in Wd:
                if len(R) >= M:
                    break
                R.append(e)

        return R

    # ── Algorithm 5：KNN-SEARCH ──────────────────────────────────
    def knn_search(self, q: np.ndarray, K: int, ef: int) -> List[int]:
        """
        论文 Algorithm 5 KNN-SEARCH：在整个图中搜索离 q 最近的 K 个节点。
        逻辑和 INSERT 的 Phase 1 + Phase 2 几乎一样，区别是不建边，直接返回结果。
        """
        ep = [self.ep]      # line 2：从全局入口开始
        L = self.max_layer  # line 3：当前最高层

        # ── Phase 1（line 4-6）：从顶层 L 到第 1 层，ef=1 快速下降 ──
        for lc in range(L, 0, -1):          # lc = L, L-1, ..., 1
            W = self.search_layer(q, ep, ef=1, layer=lc)
            ep = [min(W, key=lambda nid: self._distance(q, self.nodes[nid].vector))]

        # ── Phase 2（line 7）：只在第 0 层用大 ef 精确搜索 ──
        W = self.search_layer(q, ep, ef=ef, layer=0)

        # line 8：从 W 里取最近的 K 个返回
        W_scored = sorted(W, key=lambda nid: self._distance(q, self.nodes[nid].vector))
        return W_scored[:K]


def main():
    print("Reading CSV file...")
    data = read_csv('data/vectors.csv')
    print(data.shape)

    hnsw = HNSW(M=16, ef_construction=200)
    for i, vec in enumerate(data):
        hnsw.insert(vec)
        if i % 100 == 0:
            print(f"Inserted {i} nodes, current max_layer={hnsw.max_layer}")

    print(f"Done. Total nodes: {len(hnsw.nodes)}, enter point: {hnsw.ep}")


if __name__ == "__main__":
    main()
