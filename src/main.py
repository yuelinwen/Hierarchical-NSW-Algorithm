import heapq
import math
import random
from typing import Dict, List, Optional, Set, Tuple
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
        for lc in range(min(L, l), -1, -1):     # lc = min(L,l), ..., 1, 0
            W = self.search_layer(vector, ep, ef=self.ef_construction, layer=lc)

            # line 10：从候选集 W 中选出要连接的 M 个邻居
            M_lc = self.M_max0 if lc == 0 else self.M
            neighbors = self.select_neighbors(node_id, W, M_lc, lc)

            # line 11：在 lc 层建立双向连接（新节点→邻居 和 邻居→新节点）
            self._set_neighbors(node_id, lc, neighbors)
            for e in neighbors:
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
        visited: Set[int] = set(ep)

        # C：候选集（最小堆），堆顶是离 q 最近的候选，决定下一步往哪扩展
        C: List[Tuple[float, int]] = []
        # W：结果集（最大堆，存负距离），堆顶是当前结果里离 q 最远的，用于剪枝
        W: List[Tuple[float, int]] = []

        for node in ep:                          # line 1-3：用入口节点初始化 C 和 W
            d = self._distance(q, self.nodes[node].vector)
            heapq.heappush(C, (d, node))
            heapq.heappush(W, (-d, node))

        while C:                                 # line 4：候选集非空就继续
            c_dist, c = heapq.heappop(C)         # line 5：取候选集中离 q 最近的节点 c
            f_dist = -W[0][0]                    # line 6：结果集中离 q 最远节点的距离 f

            if c_dist > f_dist:
                break  # line 7-8：c 都比 W 里最差结果还远，继续无意义，终止

            for e in self._get_neighbors(c, layer):   # line 9：遍历 c 的所有邻居
                if e in visited:
                    continue                     # line 10：跳过已访问节点
                visited.add(e)                   # line 11：标记为已访问

                e_dist = self._distance(q, self.nodes[e].vector)
                f_dist = -W[0][0]                # line 12：重新取 W 中最远距离

                if e_dist < f_dist or len(W) < ef:    # line 13：e 更近，或 W 还没满
                    heapq.heappush(C, (e_dist, e))    # line 14：e 加入候选集
                    heapq.heappush(W, (-e_dist, e))   # line 15：e 加入结果集
                    if len(W) > ef:
                        heapq.heappop(W)         # line 16-17：踢掉最远的，保持 W 大小 ≤ ef

        return [nid for _, nid in W]             # 返回结果集中所有节点 id

    # ── Algorithm 3：SELECT-NEIGHBORS（简单版）───────────────────
    def select_neighbors(self, q_id: int, candidates: List[int], M: int, _layer: int) -> List[int]:
        """
        论文 Algorithm 3 SELECT-NEIGHBORS-SIMPLE：
        从候选集中选出离 q 最近的 M 个节点（排序取前 M）。
        """
        q_vec = self.nodes[q_id].vector
        scored = [
            (self._distance(q_vec, self.nodes[nid].vector), nid)
            for nid in candidates
            if nid != q_id
        ]
        scored.sort()
        return [nid for _, nid in scored[:M]]


def main():
    print("Reading CSV file...")
    data = read_csv('data/vectors.csv')

    hnsw = HNSW(M=16, ef_construction=200)
    for i, vec in enumerate(data):
        hnsw.insert(vec)
        if i % 100 == 0:
            print(f"Inserted {i} nodes, current max_layer={hnsw.max_layer}")

    print(f"Done. Total nodes: {len(hnsw.nodes)}, enter point: {hnsw.ep}")


if __name__ == "__main__":
    main()
