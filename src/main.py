import heapq
import math
import random
from typing import List, Optional


class HNSW:
    """
    Hierarchical Navigable Small World graph for approximate nearest-neighbor search.
    Implements Algorithms 1-5 from the original HNSW paper (Malkov & Yashunin, 2018).
    """

    def __init__(
        self,
        M: int = 16,
        Mmax: int = None,
        Mmax0: int = None,
        efConstruction: int = 200,
        mL: float = None,
        use_heuristic: bool = True,
    ):
        self.M = M
        self.Mmax = Mmax if Mmax is not None else M
        self.Mmax0 = Mmax0 if Mmax0 is not None else M * 2
        self.efConstruction = efConstruction
        self.mL = mL if mL is not None else 1.0 / math.log(M)
        self.use_heuristic = use_heuristic

        self.enter_point: Optional[int] = None
        self.max_layer: int = -1
        self.elements: List[List[float]] = []
        # connections[node_id][layer] = [neighbor_id, ...]
        self.connections: List[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dist(self, a: List[float], b: List[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def _random_level(self) -> int:
        return int(-math.log(random.uniform(0, 1)) * self.mL)

    def _neighbors(self, node_id: int, layer: int) -> List[int]:
        return self.connections[node_id].get(layer, [])

    def _nearest(self, q: List[float], candidates: List[int]) -> int:
        return min(candidates, key=lambda e: self._dist(q, self.elements[e]))

    def _select_neighbors(self, q: List[float], C: List[int], M: int, lc: int) -> List[int]:
        if self.use_heuristic:
            return self.select_neighbors_heuristic(q, C, M, lc)
        return self.select_neighbors_simple(q, C, M)

    # ------------------------------------------------------------------
    # Algorithm 2: SEARCH-LAYER
    # ------------------------------------------------------------------

    def search_layer(self, q: List[float], ep: List[int], ef: int, lc: int) -> List[int]:
        """Return ef closest neighbors to q at layer lc, starting from enter points ep."""
        v: set = set(ep)
        C: list = []   # min-heap (dist, id) — candidates to explore
        W: list = []   # max-heap (-dist, id) — current best neighbors

        for ep_id in ep:
            d = self._dist(q, self.elements[ep_id])
            heapq.heappush(C, (d, ep_id))
            heapq.heappush(W, (-d, ep_id))

        while C:
            c_dist, c = heapq.heappop(C)   # nearest candidate
            f_dist = -W[0][0]              # distance of furthest element in W

            if c_dist > f_dist:
                break  # all elements in W are evaluated

            for e in self._neighbors(c, lc):
                if e not in v:
                    v.add(e)
                    e_dist = self._dist(q, self.elements[e])
                    f_dist = -W[0][0]

                    if e_dist < f_dist or len(W) < ef:
                        heapq.heappush(C, (e_dist, e))
                        heapq.heappush(W, (-e_dist, e))
                        if len(W) > ef:
                            heapq.heappop(W)  # remove furthest

        return [e for _, e in W]

    # ------------------------------------------------------------------
    # Algorithm 3: SELECT-NEIGHBORS-SIMPLE
    # ------------------------------------------------------------------

    def select_neighbors_simple(self, q: List[float], C: List[int], M: int) -> List[int]:
        """Return M nearest elements from C to q."""
        return sorted(C, key=lambda e: self._dist(q, self.elements[e]))[:M]

    # ------------------------------------------------------------------
    # Algorithm 4: SELECT-NEIGHBORS-HEURISTIC
    # ------------------------------------------------------------------

    def select_neighbors_heuristic(
        self,
        q: List[float],
        C: List[int],
        M: int,
        lc: int,
        extendCandidates: bool = False,
        keepPrunedConnections: bool = False,
    ) -> List[int]:
        """Select M neighbors using the diversity heuristic."""
        R: List[int] = []
        W = list(C)

        if extendCandidates:
            for e in C:
                for e_adj in self._neighbors(e, lc):
                    if e_adj not in W:
                        W.append(e_adj)

        Wd: List[int] = []
        W_heap = [(self._dist(q, self.elements[e]), e) for e in W]
        heapq.heapify(W_heap)

        while W_heap and len(R) < M:
            e_dist, e = heapq.heappop(W_heap)
            # Add e to R only if e is closer to q than to every existing neighbor in R.
            # This promotes connections in diverse directions.
            if not R or all(
                e_dist < self._dist(self.elements[e], self.elements[r]) for r in R
            ):
                R.append(e)
            else:
                Wd.append(e)

        if keepPrunedConnections:
            Wd_heap = [(self._dist(q, self.elements[e]), e) for e in Wd]
            heapq.heapify(Wd_heap)
            while Wd_heap and len(R) < M:
                _, e = heapq.heappop(Wd_heap)
                R.append(e)

        return R

    # ------------------------------------------------------------------
    # Algorithm 1: INSERT
    # ------------------------------------------------------------------

    def insert(self, q: List[float]):
        """Insert element q into the HNSW graph."""
        q_id = len(self.elements)
        self.elements.append(q)
        self.connections.append({})

        ep = self.enter_point
        L = self.max_layer
        l = self._random_level()

        if ep is None:
            self.enter_point = q_id
            self.max_layer = l
            return

        # Phase 1: greedy descent from top layer down to l+1 with ef=1
        for lc in range(L, l, -1):
            W = self.search_layer(q, [ep], ef=1, lc=lc)
            ep = self._nearest(q, W)

        # Phase 2: from min(L, l) down to layer 0 with efConstruction
        ep_list = [ep]
        for lc in range(min(L, l), -1, -1):
            W = self.search_layer(q, ep_list, ef=self.efConstruction, lc=lc)
            neighbors = self._select_neighbors(q, W, self.M, lc)

            self.connections[q_id][lc] = list(neighbors)

            for e in neighbors:
                if lc not in self.connections[e]:
                    self.connections[e][lc] = []
                self.connections[e][lc].append(q_id)

                # Shrink e's connection list if it exceeds Mmax
                Mmax = self.Mmax0 if lc == 0 else self.Mmax
                if len(self.connections[e][lc]) > Mmax:
                    self.connections[e][lc] = self._select_neighbors(
                        self.elements[e], self.connections[e][lc], Mmax, lc
                    )

            ep_list = W  # found neighbors become enter points for next layer

        if l > L:
            self.enter_point = q_id
            self.max_layer = l

    # ------------------------------------------------------------------
    # Algorithm 5: K-NN-SEARCH
    # ------------------------------------------------------------------

    def knn_search(self, q: List[float], K: int, ef: int) -> List[int]:
        """Return K approximate nearest neighbors to q."""
        if self.enter_point is None:
            return []

        ep = self.enter_point
        L = self.max_layer

        # Phase 1: greedy descent from top layer to layer 1 with ef=1
        for lc in range(L, 0, -1):
            W = self.search_layer(q, [ep], ef=1, lc=lc)
            ep = self._nearest(q, W)

        # Phase 2: thorough search at layer 0 with ef
        W = self.search_layer(q, [ep], ef=ef, lc=0)

        return sorted(W, key=lambda e: self._dist(q, self.elements[e]))[:K]


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

def brute_force_knn(q, elements, K):
    def dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    ranked = sorted(range(len(elements)), key=lambda i: dist(q, elements[i]))
    return ranked[:K]


def main():
    random.seed(42)

    dim = 8
    n = 200
    K = 5
    ef = 50

    hnsw = HNSW(M=8, efConstruction=ef)

    vectors = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
    for v in vectors:
        hnsw.insert(v)

    print(f"Built HNSW index: {n} vectors, dim={dim}, M={hnsw.M}, layers={hnsw.max_layer + 1}")

    q = [random.gauss(0, 1) for _ in range(dim)]

    hnsw_results = hnsw.knn_search(q, K=K, ef=ef)
    bf_results = brute_force_knn(q, hnsw.elements, K)

    def dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    print(f"\nQuery vector: {[round(x, 3) for x in q]}")
    print(f"\n{'HNSW':>6}  {'Brute-force':>12}  {'dist':>8}  match")
    print("-" * 40)
    for h, b in zip(hnsw_results, bf_results):
        match = "YES" if h == b else "no"
        print(f"id={h:>4}  id={b:>4}        {dist(q, hnsw.elements[h]):.4f}  {match}")

    recall = len(set(hnsw_results) & set(bf_results)) / K
    print(f"\nRecall@{K}: {recall:.2f}")


if __name__ == "__main__":
    main()
