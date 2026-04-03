"""Main DRISPI research pipeline orchestration."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Solution


class DRISPIPipeline:
    """Coordinates AOLS, clustering, solvers, route pool, SP/SC, and improvement."""

    def __init__(self, config: dict, n_solver_workers: int = 4) -> None:
        self.config = config
        self.n_solver_workers = n_solver_workers

    def run(self, instance: CVRPInstance) -> Solution:
        """Execute the full DRISPI loop for ``instance``."""
        # 1. Initialize AOLS with weights from file if available
        # 2. For each segment:
        #    a. AOLS selects (k, dissimilarity, decomp_type, method, solver)
        #    b. Cluster instance using selected method
        #    c. Solve each cluster in parallel using selected solver
        #    d. Add new routes to pool
        #    e. Check coverage floor → dispatch SP or SC
        #    f. Run improvement (BG-AILS)
        #    g. Compute reward → update AOLS weights
        #    h. Every N iterations: evict low-quality routes from pool
        # 3. Return best solution found
        # TODO: implement orchestration using subpackages
        raise NotImplementedError
