import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp


class ODReplicatorNetwork:
    """
    OD-based multimodal replicator dynamics.

    State:
        x[od,m]

    where

        od = origin-destination pair
        m  = transport mode

    Constraint:
        sum_m x[od,m] = 1
        for every OD pair
    """

    def __init__(
        self,
        multilayer_graphs,
        zones,
        edge_lengths,
        edge_capacities,
        free_flow_speeds,
        rho=1.0,
        interaction_matrix=None,
        od_demands=None,
    ):
        """
        Parameters
        ----------
        multilayer_graphs : list[networkx.Graph]
            One graph per transport mode.

        zones : list
            Selected OD nodes.

        edge_lengths : array (E,)
            Edge lengths [meters].

        edge_capacities : array (E,K)
            Effective capacities per edge and mode [vehicles].

        free_flow_speeds : array (E,K)
            Free-flow speed for edge and mode.

        rho :
            Replicator rate.
        """

        self.graphs = multilayer_graphs

        self.K = len(multilayer_graphs)

        self.zones = zones

        self.rho = rho

        self.M = (
            interaction_matrix
            if interaction_matrix is not None
            else np.ones(self.K)
        )

        # --------------------------------------------------
        # Edge indexing
        # --------------------------------------------------

        self.edges = list(
            multilayer_graphs[0].edges()
        )

        self.edge_to_id = {
            e: i
            for i, e in enumerate(self.edges)
        }

        self.E = len(self.edges)

        self.lengths = np.asarray(edge_lengths)

        self.edge_cost = np.zeros((self.E, self.K))

        self.capacities = np.asarray(
            edge_capacities
        )  # shape (E,K)

        self.speeds = np.asarray(
            free_flow_speeds
        )

        # --------------------------------------------------
        # OD pairs
        # --------------------------------------------------

        self.od_pairs = []

        for i in zones:
            for j in zones:

                if i != j:
                    self.od_pairs.append((i, j))

        self.N_od = len(self.od_pairs)
        
        # --------------------------------------------------
        # OD demands
        # --------------------------------------------------
        
        if od_demands is not None:
            od_demands = np.asarray(od_demands)
            if od_demands.size != self.N_od:
                raise ValueError(
                    f"od_demands must have exactly {self.N_od} values "
                    f"(one per OD pair), got {od_demands.size}"
                )
            self.od_demands = od_demands
        else:
            # Default: uniform demand = 1.0 per OD pair
            self.od_demands = np.ones(self.N_od)

        # --------------------------------------------------
        # Precompute shortest paths
        # --------------------------------------------------

        self.paths = self._compute_paths()

    # ======================================================
    # PRECOMPUTE SHORTEST PATHS
    # ======================================================

    def _compute_paths(self):

        paths = {}

        for m, G in enumerate(self.graphs):

            # edge travel time at free flow
            for e_idx, (u, v) in enumerate(
                self.edges
            ):
                G[u][v]["weight"] = (
                    self.lengths[e_idx]
                    / self.speeds[e_idx, m]
                )

            for od_id, (o, d) in enumerate(
                self.od_pairs
            ):

                node_path = nx.shortest_path(
                    G,
                    source=o,
                    target=d,
                    weight="weight",
                )

                edge_path = []

                for k in range(
                    len(node_path) - 1
                ):
                    u = node_path[k]
                    v = node_path[k + 1]

                    if (u, v) in self.edge_to_id:
                        edge_path.append(
                            self.edge_to_id[(u, v)]
                        )
                    else:
                        edge_path.append(
                            self.edge_to_id[(v, u)]
                        )

                paths[(od_id, m)] = edge_path

        return paths

    # ======================================================
    # EDGE FLOWS
    # ======================================================

    def compute_edge_flows(
    self,
    x,
    ):
        """
        Returns

        flows[e,m]
        """

        if self.od_demands is None:
            self.od_demands = np.ones(self.N_od)

        flows = np.zeros((self.E, self.K))

        for od in range(self.N_od):

            demand = self.od_demands[od]

            for m in range(self.K):

                share = x[od, m]

                path = self.paths[(od, m)]

                for e in path:

                    flows[e, m] += demand * share

        return flows

    # ======================================================
    # EDGE COSTS
    # ======================================================

    def compute_edge_costs(
        self,
        flows,
        ):
        """
        flows shape = (E,K)

        returns costs[e,m]
        """

        costs = np.zeros((self.E, self.K))

        for e in range(self.E):

            densities = (
                flows[e]
                / self.capacities[e, :]
            )

            for m in range(self.K):

                congestion = np.sum(
                    self.M[m] * densities
                )

                ff_time = (
                    self.lengths[e]
                    / self.speeds[e, m]
                )

                costs[e, m] = (
                    ff_time
                    + congestion
                )

        return costs

    # ======================================================
    # OD COSTS
    # ======================================================

    def compute_od_costs(
        self,
                ):
        """
        Cost of each mode for each OD pair.
        """

        costs = np.zeros(
            (self.N_od, self.K)
        )

        for od in range(
            self.N_od
        ):

            for m in range(
                self.K
            ):

                path = self.paths[
                    (od, m)
                ]

                costs[od, m] = np.sum(
                    self.edge_costs[path, m]
                )

        return costs

    # ======================================================
    # REPLICATOR DYNAMICS
    # ======================================================

    def dynamics(
        self,
        t,
        x_flat,
        od_demands=None,
    ):
        x = x_flat.reshape(
            self.N_od,
            self.K,
        )

        eps = 1e-12
        x = np.nan_to_num(x, nan=eps, posinf=eps, neginf=eps)
        x = np.clip(x, eps, 1.0)
        x = x / x.sum(axis=1, keepdims=True)

        flows = self.compute_edge_flows(x)
        self.edge_costs = self.compute_edge_costs(flows)
        od_costs = self.compute_od_costs()

        dxdt = np.zeros_like(x)

        for od in range(self.N_od):

            mu = np.sum(x[od] * od_costs[od])

            dxdt[od] = (
                self.rho
                * x[od]
                * (
                    mu
                    - od_costs[od]
                )
            )

        return dxdt.flatten()

    def simulate(
        self,
        x0,
        t_span=(0, 50),
        n_steps=300,
    ):
        t_eval = np.linspace(
            t_span[0],
            t_span[1],
            n_steps,
        )

        sol = solve_ivp(
            self.dynamics,
            t_span=t_span,
            y0=x0.flatten(),
            t_eval=t_eval,
            method="Radau",      
            rtol=1e-6,
            atol=1e-8,
        )

        print("solve_ivp success:", sol.success)
        print("message:", sol.message)
        if np.isnan(sol.y).any():
            print("NaN detected in solution")

        T = len(sol.t)
        X = sol.y.T.reshape(
            T,
            self.N_od,
            self.K,
        )

        return sol.t, X