import networkx as nx
import numpy as np
import copy
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from typing import List
import random

# ==========================================
# 1. PATHFINDING & NETWORK GENERATION
# ==========================================

def get_diverse_paths(G: nx.DiGraph, source: Any, target: Any, mode: str, num_paths: int = 3, penalty_factor: float = 2.0) -> List[List[Any]]:
    """
    Computes diverse 'Google Maps style' routes for a given OD and mode.
    To avoid highly overlapping routes, it finds the shortest path, penalizes its edges,
    and repeats to find alternative distinct routes.
    """
    G_temp = G.copy()
    diverse_paths = []
    
    for _ in range(num_paths):
        try:
            # Use baseline 'B' of the specific mode as weight
            weight_fn = lambda u, v, d: d['B'][mode]
            path = nx.shortest_path(G_temp, source, target, weight=weight_fn)
            diverse_paths.append(path)
            
            # Penalize the edges of the found path to encourage diversity in the next iteration
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Multiply the baseline cost by a penalty factor
                G_temp[u][v]['B'][mode] *= penalty_factor
        except nx.NetworkXNoPath:
            break # No more paths available
            
    return diverse_paths

def create_multimodal_geometric(n: int, radius: float, modes: List[str]) -> nx.DiGraph:
    """
    Creates a spatial directed random geometric graph.
    Nodes are placed uniformly at random in the unit square.
    Two nodes are joined by an edge if the distance between them is at most 'radius'.
    """
    # Generate the random geometric graph
    G_undirected = nx.random_geometric_graph(n, radius)
    
    # Ensure the graph is fully connected to avoid "No Path" errors during simulation.
    # If it is disconnected, extract the largest connected component.
    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()
        
    # Convert to a directed graph to allow distinct two-way edges and directional flows
    G = G_undirected.to_directed()
    
    # Assign random baseline costs (e.g., travel time in minutes) for each mode on each edge
    for u, v in G.edges():
        G[u][v]['B'] = {}
        
        # We assign costs only to the modes specified in the parameters,
        # maintaining the same cost logic used in the lattice function.
        if 'active' in modes:
            G[u][v]['B']['active'] = np.random.uniform(10.0, 12.0)
        if 'bus' in modes:
            G[u][v]['B']['bus'] = np.random.uniform(2.0, 3.0)
        if 'car' in modes:
            G[u][v]['B']['car'] = np.random.uniform(1.0, 2.0)

    return G

def set_seed(seed: int = 42):
    """
    Sets global random seeds for Python's built-in random module and NumPy 
    to guarantee full reproducibility across execution runs.
    """
    random.seed(seed)
    np.random.seed(seed)

def create_multimodal_spatial(
    n: int, 
    radius: float, 
    modes: List[str], 
    city_size_km: float = 10.0,
    speeds_kmh: Dict[str, float] = None,
    seed: int = 42
) -> nx.DiGraph:
    """
    Creates a deterministic synthetic spatial city network.
    Uses a fixed random seed for reproducible graph structure and costs.
    """
    # 1. Fix global random seeds
    set_seed(seed)

    # Default speeds in km/h
    if speeds_kmh is None:
        speeds_kmh = {
            'car': 35.0,
            'bus': 18.0,
            'active': 12.0
        }

    # 2. Generate random geometric graph passing the seed parameter directly to NetworkX
    G_undirected = nx.random_geometric_graph(n, radius, seed=seed)
    
    # Extract largest connected component to ensure connectivity
    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()

    # 3. Scale node positions to real-world kilometer coordinates
    pos_unit = nx.get_node_attributes(G_undirected, 'pos')
    pos_km = {node: (coords[0] * city_size_km, coords[1] * city_size_km) for node, coords in pos_unit.items()}
    nx.set_node_attributes(G_undirected, pos_km, 'pos')

    # Convert to directed graph
    G = G_undirected.to_directed()

    # 4. Compute Euclidean distances and mode travel times (minutes)
    for u, v in G.edges():
        pos_u = np.array(G.nodes[u]['pos'])
        pos_v = np.array(G.nodes[v]['pos'])
        
        distance_km = np.linalg.norm(pos_u - pos_v)
        G[u][v]['length_km'] = distance_km
        
        G[u][v]['B'] = {}
        for mode in modes:
            speed = speeds_kmh.get(mode, 20.0)
            travel_time_min = (distance_km / speed) * 60.0
            G[u][v]['B'][mode] = travel_time_min

    return G

def create_multimodal_lattice(rows: int, cols: int, modes: List[str]) -> nx.DiGraph:
    """
    Creates a simple spatial directed lattice graph (grid).
    Each edge has a baseline free-flow cost 'B' for each available transportation mode.
    """
    # Create an undirected grid, then convert to directed to allow two-way distinct edges
    G_undirected = nx.grid_2d_graph(rows, cols)
    G = G_undirected.to_directed()
    
    # Assign random baseline costs (e.g., minutes) for each mode on each edge
    for u, v in G.edges():
        G[u][v]['B'] = {
            'active': np.random.uniform(10.0, 12.0),  # Active modes (walking, biking) have a fixed cost
            'bus': np.random.uniform(2.0, 3.0),   # Bus has a higher cost
            'car': np.random.uniform(1.0, 2.0)     # Car has a moderate cost
        }
    return G

# ==========================================
# 2. SIMULATION ENGINE
# ==========================================

class ReplicatorDynamicsSimulator:
    def __init__(self, G: nx.DiGraph, od_pairs: List[Dict], modes: List[str], M_matrix: Dict[str, Dict[str, float]], num_paths: int = 3):
        """
        Initializes the Multimodal Replicator Dynamics simulator.
        
        :param G: The physical network.
        :param od_pairs: List of dicts e.g. [{'orig': (0,0), 'dest': (2,2), 'demand': 1000, 'rho': 0.1}]
        :param modes: List of available modes e.g. ['auto', 'transit', 'bike', 'walk']
        :param M_matrix: Marginal cost dictionary, M[mode1][mode2] = impact of mode2 on mode1
        :param num_paths: Number of diverse routes per mode per OD.
        """
        self.G = G
        self.od_pairs = od_pairs
        self.modes = modes
        self.M_matrix = M_matrix
        self.num_paths = num_paths
        
        # Internal data structures
        self.paths_info = []      # Stores metadata for each path r
        self.edges_list = list(G.edges())
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.edges_list)}
        
        self.history_x = []       # Tracks flows x over time
        self.history_fitness = [] # Tracks global average cost over time
        
        self._initialize_paths()
        
    def _initialize_paths(self):
        """Finds diverse paths for all ODs and modes exactly once before simulation."""
        path_idx = 0
        for od_id, od in enumerate(self.od_pairs):
            orig, dest, demand = od['orig'], od['dest'], od['demand']
            for mode in self.modes:
                routes = get_diverse_paths(self.G, orig, dest, mode, num_paths=self.num_paths)
                for route in routes:
                    # Convert route nodes to edge sequence
                    route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
                    route_edge_indices = [self.edge_to_idx[e] for e in route_edges]
                    
                    self.paths_info.append({
                        'r_id': path_idx,
                        'od_id': od_id,
                        'mode': mode,
                        'demand': demand,
                        'rho': od.get('rho', 0.1), # Adaptation rate
                        'edges': route_edge_indices
                    })
                    path_idx += 1
                    
        # State vector x: initial flow distributed equally among all paths for a given OD
        self.num_total_paths = len(self.paths_info)
        self.x = np.zeros(self.num_total_paths)
        
        for od_id, od in enumerate(self.od_pairs):
            od_paths = [p['r_id'] for p in self.paths_info if p['od_id'] == od_id]
            if od_paths:
                initial_flow = od['demand'] / len(od_paths)
                for r in od_paths:
                    self.x[r] = initial_flow

    def step(self, dt: float = 1.0):
        """Performs one Euler integration step for the replicator dynamics."""
        num_edges = len(self.edges_list)
        
        # 1. Calculate edge volumes v_{a,m}
        v = {mode: np.zeros(num_edges) for mode in self.modes}
        for path in self.paths_info:
            r = path['r_id']
            mode = path['mode']
            flow = self.x[r]
            for edge_idx in path['edges']:
                v[mode][edge_idx] += flow
                
        # 2. Calculate edge costs c_{a,m}
        c = {mode: np.zeros(num_edges) for mode in self.modes}
        for mode in self.modes:
            for edge_idx, edge in enumerate(self.edges_list):
                # B_{a,m}
                cost = self.G[edge[0]][edge[1]]['B'][mode]
                # + sum_m' M_{m,m'} * v_{a,m'}
                for mode_prime in self.modes:
                    cost += self.M_matrix[mode][mode_prime] * v[mode_prime][edge_idx]
                c[mode][edge_idx] = cost
                
        # 3. Calculate path costs C_r
        C_r = np.zeros(self.num_total_paths)
        for path in self.paths_info:
            r = path['r_id']
            mode = path['mode']
            C_r[r] = sum([c[mode][e] for e in path['edges']])
            
        # 4. Calculate average OD costs (mu_w) and Global fitness
        mu_w = np.zeros(len(self.od_pairs))
        for od_id in range(len(self.od_pairs)):
            od_paths = [p['r_id'] for p in self.paths_info if p['od_id'] == od_id]
            total_cost_od = sum([self.x[r] * C_r[r] for r in od_paths])
            demand = self.od_pairs[od_id]['demand']
            mu_w[od_id] = total_cost_od / demand if demand > 0 else 0
            
        total_demand = sum([od['demand'] for od in self.od_pairs])
        global_fitness = sum([mu_w[od_id] * self.od_pairs[od_id]['demand'] for od_id in range(len(self.od_pairs))]) / total_demand
        
        # 5. Update state x using Replicator Equation (Euler method)
        dx = np.zeros(self.num_total_paths)
        for path in self.paths_info:
            r = path['r_id']
            od_id = path['od_id']
            rho = path['rho']
            dx[r] = rho * self.x[r] * (mu_w[od_id] - C_r[r])
            
        self.x += dt * dx
        
        # Enforce non-negativity and correct floating point drifts
        self.x = np.maximum(self.x, 0)
        for od_id in range(len(self.od_pairs)):
            od_paths = [p['r_id'] for p in self.paths_info if p['od_id'] == od_id]
            current_od_flow = np.sum(self.x[od_paths])
            if current_od_flow > 0:
                target_demand = self.od_pairs[od_id]['demand']
                self.x[od_paths] = self.x[od_paths] / current_od_flow * target_demand
        
        # 6. Save history
        self.history_x.append(self.x.copy())
        self.history_fitness.append(global_fitness)

    def run_simulation(self, steps: int, dt: float = 1.0):
        """Runs the simulation for a given number of steps."""
        for _ in range(steps):
            self.step(dt)

def plot_od_evolution(simulator, od_id: int = 0):
    """
    Visualizza l'evoluzione temporale dei flussi sui singoli percorsi e la ripartizione
    modale complessiva per una specifica popolazione (coppia OD).
    
    :param simulator: Istanza di ReplicatorDynamicsSimulator dopo aver eseguito run_simulation.
    :param od_id: Indice della coppia OD da analizzare.
    """
    if not simulator.history_x:
        raise ValueError("La cronologia 'history_x' è vuota. Esegui 'run_simulation()' prima di plottare.")
        
    # 1. Recupera i dati relativi alla coppia OD selezionata
    od_info = simulator.od_pairs[od_id]
    orig, dest, demand = od_info['orig'], od_info['dest'], od_info['demand']
    
    # Filtra i percorsi appartenenti a questo od_id
    od_paths = [p for p in simulator.paths_info if p['od_id'] == od_id]
    
    # Matrice dei flussi: righe = time_steps, colonne = totale percorsi nella rete
    history_x = np.array(simulator.history_x)
    num_steps = history_x.shape[0]
    steps_arr = np.arange(num_steps)

    # 2. Aggrega i dati per i percorsi dell'OD e per ciascun mezzo (modo)
    path_flows = {p['r_id']: history_x[:, p['r_id']] for p in od_paths}
    
    mode_flows = {mode: np.zeros(num_steps) for mode in simulator.modes}
    for p in od_paths:
        mode_flows[p['mode']] += history_x[:, p['r_id']]

    # 3. Plotting (2 Subplot affiancati)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- GRAFICO 1: Flusso per Singolo Percorso ---
    for p in od_paths:
        r_id = p['r_id']
        mode = p['mode']
        axes[0].plot(steps_arr, path_flows[r_id], label=f"Path {r_id} [{mode}]", linewidth=1.8)

    axes[0].set_title(f"Distribuzione sui Percorsi\nOD {od_id}: {orig} -> {dest} (Domanda: {demand})")
    axes[0].set_xlabel("Step di Simulazione")
    axes[0].set_ylabel("Flusso Utenti (x_r)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')

    # --- GRAFICO 2: Modal Split (Evoluzione per Mezzo di Trasporto) ---
    for mode, flow in mode_flows.items():
        if np.sum(flow) > 0:  # Traccia solo i modi effettivamente presenti
            axes[1].plot(steps_arr, flow, label=f"Modo: {mode}", linewidth=2.5)

    axes[1].set_title(f"Ripartizione Modale (Modal Split)\nOD {od_id}: {orig} -> {dest}")
    axes[1].set_xlabel("Step di Simulazione")
    axes[1].set_ylabel("Flusso Totale per Modo")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc='best')

    plt.tight_layout()
    plt.show()