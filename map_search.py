
import time
import math
import pickle
import random
import osmnx as ox # OSMnx library for street networks using OpenStreetMap data
import numpy as np
import pandas as pd
import seaborn as sns
from enum import Enum
from typing import Callable
import matplotlib.pyplot as plt
from filprofiler.api import profile
from pretty_progress import progress_bar

start_coords = (53.472075, -2.238855) # The Dalton Building

class Node():
    '''
    Generic node class to store graph nodes.

    Attributes:
        node_id: int = The node's unique id number from OSMnx
        cost_to_reach: float = Stores the cost (time) to reach to node from the starting node
        depth: int = Stores the depth in the graph of the node
        parent: Node = Stores the node which lead to this node in the search
    '''
    def __init__(self, node_id: int, cost_to_reach: float = 0, parent=None, heuristic_val: float = 0):
        self.node_id: int = node_id
        self.cost_to_reach: float = cost_to_reach
        self.heuristic_value: float = heuristic_val
        self.depth: int = 0
        
        self.parent = parent

    def path_to(self):
        ''' Recursively find the set of nodes from the start node to this one and returns a list a their IDs. '''
        if self.parent is None: return []
        
        return self.parent.path_to() + [self.node_id]
                
    def __str__(self):
        return f"<class Node Object>, ID: {self.node_id}"
    
    def __repr__(self):
        return f"({self.__str__()})"
    
    def __eq__(self, other):
        # Compare equality based on the node_id
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False

    def __hash__(self):
        return hash(self.node_id)

# Searching
class SearchType(Enum):
    BFS = 0
    DFS = 1
    UCS = 2
    A_STAR = 3

class SearchClass:
    '''
    General class for searching searching for paths between two nodes. Implements breadth-first search,
    depth-first search, uniform-cost search and A*.

    Attributes:
        start: Node = The node to begin search from
        destination: Node = The node to end on

    '''
    def __init__(self, start_node: Node, destination_node: Node, heuristic_func: Callable = None) -> None:
        self.start: Node = start_node
        self.destination: Node = destination_node
        self.done: bool = False
        self.no_path: bool = False

        self.open: list = [start_node]
        # Perhaps closed could be a set rather than a list
        self.closed: list = []

        self.current_node: Node = start_node

        self.start.heuristic_value = heuristic_func(start_node, destination_node) if heuristic_func is not None else 0
        self.heuristic_func: Callable = heuristic_func

        self.num_visited = 0


    def _reset(self) -> None:
        ''' Reset the object's attributes so no data from old searches leak into new search.'''
        self.closed = []
        self.done = False
        self.no_path = False
        self.heuristic_dict = {self.start: self.heuristic_func(self.start, self.destination)} if self.heuristic_func is not None else {}
        self.open = [self.start]
        self.current_node = self.start
        self.num_visited = 0

    def run_search(self, search_type: SearchType, heuristic_func: Callable = None) -> None:
        '''
        Runs a full search between the start and destination nodes using a given search algorithm.
        '''
        self._reset()

        if heuristic_func: self.heuristic_func = heuristic_func
        if search_type == SearchType.A_STAR and self.heuristic_func == None:
            raise Exception("Heuristic function is required to run A* search.")
        
        while not self.done:
            self.search_step(search_type=search_type)

    def search_step(self, search_type: SearchType) -> Node:
        '''
        Performs one step of a search algorithm.
        '''      
        next_node = self.pick_node(search_type)
        self.done = (next_node == self.destination) or (next_node is None)
        if next_node is None: return
        self.open.extend(
            [Node(node_id,
                    parent = next_node,
                    cost_to_reach = (next_node.cost_to_reach + self.calc_path_time([next_node.node_id, node_id]) if search_type in [SearchType.UCS, SearchType.A_STAR] else np.nan), 
                    heuristic_val = self.heuristic_func(Node(node_id), self.destination) if search_type == SearchType.A_STAR else 0
                  ) for node_id in adj_dict[next_node.node_id] if Node(node_id) not in self.closed])
        self.closed.append(next_node)
        self.current_node = next_node

        return next_node
    
    def pick_node(self, search_type: SearchType) -> Node:
        ''' Selects a node to visit next from the open set based on the type of search being performed. '''
        if len(self.open) == 0:
            self.no_path = True
            return None
        
        if search_type == SearchType.BFS:
            return self.open.pop(0)
        if search_type == SearchType.DFS:
            return self.open.pop(-1)
        if search_type == SearchType.UCS:
            self.open.sort(key=lambda node: node.cost_to_reach)
            return self.open.pop(0)
        if search_type == SearchType.A_STAR:
            # Perhaps insert nodes rather than sorting the whole list ?
            self.open.sort(key=lambda node: node.heuristic_value + node.cost_to_reach)
            return self.open.pop(0)
        
    def calc_path_time(self, path: list) -> float:
        '''
        Takes path as a list of node_id's and calculates the total time to traverse from
        the first node to the last via each node in the path. This assumes that there is
        a valid edge between each node in the path. Returns the time in seconds.
        '''
        time = np.sum([time_between(path[i], path[i+1])
                for i in range(len(path) - 1)])

        return time
    
    def describe_path(self, path: list):
        '''
        Takes path as a list of node_id's and returns a list of the unique streets traversed in order
        as the path is followed.
        '''
        path_list = [str(list(gdf_edges.xs(path[i], level="u").xs(path[i + 1], level="v")["name"])[0]) 
                     for i in range(len(path) - 1)]

        return pd.Series(path_list).unique()

    def display_final(self) -> None:
        if len(self.closed) <= 0:
            raise Exception("Must execute run_search method before displaying.")
        
        path = self.closed[-1].path_to()
        print("Final path:", self.describe_path(path))
        print(f"Path travel time: {format_time(self.calc_path_time(path))}.")
    
        # Get node coordinates from the graph
        node_x = [G.nodes[node.node_id]['x'] for node in nodes]
        node_y = [G.nodes[node.node_id]['y'] for node in nodes]

        # Set default sizes and colors
        sizes = [1 for _ in range(num_points)]
        colours = ["#DDDDDD" for _ in range(num_points)] 

        # Extract start and destination node positions
        start_x, start_y = G.nodes[self.start.node_id]['x'], G.nodes[self.start.node_id]['y']
        dest_x, dest_y = G.nodes[self.destination.node_id]['x'], G.nodes[self.destination.node_id]['y']

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(7,7), facecolor="#111111")
        ax.set_facecolor("#111111")
        ax.axis('off')
        
        # Remove margins
        ax.margins(0)
        plt.tight_layout(pad=0)

        # Plot all edges (graph edges) in grey
        for u, v, data in G.edges(data=True):
            x_coords = [G.nodes[u]['x'], G.nodes[v]['x']]
            y_coords = [G.nodes[u]['y'], G.nodes[v]['y']]
            ax.plot(x_coords, y_coords, color="#777777", alpha=1, lw=1, zorder=1)

        # Highlight path edges
        full_path = [self.start.node_id] + path  # Include start node explicitly
        for i in range(len(full_path) - 1):  # Iterate through consecutive nodes in the path
            u, v = full_path[i], full_path[i + 1]
            x_coords = [G.nodes[u]['x'], G.nodes[v]['x']]
            y_coords = [G.nodes[u]['y'], G.nodes[v]['y']]
            ax.plot(x_coords, y_coords, color="#FFFF00", lw=2, alpha=1, zorder=3)  # Highlighted path edges

        # Plot all other nodes with default sizes and colors
        ax.scatter(node_x, node_y, s=sizes, c=colours, zorder=2)

        # Plot nodes on the path
        for node_id in path:
            x, y = G.nodes[node_id]['x'], G.nodes[node_id]['y']
            ax.scatter(x, y, s=6, c="#FFFF00", zorder=4)

        # Plot visited nodes
        for node in self.closed:
            if node.node_id in path: next
            x, y = G.nodes[node.node_id]['x'], G.nodes[node.node_id]['y']
            ax.scatter(x, y, s=1, c="#CCCC55", zorder=4)    

        # Plot the start node on top (in red)
        ax.scatter(start_x, start_y, s=50, c="#FF0000", label="Start", zorder=5)

        # Plot the destination node on top (in purple)
        ax.scatter(dest_x, dest_y, s=50, c="#FF00FF", label="Destination", zorder=5)

        # Add legend for clarity
        ax.legend(facecolor="#222222", labelcolor="#FFFFFF", loc="upper right")

        plt.show()


# Helper functions
def mph_to_mps(speed_mph: float) -> float:
    ''' Convert speed from mph to metres per second. '''
    return (speed_mph * 1609.344) / 60**2

def time_between(start_node_id: int, end_node_id: int) -> float:
    '''
    Locates the edge between two nodes (based on their ID) and returns the time taken to
    drive between them in seconds.
    '''
    edge_between_nodes = gdf_edges.xs(start_node_id, level="u").xs(end_node_id, level="v")

    road_speed_mph = edge_between_nodes["maxspeed"][0].item()
    road_length_m = edge_between_nodes["length"][0].item()

    return road_length_m / mph_to_mps(road_speed_mph) 

def format_time(seconds: float) -> str:
    '''
    Converts time in seconds to a readable string in hours, minutes and seconds.
    '''
    return time.strftime(f"%H:%M:{round(seconds%60, 3)}s", time.gmtime(seconds))


# Visualisation code

# Heuristics
def manhattan_distance(start_node: Node, end_node: Node) -> float:
    '''
    Manhattan distance is not an admissible heuristic in the context of driving as roads can have different speed limits
    which means a node may be given a large heuristic value as it is far from the target but it may be a fast road and thus
    the heuristic will overestimate the true cost.        
    '''
    x_start, y_start = G.nodes[start_node.node_id]['x'], G.nodes[start_node.node_id]['y']
    x_end, y_end = G.nodes[end_node.node_id]['x'], G.nodes[end_node.node_id]['y']

    # x and y are given in latitude/longitude degrees so they need to be converted to metres
    return abs(x_start - x_end)*0.01745*math.cos(y_start) + abs(y_start - y_end)*111319

max_speed = mph_to_mps(70)
def distance_normalised(start_node: Node, end_node: Node) -> float:
    '''
    A admissible heuristic can be created by dividing the straight-line distance to the goal by the maximum speed on the roads.
    This ensures that the heuristic is maximally optimistic i.e. assuming the car travels directly at max speed to the goal.
    With an optimistic heuristic like this the cost will never be overestimated.
    '''
    global max_speed
    return manhattan_distance(start_node, end_node) / max_speed


# Running
with open("graph_data.pkl", "rb") as f:
    data = pickle.load(f)

start_node = data["start_node"]
destination_node = data["destination_node"]
nodes = data["nodes"]
node_index_map = data["node_index_map"]
adj_dict = data["adj_dict"]
G = data["G"]
searcher = data["searcher"]
gdf_nodes = data["gdf_nodes"]
gdf_edges = data["gdf_edges"]
fig = data["fig"]
ax = data["ax"]
scatter = data["scatter"]
num_points = data["num_points"]

# == PROFILE ==
method = SearchType.BFS
# Run a search with each algorithm
start_time = time.time()
profile(lambda: searcher.run_search(search_type=method), "/tmp/fil-result")
print(f"{method.name} search took {format_time(time.time() - start_time)}")
print(f"Visited {len(searcher.closed)} nodes.")
searcher.display_final()

# == RUN SAMPLES ==
n_trials = 300

results_df = pd.DataFrame({
    "Distance": np.zeros(n_trials),
    f"{method.name}_Time": np.zeros(n_trials),
    f"{method.name}_Journey_Time": np.zeros(n_trials),
})

# Run the search n times
for i in range(0, n_trials):
    progress_bar(i, n_trials)

    start = start_node
    end = random.choice(nodes)

    searcher = SearchClass(start_node, end, heuristic_func=distance_normalised)

    start_time = time.time_ns()
    searcher.run_search(search_type=method)
    if searcher.no_path: next
    duration = time.time_ns() - start_time
    journey_time = searcher.calc_path_time(searcher.closed[-1].path_to())

    # Fill row
    results_df.loc[i] = [distance_normalised(start, end)/1e3,
                         duration / 1e6,
                         journey_time]
    
results_df.to_csv(f"results_{method.name}.csv")