import osmnx as ox # OSMnx library for street networks using OpenStreetMap data
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython.display import HTML

from copy import deepcopy


# download/model a street network for some city then visualize it
# G = ox.graph_from_place("Manchester, UK", network_type="drive")

# Coordinates of the Dalton Building
point = 53.472075, -2.238855

# Get data from open street maps and plot network
G = ox.graph_from_point(point, dist=2000, network_type="drive", simplify=True)


# what sized area does our network cover in square meters?
G_proj = ox.project_graph(G)
nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)
graph_area_m = nodes_proj.union_all().convex_hull.area
print(f"Graph area is {int(graph_area_m):,} m^2")


# you can convert your graph to node and edge GeoPandas GeoDataFrames
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)


start = ox.distance.nearest_nodes(G, Y=53.472075, X=-2.238855) # Dalton Building
# dest = ox.distance.nearest_nodes(G, Y=53.487124, X=-2.242696) # Victoria Station

dest = ox.distance.nearest_nodes(G, Y=53.47, X=-2.24)
start, dest


# The xs method returns a cross-section from the DataFrame
gdf_edges.xs(start, level="u")


# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 8),facecolor="#111111");
ax.set_facecolor("#111111")
fig, ax, scatter = ox.plot_graph(G, ax=ax, node_alpha=0.7, edge_alpha=0.5);

# Create a list of nodes and get the number of points in the graph.
nodes = list(G.nodes())
num_points = len(list(G))

# Create a dictionary of adjacent node for efficient look-up.
adj_dict = {n: adjacent_node_dict for n, adjacent_node_dict in G.adjacency()}


 # Create a dictionary to map nodes to their index in a list
node_index_map = {nodes[i]: i for i in range(num_points)}

def update_graph_visuals(current_node, closed_set):
    '''
    Function to update the sizes and colours of the nodes in the graph.
    '''

    # Initially set sizes and colours to defaults (size 10, colour white)
    sizes = [10 for i in range(num_points)]
    colours = ["#FFFFFF" for i in range(num_points)] 

    # Loop through all closed (visited) nodes and set them to yellow and size 10
    if len(closed_set) > 0:
        sizes[node_index_map[closed_set[-1]]] = 10
        colours[node_index_map[closed_set[-1]]] = "#FFFF00"

    # Set destination node to size 30 and to purple colour 
    colours[node_index_map[dest]] = "#FF00FF"
    sizes[node_index_map[dest]] = 30
    
    # Set the current node to size 50 and to red colour
    sizes[node_index_map[current_node]] = 50
    colours[node_index_map[current_node]] = "#FF0000"

    # Update the sizes and colours of the nodes in the scatter plot.
    scatter.set_sizes(sizes)
    scatter.set_color(colours)

    return scatter

closed = []
open = [start]
done = False


def search():
    global done

    if done: return 

    node = open.pop(0)
    
    done = (node == dest)

    open.extend([key for key in adj_dict[node].keys() if key not in closed])

    closed.append(node)

    return node

# Animation function
def animate(frame):
    next_node = search()
    update_graph_visuals(next_node, closed)
    return scatter

# Create the animation
ani = FuncAnimation(fig, animate, frames=10, interval=200)

HTML(ani.to_jshtml())
# ani.save("bfs_manchester.mp4")

