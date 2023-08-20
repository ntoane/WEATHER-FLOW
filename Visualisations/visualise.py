import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import csv
import os

# def create_graph(adj_matrix, config):
#     G = nx.DiGraph()

#     locations_path = config['locations_path']['default'] #'DataNew/Locations/Locations.csv'
#     with open(locations_path, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip the header row
#         for row in reader:
#             number, station_name, lat, lon, province = row
#             G.add_node(number, pos=(float(lon), float(lat)), province=province)

#     for i in range(adj_matrix.shape[0]):
#         strongest_influence_indices = np.argsort(adj_matrix[:, i])[-1:]  # Enter number of influential stations
#         for j in strongest_influence_indices:
#             if adj_matrix[j, i] > 0:
#                 G.add_edge(list(G.nodes())[j], list(G.nodes())[i])

#     return G


def create_graph(adj_matrix, config):
    G = nx.DiGraph()

    locations_path = config['locations_path']['default'] #'DataNew/Locations/Locations.csv'
    with open(locations_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            number, station_name, lat, lon, province = row
            G.add_node(number, pos=(float(lon), float(lat)), province=province)

    for i in range(adj_matrix.shape[1]):  # Iterate over the columns instead of rows
        strongest_influence_indices = np.argsort(adj_matrix[i, :])[-3:]  # Enter number of influential stations
        for j in strongest_influence_indices:
            if adj_matrix[i, j] > 0:  # Use adj_matrix[i, j] instead of adj_matrix[j, i]
                G.add_edge(list(G.nodes())[j], list(G.nodes())[i], weight=adj_matrix[i, j])  # Add the 'weight' attribute

    return G






#this version only influencial
# def create_graph(adj_matrix, config):
#     G = nx.DiGraph()

#     locations_path = config['locations_path']['default'] #'DataNew/Locations/Locations.csv'
#     with open(locations_path, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip the header row
#         for row in reader:
#             number, station_name, lat, lon, province = row
#             G.add_node(number, pos=(float(lon), float(lat)), province=province)

#     for i in range(adj_matrix.shape[1]):  # Iterate over the columns instead of rows
#         strongest_influence_indices = np.argsort(adj_matrix[i, :])[-1:]  # Enter number of influential stations
#         for j in strongest_influence_indices:
#             if adj_matrix[i, j] > 0:  # Use adj_matrix[i, j] instead of adj_matrix[j, i]
#                 G.add_edge(list(G.nodes())[j], list(G.nodes())[i])  # Swap the order of nodes

#     return G


# this version only adds top x influencers to each node and adds threshold after
# def create_graph(adj_matrix, config):
#     G = nx.DiGraph()

#     locations_path = config['locations_path']['default']
#     with open(locations_path, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip the header row
#         for row in reader:
#             number, station_name, lat, lon, province = row
#             G.add_node(number, pos=(float(lon), float(lat)), province=province)

#     for i in range(adj_matrix.shape[1]):  # Iterate over the columns instead of rows
#         strongest_influence_indices = np.argsort(adj_matrix[i, :])[-40:]  # Enter number of influential stations
#         for j in strongest_influence_indices:
#             if adj_matrix[i, j] > 0.5:  # Only add edges with weight above 0.7
#                 G.add_edge(list(G.nodes())[j], list(G.nodes())[i])  # Swap the order of nodes

#     return G

#this version has threshold only
# def create_graph(adj_matrix, config):
#     G = nx.DiGraph()

#     locations_path = config['locations_path']['default']
#     with open(locations_path, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip the header row
#         for row in reader:
#             number, station_name, lat, lon, province = row
#             G.add_node(number, pos=(float(lon), float(lat)), province=province)

#     rows, cols = adj_matrix.shape
#     for i in range(rows):  
#         for j in range(cols):
#             if adj_matrix[i, j] > 0.6:  # Only add edges with weight above 0.7
#                 G.add_edge(list(G.nodes())[j], list(G.nodes())[i])  # Swap the order of nodes

#     return G



def plot_map(adj_matrix, config, split):
    hex_colors = {
        0: '#E52B50', 1: '#40826D', 2: '#8000FF', 3: '#3F00FF', 4: '#40E0D0', 5: '#008080', 6: '#483C32', 7: '#D2B48C',
        8: '#00FF7F', 9: '#A7FC00', 10: '#708090', 11: '#C0C0C0', 12: '#FF2400', 13: '#0F52BA', 14: '#92000A', 15: '#FA8072',
        16: '#E0115F', 17: '#FF007F', 18: '#C71585', 19: '#FF0000', 20: '#E30B5C', 21: '#6A0DAD', 22: '#CC8899', 23: '#003153',
        24: '#8E4585', 25: '#FFC0CB', 26: '#1C39BB', 27: '#C3CDE6', 28: '#D1E231', 29: '#FFE5B4', 30: '#DA70D6', 31: '#FF4500',
        32: '#FF6600', 33: '#808000', 34: '#CC7722', 35: '#000080', 36: '#E0B0FF', 37: '#800000', 38: '#FF00AF', 39: '#FF00FF',
        40: '#BFFF00', 41: '#C8A2C8', 42: '#FFF700', 43: '#B57EDC', 44: '#29AB87', 45: '#00A86B'
    }


    G = create_graph(adj_matrix, config)
    node_positions = nx.get_node_attributes(G, 'pos')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    min_lon = min(pos[0] for pos in node_positions.values())
    max_lon = max(pos[0] for pos in node_positions.values())
    min_lat = min(pos[1] for pos in node_positions.values())
    max_lat = max(pos[1] for pos in node_positions.values())

    width = max_lon - min_lon
    height = max_lat - min_lat

    m = Basemap(
        llcrnrlon=min_lon - 0.1 * width, llcrnrlat=min_lat - 0.1 * height,
        urcrnrlon=max_lon + 0.1 * width, urcrnrlat=max_lat + 0.1 * height,
        resolution='i', ax=ax
    )
    m.drawcoastlines(linewidth=0.5)
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='white', lake_color='lightblue')

    node_degrees = G.out_degree()
    sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)  # Sort nodes by out-degree
    top_nodes = sorted_nodes[:10]  # Select the top 10 most influential nodes

    # Add number of outgoing edges as labels for top 10 nodes
    top_node_labels = {node[0]: f"{node[0]}\n[{node[1]}]" for node in top_nodes}
    nx.draw_networkx_labels(
        G, pos=node_positions, labels=top_node_labels,
        font_color='black', font_size=5, ax=ax
    )

    # node_sizes = [100 * degree + 80 for _, degree in top_nodes]
    node_colors = [hex_colors[int(key)] for key, _ in top_nodes]
    nx.draw_networkx_nodes(
        G, pos=node_positions, nodelist=[node[0] for node in top_nodes],
        node_color='white', edgecolors=node_colors, node_size=200, ax=ax
    )

    edge_list = G.out_edges([node[0] for node in top_nodes])
    edge_colors = [hex_colors[int(key)] for key, _ in edge_list]

    nx.draw_networkx_edges(
        G, pos=node_positions, edgelist=G.out_edges([node[0] for node in top_nodes]),
        edge_color=edge_colors, arrows=True, arrowstyle='->', width=1, ax=ax
    )

    ax.set_title("Strongest Dependencies")

    directory = 'Visualisations/' + config['modelVis']['default']+ '/horizon_' + config['horizonVis']['default'] + '/' + 'geographicVis/'
    filename = 'geoVis_split_' + split + '.png'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    fig.savefig(filepath)


def plot_heatmap(adj_matrix, config, split):
    fig_heatmap, ax_heatmap = plt.subplots()
    sns.heatmap(adj_matrix, cmap='YlGnBu', ax=ax_heatmap)
    ax_heatmap.set_title("Adjacency Matrix Heatmap")



    directory = 'Visualisations/' + config['modelVis']['default']+ '/horizon_' + config['horizonVis']['default'] + '/' + 'heatmap/'
    filename = 'heatmap_split_' + split + '.png'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    fig_heatmap.savefig(filepath)


#zulu

# import matplotlib.colors as mcolors
# import matplotlib.cm as cm

# def find_chains(G, start_node, visited=None, chain=None):
#     if visited is None:
#         visited = set()
#     if chain is None:
#         chain = []
    
#     visited.add(start_node)
#     chain.append(start_node)
#     chains = []

#     for neighbor in G[start_node]:
#         if neighbor not in visited:
#             extended_chain = list(chain)
#             chains.extend(find_chains(G, neighbor, visited, extended_chain))
#         else:
#             if len(chain) > 1:
#                 chains.append(chain)
    
#     return chains

# def find_chains(G, start_node, visited=None):
#     if visited is None:
#         visited = set()

#     if start_node in visited:
#         return []

#     visited.add(start_node)

#     chains_for_node = []
#     for neighbor in G[start_node]:
#         if G[start_node][neighbor]['weight'] > 0.7:
#             for chain in find_chains(G, neighbor, visited.copy()):
#                 chains_for_node.append([start_node] + chain)

#     if not chains_for_node:
#         chains_for_node.append([start_node])

#     return chains_for_node


# def plot_strongest_chains(adj_matrix, config):
#     G = create_graph(adj_matrix, config)
    
#     chains = []
#     for node in G.nodes():
#         chains.extend(find_chains(G, node))

#     # Get all weights to normalize color mapping
#     weights = [adj_matrix[int(i), int(j)] for i, j in G.edges()]
#     norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights), clip=True)
#     mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

#     fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
#     node_positions = nx.get_node_attributes(G, 'pos')

#     # Draw the chains
#     for chain in chains:
#         for i in range(len(chain) - 1):
#             start = chain[i]
#             end = chain[i + 1]
#             weight = adj_matrix[int(end), int(start)]
#             color = mapper.to_rgba(weight)
#             nx.draw_networkx_edges(G, pos=node_positions, edgelist=[(start, end)], edge_color=[color], ax=ax)
    
#     # Drawing nodes
#     nx.draw_networkx_nodes(G, pos=node_positions, ax=ax, node_color='lightblue', node_size=200)

#     ax.set_title("Strongest Chains of Influence")
    
#     # Saving the figure
#     directory = 'Visualisations/' + config['modelVis']['default'] + '/horizon_' + config['horizonVis']['default'] + '/' + 'geographicVis/'
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     filename = 'chains_of_influence.png'
#     filepath = os.path.join(directory, filename)
#     fig.savefig(filepath)

import networkx as nx
import matplotlib.pyplot as plt

def strong_dfs_paths(G, source, weight_threshold, visited=None, path=None):
    """Recursively finds strong paths using DFS."""
    if visited is None:
        visited = set()
    if path is None:
        path = [source]

    visited.add(source)

    strong_paths = []
    for neighbor in G[source]:
        weight = G[source][neighbor].get('weight', 0)  # Safely get the weight attribute with a default value of 0
        if neighbor not in visited and weight > weight_threshold:
            visited.add(neighbor)
            path.append(neighbor)
            strong_paths.append(list(path))
            strong_paths.extend(strong_dfs_paths(G, neighbor, weight_threshold, visited, path))
            path.pop()

    return strong_paths

import itertools
from itertools import cycle
import matplotlib


def plot_strong_chains(adj_matrix, config):
    G = create_graph(adj_matrix, config)
    weight_threshold = 0.06  # Adjust this value based on what you consider as "strong"
    
    strong_paths = []
    for node in G:
        strong_paths.extend(strong_dfs_paths(G, node, weight_threshold))
    
    # Filter paths with 2 or more edges (i.e., 3 or more nodes)
    strong_paths = [path for path in strong_paths if len(path) >= 3]

    # Extract unique nodes from the strong paths
    strong_nodes = set()
    for path in strong_paths:
        strong_nodes.update(path)

    # Create unique path signatures and assign colors
    path_colors = {}
    number_of_colors = 10
    unique_colors = plt.cm.tab10(np.linspace(0, 1, number_of_colors))
    colors_cycle = cycle(unique_colors)  # This will cycle through the colors if there are more than 10 paths

    for index, path in enumerate(strong_paths):
        signature = "-".join(path)
        if signature not in path_colors:
            path_colors[signature] = next(colors_cycle)

    # Visualize the strong paths
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 6))  # You can adjust the size of the visualization

    # Draw nodes and their labels
    nx.draw_networkx_nodes(G, pos, nodelist=strong_nodes, node_size=300, node_color="skyblue")
    nx.draw_networkx_labels(G, pos, labels={node: node for node in strong_nodes})

    # Draw the paths with unique colors
    for path in strong_paths:
        signature = "-".join(path)
        color = path_colors[signature]
        for i in range(len(path) - 1):
            nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i + 1])], edge_color=color, width=2, arrows=True, arrowsize=20)

    plt.show()



#colour map version
# import matplotlib

# def plot_strong_chains(adj_matrix, config):
#     G = create_graph(adj_matrix, config)
#     weight_threshold = 0.5  # Adjust this value based on what you consider as "strong"
    
#     strong_paths = []
#     for node in G:
#         strong_paths.extend(strong_dfs_paths(G, node, weight_threshold))
    
#     # Filter paths with 2 or more edges (i.e., 3 or more nodes)
#     strong_paths = [path for path in strong_paths if len(path) >= 3]

#     # Create a dictionary to count the number of paths each edge belongs to
#     edge_counts = {}
#     for path in strong_paths:
#         for i in range(len(path)-1):
#             edge = (path[i], path[i+1])
#             edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
#     # Extract unique nodes from the strong paths
#     strong_nodes = set()
#     for path in strong_paths:
#         strong_nodes.update(path)

#     # Visualize the strong paths
#     pos = nx.get_node_attributes(G, 'pos')
#     plt.figure(figsize=(10, 6))  # You can adjust the size of the visualization

#     # Draw nodes and their labels
#     nx.draw_networkx_nodes(G, pos, nodelist=strong_nodes, node_size=1000, node_color="skyblue")
#     nx.draw_networkx_labels(G, pos, labels={node: node for node in strong_nodes})

#     # Use a gradient colormap to color edges based on the count
#     max_count = max(edge_counts.values())
#     color_map = matplotlib.colormaps.get_cmap('Reds')

#     for edge, count in edge_counts.items():
#         color = color_map(count / max_count)  # This will scale the color based on the count
#         nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=color, width=2, arrows=True, arrowsize=20)

#     fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size
#     sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=1, vmax=max_count))
#     cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, aspect=20)
#     cbar.set_label('Number of Paths')

#     # Draw nodes and their labels
#     nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=strong_nodes, node_size=1000, node_color="skyblue")
#     nx.draw_networkx_labels(G, pos, labels={node: node for node in strong_nodes}, ax=ax)

#     # Use a gradient colormap to color edges based on the count
#     for edge, count in edge_counts.items():
#         color = color_map(count / max_count)  # This will scale the color based on the count
#         nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[edge], edge_color=color, width=2, arrows=True, arrowsize=20)

#     plt.show()



# import itertools

# def plot_strong_chains(adj_matrix, config):
#     G = create_graph(adj_matrix, config)
#     weight_threshold = 0.5  # Adjust this value based on what you consider as "strong"
    
#     strong_paths = []
#     for node in G:
#         strong_paths.extend(strong_dfs_paths(G, node, weight_threshold))
    
#     # Filter paths with 2 or more edges (i.e., 3 or more nodes)
#     strong_paths = [path for path in strong_paths if len(path) >= 3]

#     # Extract unique nodes from the strong paths
#     strong_nodes = set()
#     for path in strong_paths:
#         strong_nodes.update(path)
    
#     # Visualize the strong paths
#     pos = nx.get_node_attributes(G, 'pos')
#     plt.figure(figsize=(10, 6))  # You can adjust the size of the visualization

#     # Draw nodes and their labels
#     nx.draw_networkx_nodes(G, pos, nodelist=strong_nodes, node_size=1000, node_color="skyblue")
#     nx.draw_networkx_labels(G, pos, labels={node: node for node in strong_nodes})

#     # List of colors
#     colors = itertools.cycle(plt.cm.tab20.colors)  # Using a colormap with 20 colors, loop over if more paths

#     # Draw only the strong edges with different colors for different paths
#     for path in strong_paths:
#         color = next(colors)  # Get the next color
#         nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i+1]) for i in range(len(path)-1)], edge_color=color, width=2, arrows=True, arrowsize=20)  # Adjust arrowsize as needed

#     plt.show()




def plot(config):
    
    number_of_splits = int(config['splitVis']['default'])
    for split in range(number_of_splits+1):
        split=str(split)
        matrix_path = "Results/" + config['modelVis']['default'] + "/" + config['horizonVis']['default'] + " Hour Forecast/Matrices/adjacency_matrix_" + split + ".csv"
        df = pd.read_csv(matrix_path, index_col=0)
        adj_matrix = df.values
        plot_strong_chains(adj_matrix, config)
        plot_map(adj_matrix, config , split)
        plot_heatmap(adj_matrix, config, split)
