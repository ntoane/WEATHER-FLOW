import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import csv
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable



def create_graph(adj_matrix, config):
    G = nx.DiGraph()
    locations_path = config['locations_path']['default']
   
    with open(locations_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            number, station_name, lat, lon, province = row
            G.add_node(number, pos=(float(lon), float(lat)), province=province)

    # adj_matrix=adj_matrix.T

    for i in range(adj_matrix.shape[1]):  # Iterate over the columns instead of rows
        strongest_influence_indices = np.argsort(adj_matrix[i, :])[-2:]
        for j in strongest_influence_indices:
            weight = adj_matrix[i, j]
            if weight > config['thresholdVis']['default']:  # Check if the weight is greater than the threshold
                G.add_edge(list(G.nodes())[j], list(G.nodes())[i], weight=weight)

    return G



def plot_map(adj_matrix, config):
    # Create graph using adjacency matrix
    G = create_graph(adj_matrix, config)
    
    node_positions = nx.get_node_attributes(G, 'pos')

    # Setup plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    
    # Calculate bounding box
    lons = [pos[0] for pos in node_positions.values()]
    lats = [pos[1] for pos in node_positions.values()]
    width, height = max(lons) - min(lons), max(lats) - min(lats)

    m = Basemap(
        llcrnrlon=min(lons) - 0.1 * width, 
        llcrnrlat=min(lats) - 0.1 * height,
        urcrnrlon=max(lons) + 0.1 * width, 
        urcrnrlat=max(lats) + 0.1 * height,
        resolution='i', ax=ax
    )
    m.drawcountries()
    m.drawcoastlines()

    # Drawing nodes, edges, and labels
    node_degrees = list(G.out_degree())
    numberNodesDisplay = config['numberNodesDisplay']['default']
    top_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)[:numberNodesDisplay]
    drawn_nodes = [node[0] for node in top_nodes]
    
    edge_weights = nx.get_edge_attributes(G, "weight")
    # min_weight, max_weight = min(edge_weights.values()), max(edge_weights.values())
    # normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights.values()]
    min_weight, max_weight = np.min(adj_matrix), np.max(adj_matrix)
    # normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights.values()]
    if max_weight == min_weight:  # All weights are the same
        normalized_weights = [0.5 for w in edge_weights.values()]  # Use 0.5 as a default normalized value
    else:
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights.values()]



    
    colormap = plt.cm.viridis
    edge_colors = [colormap(w) for w in normalized_weights]
    drawn_edges = [edge for edge in edge_weights.keys() if edge[0] in drawn_nodes]
    
    nx.draw_networkx_labels(G, pos=node_positions, labels={node[0]: f"{node[0]}" for node in top_nodes}, font_color='black', font_size=9, ax=ax)
    nx.draw_networkx_nodes(G, pos=node_positions, nodelist=drawn_nodes, node_color='white', edgecolors='#000000', node_size=200, ax=ax)
    nx.draw_networkx_edges(G, pos=node_positions, edgelist=drawn_edges, edge_color=edge_colors, arrows=True, arrowstyle='->', width=1.3, ax=ax)

    # Add colorbar
    norm = Normalize(vmin=min_weight, vmax=max_weight)
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', ax=ax)
    cbar.set_label('Edge Weight')

    ax.set_title("Strongest Dependencies")
    
    # Save the plot
    directory = f"Visualisations/{config['modelVis']['default']}/horizon_{config['horizonVis']['default']}/geographicVis/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    split=config['splitVis']['default']
    filepath = os.path.join(directory, f'geoVis_split_{split}.png')
    fig.savefig(filepath, dpi=600)




def plot_heatmap(adj_matrix, config):
    fig_heatmap, ax_heatmap = plt.subplots()
    sns.heatmap(adj_matrix, cmap='YlGnBu', ax=ax_heatmap)
    ax_heatmap.set_title("Adjacency Matrix Heatmap")

    split = config['splitVis']['default']
    directory = 'Visualisations/' + config['modelVis']['default']+ '/horizon_' + config['horizonVis']['default'] + '/' + 'heatmap/'
    filename = 'heatmap_split_' + split + '.png'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    fig_heatmap.savefig(filepath)



def strong_dfs_paths(G, source, visited=None, path=None):
    """Recursively finds strong paths using DFS."""
    if visited is None:
        visited = set()
    if path is None:
        path = [source]

    visited.add(source)

    strong_paths = []
    for neighbor in G[source]:
        weight = G[source][neighbor].get('weight', 0)  # Safely get the weight attribute with a default value of 0
        if neighbor not in visited:
            visited.add(neighbor)
            path.append(neighbor)
            strong_paths.append(list(path))
            strong_paths.extend(strong_dfs_paths(G, neighbor, visited, path))
            path.pop()

    return strong_paths



def plot_strong_chains(adj_matrix, config):
    G = create_graph(adj_matrix, config)
    strong_paths = []
    

    for node in G:
        strong_paths.extend(strong_dfs_paths(G, node))
    
    # Filter paths with 2 or more edges (i.e., 3 or more nodes)
    strong_paths = [path for path in strong_paths if len(path) >= 3]

    # Extract unique nodes from the strong paths
    strong_nodes = set()
    for path in strong_paths:
        strong_nodes.update(path)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

    node_positions = nx.get_node_attributes(G, 'pos')
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

    # Convert latitude and longitude to x, y for the basemap
    pos = {node: m(G.nodes[node]['pos'][0], G.nodes[node]['pos'][1]) for node in G.nodes()}

    # Draw nodes and their labels
    nx.draw_networkx_nodes(G, pos, nodelist=strong_nodes, node_size=300, node_color="white", edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={node: node for node in strong_nodes}, font_size=9, ax=ax)

    # Getting edge weights and normalizing them for color mapping
    edge_weights = nx.get_edge_attributes(G, "weight")

    colormap = plt.cm.viridis
    norm = Normalize(vmin=min(edge_weights.values()), vmax=max(edge_weights.values()))

    # Draw the paths with colors based on edge weights
    for path in strong_paths:
        for i in range(len(path) - 1):
            edge_color = colormap(norm(edge_weights[(path[i], path[i + 1])]))
            nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i + 1])], edge_color=edge_color, width=2, arrows=True, arrowstyle='->', ax=ax)

    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', ax=ax)
    cbar.set_label('Edge Weight')
    
    split = config['splitVis']['default']
    directory = 'Visualisations/' + config['modelVis']['default'] + '/horizon_' + config['horizonVis']['default'] + '/' + 'chains/'
    filename = 'chains_split_' + split + '.png'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=600)



def plot(config):
    matrix_path = "Results/" + config['modelVis']['default'] + "/" + config['horizonVis']['default'] + " Hour Forecast/Matrices/adjacency_matrix_" + config['splitVis']['default'] + ".csv"
    df = pd.read_csv(matrix_path, index_col=0)
    adj_matrix = df.values

    adj_matrix = (adj_matrix - np.min(adj_matrix)) / (np.max(adj_matrix) - np.min(adj_matrix))

    
    plot_map(adj_matrix, config )
    plot_strong_chains(adj_matrix, config)
    plot_heatmap(adj_matrix, config)
