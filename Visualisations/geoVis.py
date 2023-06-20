import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import csv
import sys


def plot(args):
    # Read the adjacency matrix from the CSV file
    matrix_path="Results/"+ args.modelVis + "/" + args.horizonVis + " Hour Forecast/Matrices/adjacency_matrix_" + args.splitVis + ".csv"
    df = pd.read_csv(matrix_path, index_col=0)
    adj_matrix = df.values

    # Define province colors
    province_colors = {
        'Eastern Cape': 'blue',
        'Western Cape': 'red',
        'Northern Cape': 'green'
    }

    # Create a NetworkX directed graph
    G = nx.DiGraph()

    # Read the CSV file containing province information
    with open('Data/Weather Station Data/Provinces/Provinces.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            number, station_name, lat, lon, province = row
            G.add_node(number, pos=(float(lon), float(lat)), province=province)

    # Add edges with the strongest influencers for each node
    for i in range(adj_matrix.shape[0]):
        strongest_influence_indices = np.argsort(adj_matrix[:, i])[-2:]  # Enter number of influential stations
        for j in strongest_influence_indices:
            if adj_matrix[j, i] > 0:
                G.add_edge(list(G.nodes())[j], list(G.nodes())[i])

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Get the node positions
    node_positions = nx.get_node_attributes(G, 'pos')

    # Calculate the bounding box of the node positions
    min_lon = min(pos[0] for pos in node_positions.values())
    max_lon = max(pos[0] for pos in node_positions.values())
    min_lat = min(pos[1] for pos in node_positions.values())
    max_lat = max(pos[1] for pos in node_positions.values())

    # Calculate the center and width/height of the bounding box
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    width = max_lon - min_lon
    height = max_lat - min_lat

    # Create a Basemap object with adjusted boundaries
    m = Basemap(
        llcrnrlon=min_lon - 0.2 * width, llcrnrlat=min_lat - 0.2 * height,
        urcrnrlon=max_lon + 0.2 * width, urcrnrlat=max_lat + 0.2 * height,
        resolution='l', ax=ax
    )

    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='white', lake_color='lightblue')

    # Get the out-degree of each node (number of edges leaving the node)
    node_degrees = G.out_degree()

    # Plot nodes
    node_sizes = [50 * degree + 80 for _, degree in node_degrees]  # Calculate node sizes based on out-degree
    node_colors = [province_colors[G.nodes[node]['province']] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos=node_positions, node_color=node_colors, node_size=node_sizes, ax=ax)

    # Plot edges
    edge_colors = [province_colors[G.nodes[u]['province']] for u, v in G.edges()]  # Use the color of the source node
    nx.draw_networkx_edges(G, pos=node_positions, edge_color=edge_colors, arrows=True, arrowstyle='->', width=0.5, ax=ax)

    # Add numbers to nodes
    nx.draw_networkx_labels(
        G, pos=node_positions, labels={node: node for node in G.nodes()},
        font_color='white', font_size=8, ax=ax
    )

    # Add a legend for province colors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=province)
        for province, color in province_colors.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add title and show the plot
    plt.title("Strongest Influencers in South Africa (Directed)")
    # plt.show()

    


    # Create the directory if it does not exist

    plt.savefig('Visualisations/vis.png')