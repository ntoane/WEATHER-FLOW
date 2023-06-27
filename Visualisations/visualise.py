import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import csv

def create_graph(adj_matrix):
    G = nx.DiGraph()

    with open('Data/Locations/Locations.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            number, station_name, lat, lon, province = row
            G.add_node(number, pos=(float(lon), float(lat)), province=province)

    for i in range(adj_matrix.shape[0]):
        strongest_influence_indices = np.argsort(adj_matrix[:, i])[-1:]  # Enter number of influential stations
        for j in strongest_influence_indices:
            if adj_matrix[j, i] > 0:
                G.add_edge(list(G.nodes())[j], list(G.nodes())[i])

    return G


def plot_map(adj_matrix, province_colours):
    G = create_graph(adj_matrix)
    node_positions = nx.get_node_attributes(G, 'pos')
    
    fig, ax = plt.subplots(figsize=(8, 8),dpi=300)

    min_lon = min(pos[0] for pos in node_positions.values())
    max_lon = max(pos[0] for pos in node_positions.values())
    min_lat = min(pos[1] for pos in node_positions.values())
    max_lat = max(pos[1] for pos in node_positions.values())

    width = max_lon - min_lon
    height = max_lat - min_lat

    m = Basemap(
        llcrnrlon=min_lon - 0.2 * width, llcrnrlat=min_lat - 0.2 * height,
        urcrnrlon=max_lon + 0.2 * width, urcrnrlat=max_lat + 0.2 * height,
        resolution='i', ax=ax
    )
    m.drawcoastlines(linewidth=0.5)
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='white', lake_color='lightblue')

    node_degrees = G.out_degree()
    node_sizes = [100 * degree + 80 for _, degree in node_degrees]
    node_colors = [province_colours[G.nodes[node]['province']] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos=node_positions, node_color=node_colors, node_size=node_sizes, ax=ax)

    edge_colors = [province_colours[G.nodes[u]['province']] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos=node_positions, edge_color=edge_colors, arrows=True, arrowstyle='->', width=1, ax=ax)

    nx.draw_networkx_labels(
        G, pos=node_positions, labels={node: node for node in G.nodes()},
        font_color='white', font_size=8, ax=ax
    )

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=province)
        for province, color in province_colours.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title("Strongest Dependencies")
    fig.savefig('Visualisations/geographicVis.png')


def plot_heatmap(adj_matrix):
    fig_heatmap, ax_heatmap = plt.subplots()
    sns.heatmap(adj_matrix, cmap='YlGnBu', ax=ax_heatmap)
    ax_heatmap.set_title("Adjacency Matrix Heatmap")
    fig_heatmap.savefig('Visualisations/heatmap.png')


def plot(config):
    matrix_path = "Results/" + config['modelVis']['default'] + "/" + config['horizonVis']['default'] + " Hour Forecast/Matrices/adjacency_matrix_" + config['splitVis']['default'] + ".csv"
    df = pd.read_csv(matrix_path, index_col=0)
    adj_matrix = df.values

    province_colours = {
    'Eastern Cape': 'blue',
    'Western Cape': 'red',
    'Northern Cape': 'green'
    }

    plot_map(adj_matrix,province_colours)
    plot_heatmap(adj_matrix)