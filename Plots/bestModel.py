import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def get_smape_from_file(file_path):
    if os.path.exists(file_path):  # Ensure the file exists
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "smape:" in line:
                    return float(line.split(":")[1].strip())
    return None

def get_lowest_smape_model(station_name, models, split):
    lowest_smape = float('inf')
    best_model = None
    for model in models:
        if model=='TCN':
            path = f"./Results/{model}/3 Hour Forecast/{station_name}/metrics/metrics_" + str(split) + ".txt"
        else:
            path = f"./Results/{model}/3 Hour Forecast/Metrics/{station_name}/metrics_" + str(split) + ".txt"
        smape = get_smape_from_file(path)
        # print("this is model:")
        # print(model)
        # print("this is smape:")
        # print(smape)
        if smape is not None and smape < lowest_smape:
            lowest_smape = smape
            best_model = model
    return best_model, lowest_smape

def main():
    split=0
    # Define model colors in a dictionary
    MODEL_COLORS = {
        'GWN': 'blue',
        'TCN': 'green',
        'AGCRN': 'red',
        # Add other models as you like
        # 'MODEL_NAME': 'color',
    }

    # Load the CSV containing station coordinates
    df = pd.read_csv("./DataNew/Locations/Locations.csv")

    # Create new figure
    fig =plt.figure(figsize=(10, 10))

    # Set up the base map centered around South Africa
    m = Basemap(projection='merc', llcrnrlat=-35, urcrnrlat=-25,
                llcrnrlon=15, urcrnrlon=33, resolution='i')
    m.drawcountries()
    m.drawcoastlines()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='lightgray', lake_color='aqua')

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MODEL_COLORS[model], markersize=10, label=model) for model in MODEL_COLORS.keys()]
    plt.legend(handles=legend_handles, loc='lower left', fontsize='small', title='Models')
    plt.title("Model which produced lowest SMAPE score per station", fontsize=14, fontweight='bold')




    for index, row in df.iterrows():
        station_number = row['Number']
        station_name = row['StasName']
        latitude = row['Latitude']
        longitude = row['Longitude']

        x, y = m(longitude, latitude)
        best_model, smape = get_lowest_smape_model(station_name, MODEL_COLORS.keys(), split)
        m.plot(x, y, 'o', color=MODEL_COLORS[best_model])
        plt.text(x, y, f"{station_number}", fontsize=9)

    directory = 'Plots/BestModels'
    filename = 'bestModel_split_' + str(split) + '.png'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=300)

if __name__ == "__main__":
    main()
