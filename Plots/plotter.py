import matplotlib.pyplot as plt
import yaml

def create(model, sharedConfig):
# def main():
    config = sharedConfig
    print("Creating box and whiskers plot")
    metrics = {'MSE': {3:[],6:[],9:[],12:[],24:[]},'MAE': {3:[],6:[],9:[],12:[],24:[]},'RMSE': {3:[],6:[],9:[],12:[],24:[]},'SMAPE': {3:[],6:[],9:[],12:[],24:[]}}

    # with open('../configurations/sharedConfig.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # horizons = config['horizons']['default']
    horizons = [3]
    stations = config['stations']['default']
    
    start = 12
    # model = "TCN"
    # Iterate over each station
    for station in stations:
        # Iterate over each forecasting horizon
        for horizon in horizons:
            try:
                metric_file = f'Results/{model}/{horizon}_Hour_Forecast/Metrics/{station}/metrics.txt'
                # metric_file = f'../Results/Metrics/{station}/metrics_{horizon}'

                # metric_file = f'../Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt'
                with open(metric_file, 'r') as file:
                    # Read the file line by line
                    lines = file.readlines()   

                for line in lines:
                # print(line)
                    # collin_index = line.index(":")
                    collin_index = line[start:].index(" ") + 12
                    # print(collin_index)
                    metric = line[start:collin_index]
                    # print(metric)
                    if metric in metrics:
                        value = line[collin_index + 2:]
                        value = float(value)
                        # print(value)
                        metrics[metric][horizon].append(value)

            except Exception as e:
                print(e)
                print('Error! : Unable to read data or write metrics for station {} and forecast length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
    # print(metrics)
    
    # Get all keys using the keys() method
    keys = metrics.keys()

    # Convert keys to a list if needed
    key_list = list(keys)
    # Define the colors for each box
    box_colors = ['blue', 'orange', 'green', 'red', 'purple']

    for key in key_list:
         # Create a figure and axis
        fig, ax = plt.subplots()
        # Create the box and whisker plot
        boxplot = ax.boxplot([metrics[key][3], metrics[key][6], metrics[key][9], metrics[key][12], metrics[key][24]], patch_artist=True)
        # Set the colors for each box individually
        for box, color in zip(boxplot['boxes'], box_colors):
            box.set(facecolor=color)

        # Set the title and labels
        ax.set_title(model)
        ax.set_xlabel('HORIZONS')
        ax.set_ylabel(key)

        # Set the x-axis ticks and labels
        x_values = [3, 6, 9, 12,24]
        ax.set_xticks(range(1, len(x_values) + 1))
        ax.set_xticklabels(x_values)

        # Set the color of the median line
        median_color = 'black'  # Change to the desired color
        medianprops = dict(color=median_color)
        plt.setp(ax.lines, **medianprops)

        # Show the plot
        # plt.show()

        # Save the box plot to a file
        plt.savefig(f'Plots/{model}/{key}_BWplot.png')
        # plt.savefig(f'{key}_BWplot.png')

# if __name__ == '__main__':
#     main()
