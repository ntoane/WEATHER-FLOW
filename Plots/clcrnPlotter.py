import matplotlib.pyplot as plt
import yaml

def create(model, sharedConfig):
    config = sharedConfig
    print("Creating box and whiskers plot")
    metrics = {'MSE': {3:[],6:[],9:[],12:[],24:[]},'SMAPE': {3:[],6:[],9:[],12:[],24:[]}}


    horizons = config['horizons']['default']
    stations = config['stations']['default']
    
    start = 0
    # Iterate over each forecasting horizon
    for horizon in horizons:
        try:
            metric_file = f'Results/{model}/stationScore_{horizon}.txt'
            with open(metric_file, 'r') as file:
                # Read the file line by line
                lines = file.readlines()   

            for line in lines:
                collin_index = line.index(" ")
                metric = line[:collin_index]
                if metric in metrics:
                    collin_index = line.index(":") + 2
                    end = line.index('\n')
                    value = line[collin_index:end-1]
                    value = float(value)
                    metrics[metric][horizon].append(value)

        except Exception as e:
            print(e)
            print('Error! : Unable to read data or write metrics for forecast length {}. Please review the data or code for the metrics for errors.'.format(horizon))
    
    # Get all keys using the keys() method
    keys = metrics.keys()

    # Convert keys to a list if needed
    key_list = list(keys)
    # Define the colors for each box
    box_colors = ['cyan', 'orange', 'green', 'red', 'purple']

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
        # Customize the outlier symbols
        outlier_marker = 'D'  # Change to the desired marker style (e.g., 'x', '+', 's', 'D', etc.)
        outlier_size = 4  # Change to the desired size
        outlier_color = 'black'  # Change to the desired color
        flierprops = dict(marker=outlier_marker, markersize=outlier_size, markerfacecolor=outlier_color)
        
        # Apply the outlier customization
        for flier in boxplot['fliers']:
            flier.set_marker(outlier_marker)
            flier.set_markersize(outlier_size)
            flier.set_markerfacecolor(outlier_color)


        # Save the box plot to a file
        plt.savefig(f'Plots/{model}/{key}_BWplot.png')

