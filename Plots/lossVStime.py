import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_results(horizon, k, stations):
    fileDictionary = {
        'predFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Predictions/outputs_' + str(k),
        'targetFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Targets/targets_' + str(k),
        # ... [rest of your dictionary items]
    }

    y_pred = np.load(fileDictionary["predFile"] + ".npy")
    y_true = np.load(fileDictionary["targetFile"] + ".npy")

    for i in range(45):
        station_pred = y_pred[:, :, i, 0]
        station_true = y_true[:, :, i, 0]
        print("Evaluating horizon:" + str(horizon) + " split:" + str(k) + " for station:" + stations[i])

        # Visualize actual vs predicted for each station
        plt.figure(figsize=(12, 6))
        loss= abs(station_true[:,2]-station_pred[:,2])
        # loss = loss.reshape(-1, 9)
        print(station_pred[:,2])
        plt.plot(station_true[:,2], label="Actual", color="blue")
        plt.plot(station_pred[:,2], label="Predicted", color="red", linestyle="--")
        # plt.plot(loss, label="Actual", color="blue")

        plt.title(f"Station {stations[i]} - Actual vs Predicted")
        plt.xlabel("Time steps")
        plt.ylabel("Values")
        plt.legend()
        plt.tight_layout()

        # Save the figure or display it
        directory=f"./Plots/AGCRN/{str(horizon)} Hour Forecast/Plots/station_{stations[i]}_plot.png"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.show()
        # If you want to display the plot instead of saving it, use plt.show() instead of plt.savefig()
        # plt.show()

        plt.close()


def main():
    horizon =  3 # Example horizon value, adjust as necessary
    k = 0  # Example split value, adjust as necessary
    stations = ["ADDO ELEPHANT PARK"]  # Your list of stations
    
    visualize_results(horizon, k, stations)


if __name__ == "__main__":
    main()
