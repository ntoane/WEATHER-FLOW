import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime, timedelta


def visualize_results(horizon, stations):
    fileDictionary = {
        'predFile': 'Logs/CLCRN/' + str(horizon) + ' Hour Forecast/predictions.pkl',
        'targetFile': 'Logs/CLCRN/' + str(horizon) + ' Hour Forecast/actuals.pkl',
    }

    

    with open(fileDictionary["predFile"], 'rb') as f:
        y_pred = pickle.load(f)

    with open(fileDictionary["targetFile"], 'rb') as f:
        y_true = pickle.load(f)
    y_pred = y_pred['y_preds']
    y_true = y_true['y_trues']
   
    start_date = "2021-01-17"  # Format: "YYYY-MM-DD HH:MM:SS"

# Convert start date to a datetime object
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")

# Calculate time values based on start date and hours passed


    for i in range(45):
        l = len(y_pred)
        a = len(y_pred[0])
        station_pred = y_pred[0:l-1,0,i,0]
        station_pred= np.append(station_pred,y_pred[l-1,:,i,0])
        
        station_true = y_true[0:l-1,0,i,0]
        station_true= np.append(station_true,y_true[l-1,:,i,0])
        print("Evaluating horizon:" + str(horizon) + " for station:" + stations[i])

        # Visualize actual vs predicted for each station
        plt.figure(figsize=(12, 6))
        loss= abs(station_true-station_pred)
        numHours = len(loss)
        hoursLeft = numHours % 24
        numDays = numHours // 24
        avgDayLoss = []
        hours = []
        for d in range(numDays):
            hourlyLoss = loss[d*24:(d+1)*24]
            avgDayLoss.append(np.mean(hourlyLoss))

        hourlyLoss = loss[-1*hoursLeft:]
        avgDayLoss.append(np.mean(hourlyLoss))

        time_values = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')

        numDays = numDays//100
        for _ in range(numDays + 1):
            time_values.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=100)

        window_size = 10
        trend_line = np.convolve(avgDayLoss, np.ones(window_size)/window_size, mode='valid')
        plt.plot(avgDayLoss, label="Prediction Loss", color="blue")

        # Set custom x-axis tick locations and labels
        custom_ticks = [0, 100, 200, 300, 400, 500, 600, 700 ]  # Specify the tick locations you want
        custom_labels = time_values  # Specify the labels for those tick locations

        plt.xticks(custom_ticks, custom_labels)

        plt.title(f"Station: {stations[i]} - Average daily loss between actual and predicted")
        plt.xlabel("Time steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        # Save the figure or display it
        directory=f"./Plots/CLCRN/{str(horizon)} Hour Forecast/{stations[i]}_plot.png"
        plt.savefig(directory)
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
