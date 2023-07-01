import yaml
import os

# Get the parent directory
parent_directory = os.path.abspath(os.path.join(os.getcwd(), "../"))
print(f"Parent directory: {parent_directory}")

# Specify the file path
yaml_file_path = os.path.join(parent_directory, "config.yaml")
print(f"YAML file path: {yaml_file_path}")

# Open and read the YAML file
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)
    
forecasting_horizons = [3, 6, 9, 12, 24]

# Specify the path for the main directory
main_directory = '/Logs/TCN/Train'

# Create the main directory
os.makedirs(main_directory, exist_ok=True)
print(f"Main directory '{main_directory}' created.")


for forecast_len in forecasting_horizons:
        for station in stations:
            print('Forecasting at station ', station)
            subdir = + str(forecast_len) + ' Hour Forecast/' + station 
            
            subdirectory_path = os.path.join(main_directory, subdir)
            os.makedirs(subdirectory_path, exist_ok=True)
            print(f"Subdirectory '{subdirectory_path}' created.")
    
            #targetFile = 'Logs/TCN/Train/' + str(forecast_len) + ' Hour Forecast/' + station + str(forecast_len) +"_" +station+".csv"

            #print('Forecasting at done station ', station)
                
