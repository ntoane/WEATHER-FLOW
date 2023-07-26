import pandas as pd
import os

def join_excel_sheets(excel_folder):
    # Get a list of all Excel files in the folder
    excel_files = [f for f in os.listdir(excel_folder) if f.endswith(".xlsx")]

    # Iterate over each Excel file
    for file in excel_files:
        file_path = os.path.join(excel_folder, file)

        # Read the Excel file into a list of DataFrames
        sheets = pd.read_excel(file_path, sheet_name=None)

        # Create a new Excel file to store the joined sheets
        output_file = os.path.join(excel_folder, f"NC_{file}")

        # Concatenate all sheets into a single DataFrame
        joined_data = pd.concat(sheets.values(), axis=0, ignore_index=True)

        # Write the joined data to the new Excel file
        with pd.ExcelWriter(output_file) as writer:
            joined_data.to_excel(writer, sheet_name="Joined", index=False)

        print(f"Successfully joined all sheets in {file} into {output_file}.")

def main():
    print("Starting")
    excel_folder = "test"  # Specify the path to the folder containing Excel files

    join_excel_sheets(excel_folder)
    print("Complete")

if __name__ == '__main__':
    main()

