# # 修改输出的CSV文件，添加一列以区分不同Brand
# import os
# import re
# import csv
#
#
# csv_dir = "     "
#
# # Regular expression pattern to match the brand name in the filename
# pattern = r'^([a-zA-Z]+)_.*\.csv$'
#
# # Iterate over all CSV files in the specified directory
# for filename in os.listdir(csv_dir):
#     if not filename.endswith('.csv'):
#         # Skip files that are not CSV files
#         continue
#
#     # Use the regular expression pattern to match the brand name in the filename
#     match = re.match(pattern, filename)
#     if not match:
#         # Skip files that don't match the pattern
#         continue
#     brand = match.group(1)
#
#     # Open the CSV file for reading
#     with open(os.path.join(csv_dir, filename), 'r') as f:
#         reader = csv.reader(f)
#         rows = list(reader)
#
#     # Add a header row with the new column
#     rows[0].insert(0, 'Brand')
#
#     # Iterate over the data rows and add the brand to each row
#     for row in rows[1:]:
#         row.insert(0, brand)
#
#     # Write the modified data back to the CSV file
#     with open(os.path.join(csv_dir, filename), 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerows(rows)
#
# # 基于运行上述代码得到的文件，重新进行合并
# def CSVWriter(CSV_writer, row, num_columns, DeleteRepeat):
#     if len(row) == num_columns and tuple(row) not in DeleteRepeat:
#         CSV_writer.writerow(row)
#         DeleteRepeat.add(tuple(row))
#
# for SubDirectory in os.listdir(csv_dir):
#     sub_directory_path = os.path.join(csv_dir, SubDirectory)
#     if os.path.isdir(sub_directory_path):
#         csv_filename = os.path.join(csv_dir, SubDirectory + '.csv')
#         with open(csv_filename, 'w', newline='') as csvfile:
#             csv_writer = csv.writer(csvfile)
#             num_columns = None
#             seen_rows = set()
#             for file in os.listdir(sub_directory_path):
#                 file_path = os.path.join(sub_directory_path, file)
#                 if file.endswith('.csv'):
#                     with open(file_path) as f:
#                         csv_reader = csv.reader(f)
#                         for row in csv_reader:
#                             if num_columns is None:
#                                 num_columns = len(row)
#                             CSVWriter(csv_writer, row, num_columns, seen_rows)


# 上述合并是基于不同的Feature，现在根据Brand进行重新合并
import os
import glob
import pandas as pd

# Set the parent directory containing all the subfolders
parent_dir = "./result"

# Set the output directory for the XLS files
output_dir = "./merged"

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each brand
for brand_name in os.listdir(parent_dir):
    brand_path = os.path.join(parent_dir, brand_name)

    # Create an empty Excel writer object to write to the XLS file
    output_filename = os.path.join(output_dir, f"{brand_name}.xls")
    writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')

    # Loop through each subfolder for the current brand
    for foldername in os.listdir(brand_path):
        folderpath = os.path.join(brand_path, foldername)

        # Loop through each CSV file for the current brand and feature
        csv_files = glob.glob(os.path.join(folderpath, f"{brand_name}_*.csv"))
        for csv_file in csv_files:
            # Read in the data from the CSV file
            df = pd.read_csv(csv_file)

            # Extract the feature name from the filename
            feature_name = os.path.splitext(os.path.basename(csv_file))[0].split("_")[1]

            # Add the data to a new sheet in the Excel writer object
            sheet_name = f"{feature_name}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save the Excel writer object to the XLS file
    writer.save()



