"""

│─Root
├───Directory 1 (Posts)
├─────Sub Directory 1 (Brand 1)
├─────Sub Directory 2 (Brand 2)
├─────Sub Directory 3 (Brand 3)
│
├───Directory 2 (GV Results)
├─────Sub Directory 1 (Brand 1)
├─────Sub Directory 2 (Brand 2)
├─────Sub Directory 3 (Brand 3)

"""

import os
import csv
import shutil
from def_func_demo import ExtractEngageNumber, CountFaceAppear, ImageColorfulPotential, HSVSpaceFeature, ColorPercentage
from def_func_demo import CalculateFg2BgRatio, Face2Diagonal, Face2IntersectionPoint, Shoe2Diagonal, Shoe2IntersectionPoint


# 定义path变量
post_dir = "path/to/post/directory"   # 就是Directory 1那一级
GV_dir = "path/to/GV/results/directory"  # 同上
result_output_path = "path/to/save/result/directory"  # 最好新建一个文件夹
cache_save_dir = "path/to/save/cache"    # 用于存放GrabCut识别Fg, Bg后产生的Binary Image, 最好也新建一个文件夹

# 调用demo folder进行测试的路径
# post_dir = "./test_dir/posts"
# GV_dir = "./test_dir/gv_re"
# result_output_path = "/content/sample_data"
# cache_save_dir = "/content/sample_data/cache"


# 读取并计算GV返回结果
for subdir in os.listdir(GV_dir):
    subdir_path = os.path.join(GV_dir, subdir)
    if os.path.isdir(subdir_path):
        CountFaceAppear(subdir_path, result_output_path)
        ImageColorfulPotential(subdir_path, result_output_path)
        ColorPercentage(subdir_path, result_output_path)


# 读取并计算Post图片特征等
for subdir in os.listdir(post_dir):
    subdir_path = os.path.join(post_dir, subdir)
    if os.path.isdir(subdir_path):
        ExtractEngageNumber(subdir_path, result_output_path)
        HSVSpaceFeature(subdir_path, result_output_path)
        CalculateFg2BgRatio(subdir_path, cache_save_dir, result_output_path)


# 计算构图相关的图片特征
for subdir in os.listdir(GV_dir):
    subdir_path = os.path.join(GV_dir, subdir)
    if os.path.isdir(subdir_path):
        Face2Diagonal(subdir_path, post_dir, result_output_path)
        Face2IntersectionPoint(subdir_path, post_dir, result_output_path)
        Shoe2Diagonal(subdir_path, post_dir, result_output_path)
        Shoe2IntersectionPoint(subdir_path, post_dir, result_output_path)


# 将相同类型数据归类至一个文件夹中
for resultfile in os.listdir(result_output_path):
    if resultfile.endswith('.csv'):
        end_of_name = resultfile.split('_')[-1].replace('.csv', '')
        SubDirectory = os.path.join(result_output_path, end_of_name)
        os.makedirs(SubDirectory, exist_ok=True)
        shutil.move(os.path.join(result_output_path, resultfile), os.path.join(SubDirectory, resultfile))



# 写到新csv file中，结果将会输出到result_output_path，其中，num_columns将会以csv文件第一行的个数来约束后续行的数目，并剔除
def CSVWriter(CSV_writer, row, num_columns, DeleteRepeat):
    if len(row) == num_columns and tuple(row) not in DeleteRepeat:
        CSV_writer.writerow(row)
        DeleteRepeat.add(tuple(row))

for SubDirectory in os.listdir(result_output_path):
    sub_directory_path = os.path.join(result_output_path, SubDirectory)
    if os.path.isdir(sub_directory_path):
        csv_filename = os.path.join(result_output_path, SubDirectory + '.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            num_columns = None
            seen_rows = set()
            for file in os.listdir(sub_directory_path):
                file_path = os.path.join(sub_directory_path, file)
                if file.endswith('.csv'):
                    with open(file_path) as f:
                        csv_reader = csv.reader(f)
                        for row in csv_reader:
                            if num_columns is None:
                                num_columns = len(row)
                            CSVWriter(csv_writer, row, num_columns, seen_rows)




# 修改输出的CSV文件，在前面加一列以区分不同Brand
for result_file in os.listdir(result_output_path):
    if result_file.endswith(".csv"):
        file_path = os.path.join(result_output_path, result_file)
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            folder_base = os.path.basename(result_output_path)
            brand = folder_base.split("_")[1]
            writer.writerow(["Brand", brand])

