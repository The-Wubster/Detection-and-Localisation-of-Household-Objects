import cv2
from numpy.core.fromnumeric import size
import pandas as pd
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

os.chdir("/Volumes/Ryan_Extern")

# File paths for input/output images:
DIR_INPUT = 'Normal_Dataset_Full'

# Splitting labels in yolo format:
train_input_path_l = f'{DIR_INPUT}/Training_Data/labels'
validation_input_path_l = f'{DIR_INPUT}/Validation_Data/labels'
test_input_path_l = f'{DIR_INPUT}/Test_Data/labels'
labels_file = f'{DIR_INPUT}/Test_Data/labels/labels.txt'
label_list = []

input_label_directory = [train_input_path_l, validation_input_path_l, test_input_path_l]

for i, input_dir in enumerate(input_label_directory):
    for txt_file in listdir(input_dir):
        if (txt_file.endswith(".txt") and txt_file[0] != 'l'):
            image_path = input_dir + '/' + txt_file
            label_df = pd.read_csv(image_path, sep=" ", names=['Item_id', 'x_center', 'y_center', 'x_width', 'y_width'])
            first_column = label_df.iloc[:, 0]
            for count in range(len(label_df)):
                label_list.append(label_df.iloc[count, 0])
print(len(label_list))

print('Max val is: '+ str(max(label_list)))

class_df = pd.read_csv(labels_file, sep=" ", names=['Class Name'])
class_df["Class Count"] = 0

for count in range(len(label_list)):
    current_val = label_list[count]
    class_df.at[current_val, 'Class Count'] = class_df.loc[current_val, 'Class Count'] + 1
class_df = class_df.drop(class_df[class_df['Class Count'] < 1].index)
print(class_df)
class_df = class_df.sort_values('Class Count', ascending=False)
print(class_df)
ax = class_df.plot.bar(x='Class Name', y='Class Count')

matplotlib.rcParams['savefig.dpi'] = 300
plt.yscale('log')
plt.rc('xtick', labelsize=7)
plt.ylabel('Number of Class Instances')
plt.xlabel('Class Name')
plt.title('Bar Plot Showing Class Distribution for Dataset')
ax.get_legend().remove()
plt.savefig('Dataset Class Distribution.png', bbox_inches="tight")
plt.show()
