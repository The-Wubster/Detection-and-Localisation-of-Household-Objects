import cv2
import pandas as pd
import os
from os import listdir
import numpy as np

os.chdir("/Volumes/Ryan_Extern")

# File paths for input/output images:
DIR_INPUT = 'Full_Dataset_All_Augs'
DIR_OUTPUT = 'Full_Dataset_All_Augs_resized'
train_input_path = f'{DIR_INPUT}/Training_Data/images'
validation_input_path = f'{DIR_INPUT}/Validation_Data/images'
test_input_path = f'{DIR_INPUT}/Test_Data/images'
train_output_path = f'{DIR_OUTPUT}/Train_Data/images'
valid_output_path = f'{DIR_OUTPUT}/Validation_Data/images'
test_output_path = f'{DIR_OUTPUT}/Test_Data/images'
ratio_to_original = 0.424    # Scale by 0.424 for smaller width=1280 pixels, scale by 0.212 for smaller width=640 pixels,
square_image = 0
# Function to resize images and save them in a new directory:
def resize_image(image_directory, new_image_directory):
    original_image = cv2.imread(image_directory, cv2.IMREAD_UNCHANGED)

    # Scaling Image
    new_width = int(original_image.shape[1] * ratio_to_original)
    if square_image == 0:
        new_height = int(original_image.shape[0] * ratio_to_original)
    else:
        new_height = new_width

    resized_image = cv2.resize(original_image, (new_width, new_height))
    rotated_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)

    # Save New Image
    cv2.imwrite(new_image_directory,resized_image)

# Locating training, validation and test images to be resized:
train_list = os.listdir(train_input_path)
image_train = []

validation_list = os.listdir(validation_input_path)
image_valid = []

test_list = os.listdir(test_input_path)
image_test = []

for item in train_list:
    if ".jpeg" in item:
        image_path = train_input_path + '/' +item
        output_path = train_output_path + '/' + item
        resize_image(image_path, output_path)
        image_train.append(item)

for item in validation_list:
    if ".jpeg" in item:
        image_path = validation_input_path + '/' +item
        output_path = valid_output_path + '/' + item
        resize_image(image_path, output_path)
        image_valid.append(item)

for item in test_list:
    if ".jpeg" in item:
        image_path = test_input_path + '/' +item
        output_path = test_output_path + '/' + item
        resize_image(image_path, output_path)
        image_test.append(item)

"""# Resizing images in yolo format:
train_input_path_l = f'{DIR_INPUT}/Training_Data/labels'
validation_input_path_l = f'{DIR_INPUT}/Validation_Data/labels'
test_input_path_l = f'{DIR_INPUT}/Test_Data/labels'
train_output_path_l = f'{DIR_OUTPUT}/Train_Data/labels'
valid_output_path_l = f'{DIR_OUTPUT}/Validation_Data/labels'
test_output_path_l = f'{DIR_OUTPUT}/Test_Data/labels'

input_label_directory = [train_input_path_l, validation_input_path_l, test_input_path_l]
output_label_directory = [train_output_path_l, valid_output_path_l, test_output_path_l]

for i, input_dir in enumerate(input_label_directory):
    output_dir = output_label_directory[i]
    for txt_file in listdir(input_dir):
        if (txt_file.endswith(".txt") and txt_file[0] != 'l'):
            image_path = input_dir + '/' + txt_file
            label_df = pd.read_csv(image_path, sep=" ", names=['Item_id', 'X_min', 'Y_min', 'X_width', 'y_width'])
            for count in range(len(label_df)):
                label_df.at[count, 'X_min'] = label_df.loc[count, 'X_min'] * ratio_to_original
                label_df.at[count, 'Y_min'] = label_df.loc[count, 'Y_min'] * ratio_to_original
                label_df.at[count, 'X_width'] = label_df.loc[count, 'X_width'] * ratio_to_original
                label_df.at[count, 'y_width'] = label_df.loc[count, 'y_width'] * ratio_to_original
            output_path = output_dir + '/' + txt_file    
            np.savetxt(output_path, label_df, delimiter=' ')"""

# Resize bounding boxes and image dimentions in csv document 
"""training_csv_path = f'{DIR_INPUT}/train/train.csv'
validation_csv_path = f'{DIR_INPUT}/validation/validation.csv'
test_csv_path = f'{DIR_INPUT}/validation/validation.csv'
training_csv_output = f'{DIR_OUTPUT}/train/train.csv'
validation_csv_output = f'{DIR_OUTPUT}/validation/validation.csv'
test_csv_output = f'{DIR_OUTPUT}/validation/validation.csv'

columns_to_scale = ['image_width', 'image_height', 'x_min', 'y_min', 'x_width', 'y_width']
train_df = pd.read_csv(training_csv_path)
validation_df = pd.read_csv(validation_csv_path)
test_df = pd.read_csv(test_csv_path)

for x in columns_to_scale:
    for y in range(len(train_df)):
        train_df.at[y, x] = int(train_df.iloc[y][x] * ratio_to_original)
train_df.to_csv(training_csv_output)

for x in columns_to_scale:
    for y in range(len(validation_df)):
        validation_df.at[y, x] = int(validation_df.iloc[y][x] * ratio_to_original)
validation_df.to_csv(validation_csv_output)

for x in columns_to_scale:
    for y in range(len(test_df)):
        test_df.at[y, x] = int(test_df.iloc[y][x] * ratio_to_original)
test_df.to_csv(test_csv_output)
"""