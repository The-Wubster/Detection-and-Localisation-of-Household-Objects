import cv2
import pandas as pd
import os
from os import listdir
import numpy as np

os.chdir("/Volumes/Ryan_Extern")

# Function to flip and labels images and save them in a new directory:
def flip_image(image_directory, new_image_directory):
    original_image = cv2.imread(image_directory, cv2.IMREAD_UNCHANGED)
    flipped_image = cv2.flip(original_image, 1)
    flipped_image = cv2.rotate(flipped_image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    change_brightness(flipped_image, output_path, 35)
    #cv2.imwrite(new_image_directory,flipped_image)

def flip_image_labels(image_path1, output_path1):
    label_df1 = pd.read_csv(image_path1, sep=" ", names=['Item_id', 'x_center', 'y_center', 'x_width', 'y_width'])
    for count in range(len(label_df1)):
        label_df1.at[count, 'x_center'] = 1 - label_df1.loc[count, 'x_center']
    label_df1.to_csv(output_path1, header=None, index=None, sep=' ', mode='w')

# Function to change brightness and add noise:
def change_brightness(image_directory, new_image_directory, value_to_change=30):  # Sourced from: https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    #curr_image = cv2.imread(image_directory)
    curr_image = image_directory
    hsv = cv2.cvtColor(curr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value_to_change
    v[v > lim] = 255
    v[v <= lim] += value_to_change

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(new_image_directory, img)

def add_noise_to_image(image_directory, new_image_directory): # Adapted from https://theailearner.com/2019/05/07/add-different-noise-to-an-image/
    curr_image = cv2.imread(image_directory)
    gauss_noise = np.random.normal(0,1,curr_image.size)
    gauss_noise = gauss_noise.reshape(curr_image.shape[0],curr_image.shape[1],curr_image.shape[2]).astype('uint8')
    image_with_noise = cv2.add(curr_image, gauss_noise)
    #image_with_noise = curr_image + curr_image * gauss_noise
    
    cv2.imwrite(new_image_directory, image_with_noise)

###################################################################################################################

# File paths for input/output images:
DIR_INPUT = 'Normal_Dataset_Full'
DIR_OUTPUT = 'mixed'
train_input_path = f'{DIR_INPUT}/Training_Data/images'
validation_input_path = f'{DIR_INPUT}/Validation_Data/images'
test_input_path = f'{DIR_INPUT}/Test_Data/images'
train_output_path = f'{DIR_OUTPUT}/Train_Data/images'
valid_output_path = f'{DIR_OUTPUT}/Validation_Data/images'
test_output_path = f'{DIR_OUTPUT}/Test_Data/images'
train_input_path_l = f'{DIR_INPUT}/Training_Data/labels'
validation_input_path_l = f'{DIR_INPUT}/Validation_Data/labels'
test_input_path_l = f'{DIR_INPUT}/Test_Data/labels'
train_output_path_l = f'{DIR_OUTPUT}/Train_Data/labels'
valid_output_path_l = f'{DIR_OUTPUT}/Validation_Data/labels'
test_output_path_l = f'{DIR_OUTPUT}/Test_Data/labels'

# Locating training, validation and test images to be resized:
train_list = os.listdir(train_input_path)
image_train = []

validation_list = os.listdir(validation_input_path)
image_valid = []

test_list = os.listdir(test_input_path)
image_test = []

dir_lists = [train_list, validation_list, test_list]
image_lists = [image_train, image_valid, image_test]
input_image_directories = [train_input_path, validation_input_path, test_input_path]
output_image_directories = [train_output_path, valid_output_path, test_output_path]

for i, current_dir in enumerate(dir_lists):
    for item in current_dir:
        if ".jpeg" in item:
            image_path = input_image_directories[i] + '/' + item
            output_path = output_image_directories[i] + '/' + item[:8] + '_noise.jpeg'
            if int(item[7]) % 3 == 1:
                add_noise_to_image(image_path, output_path)
            else:
                flip_image(image_path, output_path)
                #change_brightness(image_path, output_path, 35)
            image_lists[i].append(item)

input_label_directory = [train_input_path_l, validation_input_path_l, test_input_path_l]
output_label_directory = [train_output_path_l, valid_output_path_l, test_output_path_l]

"""for i, input_dir in enumerate(input_label_directory):
    output_dir = output_label_directory[i]
    for txt_file in listdir(input_dir):
        if (txt_file.endswith(".txt") and txt_file[0] != 'l'):
            image_path = input_dir + '/' + txt_file
            output_path = output_dir + '/' + txt_file[:8] + '_flip.txt'
            if int(txt_file[7]) % 3 == 1:
                label_df = pd.read_csv(image_path, sep=" ", names=['Item_id', 'x_center', 'y_center', 'x_width', 'y_width'])
                label_df.to_csv(output_path, header=None, index=None, sep=' ', mode='w')
            
            flip_image_labels(image_path, output_path)"""
