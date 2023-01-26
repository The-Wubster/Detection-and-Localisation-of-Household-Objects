# Skripsie
## Introduction
Repository for storing files related to Skripsie. These files include adjusted validation and detection scripts from the YOLOv5 and YOLOv7 repositories at https://github.com/ultralytics/yolov5 and https://github.com/WongKinYiu/yolov7 respectively as well as utility files to process data and successfully allow for a language interface to be integrated with the YOLO models for use in household robotic applications. The models have been trained to detect 35 household items. 

## Description
### build_data frame_by_row	
Test Programme to construct a data frame row by row.

### class_label_count	
Counts the number of occurrences of labels for each class in a group of images.

### file_dir_to_list	
Locates all images and labels for those images in a specified folder and then removes all images in the folder without labels.

### image_augmentations	
Performs specified transformations and augmentations required by the user on the dataset.

### img_to_array	
Converts all image names in a directory to an array.

### object_induction_test	
Contains functions for object enrolment and object detection specified by the user. These custom functions are called in the “detect.py” script of a YOLO model and allow the user to interact with the vision system through speech or text. The speech interface makes use of five-shot object enrolment to ensure a more robust system.

### resize_images_and_labels	
Allows user to resize a dataset to a specified size required by the object detector. 

### run_metrics_processing	
Allows user to visualize performance metrics of multiple computer vision models in plots. This script converts .csv files outputted during model validation to a specified format so that the run metrics may be added to a single .csv file which compares the performance of every model which has been validated.

### textfile_to_data frame	
This script converts contents of a text file to a data frame for all text files in a directory. This script remaps the labels from their index in the custom dataset to the MS COCO index. The resulting data frame is saved to a text file. This text file is used during validation of a model trained on the MS COCO dataset. In our case, this is the benchmark.

### textfile_to_list	
Converts contents of a text file to a list.

### validation_shell_script	
Runs multiple validation scripts at different confidence and IOU thresholds for different models specified by the user.

## Dataset and Model Weights
The dataset and weights of the top 5 YOLO models implemented in this project can be found at: 

## Environment Setup
To setup the environment for the object detector and language interface one should follow the instructions at https://github.com/ultralytics/yolov5 or https://github.com/WongKinYiu/yolov7. Additionally, one should import the speech_recognition package. Furthermore, one should ensure all required files are located in the same repository as the YOLO model.

## License

Copyright (c) 2022 Stellenbosch University

This data is released under a Creative Commons Attribution-ShareAlike 
license
(<http://creativecommons.org/licenses/by-sa/4.0/>).
