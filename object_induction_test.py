import os
import pandas as pd
import cv2
import numpy as np
import sys
import speech_recognition as sr

os.chdir("/Users/ryanmildenhall/Desktop/Skripsie_Topics/Final/Yolov5/datasets/skripsie_dataset/Validation_Data")
image_input_directory = './images'
label_input_directory = './labels'
sample_list = os.listdir(image_input_directory)
item_to_detect = "empty"
item_to_detect_index = 0
display_width = 500
display_length = 500

def get_input_from_user():
    input_method = '0'
    processed_text = []
    while input_method != ['1', '2']:       
        input_method = input("Enter your input method; Audio (1) or Text (2).\n")

        if input_method == '1': #If input is in audio format.
            file_path = "Recordings/Kettle.wav"     # Audio file of item name.
            speech_recognizer = sr.Recognizer()

            with sr.AudioFile(file_path) as source:
                text_as_audio = speech_recognizer.listen(source)
                processed_text = speech_recognizer.recognize_google(text_as_audio).upper()
                print('Converting speech to text...')
                # print(processed_text)
            break

        elif input_method == '2':   #If input is in text format.
            for speech_count in range(5):
                processed_text.append(input("What is this item called?\n").upper())
            break
        else:
            print("Please enter a valid input method.")
    
    return processed_text, input_method

def get_item_names():
    recorded_labels = []
    complete_flag = 0
    num_classes = int(input('How many classes are in the dataset (enter a number)?'))
    input_type = int(input("Enter your input method; Audio (1) or Text (2).\n"))
    num_rows = np.arange(num_classes, dtype=int)
    item_df = pd.DataFrame(index=num_rows, columns=["Item ID", "Item Name"]) 
    for item in sample_list:
        if ".jpeg" in item and len(recorded_labels) < num_classes:
            # Insert Loop for moving through images here:
            image_path = image_input_directory + '/' + item 
            label_path = label_input_directory + '/' + item[:8] + ".txt" 
            label_df = pd.read_csv(label_path, sep=" ", names=['Item_id', 'x_center', 'y_center', 'x_width', 'y_width'])
            i_count = 0

            for count in range(len(label_df)):
                x_center = label_df.loc[count, 'x_center'] 
                y_center = label_df.loc[count, 'y_center'] 
                x_width = label_df.loc[count, 'x_width']
                y_width = label_df.loc[count, 'y_width']
                item_id = int(label_df.loc[count, 'Item_id'])
                if item_id not in recorded_labels and i_count < 5:
                    im = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if (y_width > 0.05) and (x_width > 0.05):                       
                        x_min = round((x_center - x_width/2) * display_width)
                        y_min = round((y_center - y_width/2) * display_length)
                        x_max = round((x_center + x_width/2) * display_width)
                        y_max = round((y_center + y_width/2) * display_length)
                        im = cv2.resize(im, (display_width, display_length))
                        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                        cv2.imshow("current_image", im)
                        cv2.waitKey(1)
                        recorded_labels.append(item_id)
                        item_df.at[item_id, 'Item ID'] = item_id
                        item_df.at[item_id, 'Item Name'] = get_input_from_user()
                        if len(recorded_labels) == num_classes:
                            print("100%")
                            if input_type == 1:
                                item_df.to_csv("Item_Names_speech.csv", index=False)
                            else:
                                item_df.to_csv("Item_Names_text.csv", index=False)
                            complete_flag = 1
                        i_count += 1
                        cv2.destroyAllWindows()
    if complete_flag == 0:
        print("Not all items are recorded.")
        if input_type == 1:
            item_df.to_csv("Item_Names_speech.csv", index=False)
        else:
            item_df.to_csv("Item_Names_text.csv", index=False)

################################################################################################################################
