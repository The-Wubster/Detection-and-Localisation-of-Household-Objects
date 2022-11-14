import pandas as pd
from os import listdir

file_path_input = 'Test_Data/labels'
file_path_output = 'Test_Data/labels_rearranged'

for txt_file in listdir(file_path_input):
    if (txt_file.endswith(".txt")):
        image_path = file_path_input + '/' + txt_file
        label_df = pd.read_csv(image_path, sep=" ", names=['Item_id', 'X_min', 'Y_min', 'X_width', 'y_width'])
        for count in range(len(label_df)):
            index_num = label_df.iloc[count]['Item_id']
            if (index_num == 23):
                label_df.at[count, 'Item_id'] = 39
            elif (index_num == 17):
                label_df.at[count, 'Item_id'] = 41
            elif (index_num == 18):
                label_df.at[count, 'Item_id'] = 45
            elif (index_num == 30):
                label_df.at[count, 'Item_id'] = 56
            elif (index_num == 26):
                label_df.at[count, 'Item_id'] = 57
            elif (index_num == 32):
                label_df.at[count, 'Item_id'] = 59
            elif (index_num == 31):
                label_df.at[count, 'Item_id'] = 60
            elif (index_num == 19):
                label_df.at[count, 'Item_id'] = 61
            elif (index_num == 27):
                label_df.at[count, 'Item_id'] = 62
            elif (index_num == 35):
                label_df.at[count, 'Item_id'] = 63
            elif (index_num == 11):
                label_df.at[count, 'Item_id'] = 68
            elif (index_num == 10):
                label_df.at[count, 'Item_id'] = 69
            elif (index_num == 13):
                label_df.at[count, 'Item_id'] = 70
            elif (index_num == 15):
                label_df.at[count, 'Item_id'] = 71
            elif (index_num == 9):
                label_df.at[count, 'Item_id'] = 72
            else:
                pass
        output_path = file_path_output + '/' + txt_file    
        label_df.to_csv(output_path, header=None, index=None, sep=' ', mode='w')
