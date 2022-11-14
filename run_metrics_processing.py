from typing import final
from numpy.lib.function_base import append
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

path_to_run_metrics = '/Users/ryanmildenhall/Desktop/Skripsie_Topics/Final/Yolov5/yolov5/Run_metrics_combined'
file_path_output = '/Users/ryanmildenhall/Desktop/Skripsie_Topics/Final/Yolov5/yolov5/Run_metrics1'
file_path_output1 = '/Users/ryanmildenhall/Desktop/Skripsie_Topics/Final/Yolov5/yolov5/Run_metrics_comparing_to_pretrained'
temp_df = pd.DataFrame(columns=('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95', 'F1-Score'))
final_metric_file_path = '/Users/ryanmildenhall/Desktop/Skripsie_Topics/Final/Model Run Metrics.csv'
final_metric_file_path1 = '/Users/ryanmildenhall/Desktop/Skripsie_Topics/Final/Model Run Metrics1 (old).csv'
final_df = pd.read_csv(final_metric_file_path, sep=';')

#####################################################################################################################################################################
def process_name(file_name):
    metric_list = []
    count = 0

    while len(file_name) > 6:
        count += 1
        dash_index = file_name.find('-')
        if count == 3:
            metric_list.append(file_name[2:dash_index-5])
        else:
            metric_list.append(file_name[:dash_index])
        file_name = file_name[dash_index+1:]
    return metric_list

#####################################################################################################################################################################
process_metrics = 0
COCO_subset = 1

if process_metrics == 1:
    for csv_file in listdir(path_to_run_metrics):
        if (csv_file.endswith(".csv")):
            image_path = path_to_run_metrics + '/' + csv_file
            metric_df = pd.read_csv(image_path)
            metric_list = process_name(csv_file)

            for count in range(len(metric_df)):
                if (metric_list[4] == "Coco Pretrained"):
                    if (metric_df.iloc[count][0] in ['bottle', 'cup', 'bowl', 'chair', 'couch', 'bed', 
                    'dining table', 'toilet', 'tv', 'laptop', 'microwave', 'oven', 'toaster', 'sink', 
                    'refrigerator']): 
                        temp_df = temp_df.append(metric_df.iloc[count])
                else:
                    if COCO_subset ==1:
                        if (metric_df.iloc[count][0] in ['Bottle', 'Cup', 'Bowl', 'Chair', 'Couch', 'Bed', 
                        'Table', 'Toilet', 'Television', 'Computer', 'Microwave', 'Oven', 'Toaster', 'Sink', 
                        'Fridge']):
                            temp_df = temp_df.append(metric_df.iloc[count])
                    else:
                        temp_df = temp_df.append(metric_df.iloc[count])

            output_path = file_path_output1 + '/' + csv_file  
            precision_mean = temp_df['Precision'].mean()
            recall_mean = temp_df['Recall'].mean()
            map5_mean = temp_df['mAP@.5'].mean()
            map95_mean = temp_df['mAP@.5:.95'].mean()
            f1_mean = temp_df['F1-Score'].mean()
            temp_df.loc[len(temp_df)] = ['Averages:', ' ', ' ', precision_mean, recall_mean, map5_mean, map95_mean, f1_mean]
            temp_df.to_csv(output_path, index=False, mode='w')
            temp_df = temp_df.iloc[0:0]

            metric_list.extend([precision_mean, recall_mean, map5_mean, map95_mean, f1_mean])
            metric_list = metric_list[1:]
            final_df.loc[len(final_df)] = metric_list

    final_df.to_csv(final_metric_file_path1, index=False, mode='w')

##############################################################################################################################
plot_metrics = 1
const_iou = 1  #If constant IOU not used then a constant confidence of 0.001 is assumed
twin_plot = 1


if plot_metrics == 1:
    plotting_df = pd.read_csv(final_metric_file_path1)
    #models_to_compare = ['yolov5_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov5_full_dataset_frozen_backbone_from_pretrained_weights', 'yolov7_full_dataset_from_scratch', 'yolov7_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov5_full_dataset_no_frozen_layers_from_scratch', 'yolov5_full_dataset_no_frozen_layers_from_scratch', 'yolov7_tiny_full_dataset_from_scratch', 'yolov7_tiny_full_dataset_no_frozen_from_pretrained', 'yolov7_full_dataset_backbone_frozen_from_pretrained_weights', 'yolov5_full_dataset_frozen_backbone_and_head_from_pretrained_weights']
    #models_to_compare = ['yolov5s', 'yolov5_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov5_full_dataset_no_frozen_layers_from_scratch', 'yolov7_full_dataset_from_scratch', 'yolov7_full_dataset_no_frozen_layers_from_pretrained_weights'] #, 'yolov7_tiny_full_dataset_no_frozen_from_pretrained', 'yolov7_tiny_full_dataset_from_scratch']
    #models_to_compare = ['yolov5s', 'yolov5_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov5_full_dataset_frozen_backbone_from_pretrained_weights', 'yolov5_full_dataset_frozen_backbone_and_head_from_pretrained_weights'] #, 'yolov7_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov7_full_dataset_backbone_frozen_from_pretrained_weights']
    #models_to_compare = ['yolov5s', 'yolov5_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov5_full_dataset_no_frozen_layers_from_scratch', '', '']
    #models_to_compare = ['yolov5s', 'yolov5s_full_dataset_frozen_backbone_from_pretrained_aug', 'yolov5_full_dataset_frozen_backbone_from_pretrained_weights', 'yolov7_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov7_full_dataset_from_pretrained_all_augment', 'yolov7_tiny_from_pretrained_all_augment', 'yolov7_tiny_full_dataset_no_frozen_from_pretrained']
    #models_to_compare = ['yolov5s', 'yolov5s_full_dataset_from_scratch_all_augs', 'yolov5_full_dataset_no_frozen_layers_from_scratch', 'yolov7_full_dataset_from_scratch', 'yolov7_full_dataset_from_scratch_augment', 'yolov7_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov7_full_dataset_from_pretrained_all_augment'] #, 'yolov7_tiny_from_scratch_all_augment', 'yolov7_tiny_full_dataset_from_scratch']
    #models_to_compare = ['yolov5s', 'yolov5_full_dataset_frozen_backbone_from_pretrained_weights', 'yolov5s_full_dataset_frozen_backbone_from_pretrained_aug', 'yolov7_full_dataset_from_scratch', 'yolov7_full_dataset_from_scratch_augment', 'yolov7_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov7_full_dataset_from_pretrained_all_augment']    
    
    models_to_compare = ['yolov5s', 'yolov5_full_dataset_frozen_backbone_from_pretrained_weights', 'yolov5s_frozenbackbone_halfdataset_100', 'yolov7_tiny_full_dataset_no_frozen_from_pretrained', 'yolov7_tiny_no_frozen_half_dataset_from_pretrained_16', 'yolov7_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov7_no_frozen_half_dataset_from_pretrained_batch_16']
    #models_to_compare = ['yolov7_tiny_no_frozen_half_dataset_from_pretrained_batch_8', 'yolov7_tiny_no_frozen_half_dataset_from_pretrained_16', 'yolov7_tiny_no_frozen_half_dataset_from_pretrained_batch_32', 'yolov7_tiny_no_frozen_half_dataset_from_pretrained_batch_48']
    #models_to_compare = ['yolov5s', 'yolov7_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov7_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov7_tiny_full_dataset_frozen_backbone_from_pretrained', 'yolov7_tiny_full_dataset_no_frozen_from_pretrained']
    #models_to_compare = ['yolov5s', 'yolov5_full_dataset_no_frozen_layers_from_scratch', 'yolov5s_full_dataset_from_scratch_all_augs', 'yolov7_full_dataset_from_scratch', 'yolov7_full_dataset_from_scratch_augment', 'yolov7_tiny_full_dataset_no_frozen_from_pretrained', 'yolov7_tiny_from_pretrained_all_augment']
    #models_to_compare = ['yolov5s', 'yolov5_full_dataset_frozen_backbone_from_pretrained_weights', 'yolov5_full_dataset_no_frozen_layers_from_scratch', 'yolov5_full_dataset_no_frozen_layers_from_pretrained_weights', 'yolov5s6_full_dataset_frozen_backbone_from_pretrained', 'yolov5s6_full_dataset_from_scratch', 'yolov5s6_full_dataset_no_frozen_from_pretrained']
    #models_to_compare = ['yolov5s', 'yolov5s6_full_dataset_from_scratch', 'yolov5s6_full_dataset_from_scratch_all_augs', 'yolov5s6_full_dataset_no_frozen_from_pretrained', 'yolov5s6_full_dataset_from_pretrained_all_augs']
    #models_to_compare = ['yolov5s', 'yolov7_tiny_from_pretrained_all_augment', 'yolov7_full_dataset_from_pretrained_all_augment', 'yolov5s6_full_dataset_frozen_backbone_from_pretrained', 'yolov5s6_full_dataset_from_scratch_all_augs', 'yolov7_full_dataset_from_scratch_augment', 'yolov5s6_full_dataset_from_pretrained_all_augs']
    #models_to_compare = ['yolov5s']
    if const_iou == 1:
        temp_df = plotting_df[plotting_df['Confidence Threshold'] != 0.001]
        temp_df = temp_df[temp_df['Specific Model'].isin(models_to_compare)]
    else:
        temp_df = plotting_df[plotting_df['Confidence Threshold'] == 0.001]
        temp_df = temp_df[temp_df['Specific Model'].isin(models_to_compare)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    if twin_plot == 1:
        ax2=ax.twinx()
        y_var2 = "Precision"  # Options include: F1 Score, Precision, Recall, mAP@.5:.95, mAP@0.5
    y_var = "F1 Score"  # Options include: F1 Score, Precision, Recall, mAP@.5:.95, mAP@0.5
    if const_iou == 1:
        x_var = "Confidence Threshold" # Options include: IOU Threshold, Confidence Threshold
    else:
        x_var = "IOU Threshold" # Options include: IOU Threshold, Confidence Threshold 

    legend_var = 'Specific Model'
    legend_col = temp_df[legend_var].unique()

    #legend_labels = ['', '', '', '', '', '', '', '', '']

    for i, colour_var in enumerate(legend_col):
        temp_df1 = temp_df[temp_df[legend_var] == colour_var].sort_values(x_var)
        print(colour_var)
        ax.plot(x_var, y_var, label=colour_var, data=temp_df1)
        if twin_plot == 1:
            ax2.plot(x_var, y_var2, label=colour_var, data=temp_df1, linestyle='dashed')

    if const_iou == 1:
        if twin_plot == 1:
            ax.set_title(y_var + " vs " + y_var2 + " of Different Models with Constant IOU Threshold") 
        else: 
            ax.set_title(y_var + " of Different Models with Constant IOU Threshold") 
    else:
        if twin_plot == 1:
            ax.set_title(y_var + " vs " + y_var2 + " of Different Models with Confidence Threshold of 0.001") 
        else:
            ax.set_title(y_var + " of Different Models with Confidence Threshold of 0.001")

    ax.set_xlabel(x_var)  
    ax.set_ylabel(y_var + " (Solid Line)") 
    if twin_plot == 1:
        ax2.set_ylabel(y_var2 + " (Dashed Line)")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_order = [1,0,2,3,4,5,6] #[1, 0, 2, 3, 4, 6, 5] #[1, 0, 2, 4, 3, 6, 5] #[0, 1, 2, 3]
    ax.legend([handles[index] for index in legend_order],[labels[index] for index in legend_order], loc='lower left')
    #ax.legend(); 
    plt.show()

##########################################################################################################################
remap_index = 0

if remap_index == 1:
    my_classes = ['Window', 'Door', 'Light', 'Carpet', 'Cupboard', 'Mirror', 'Curtains', 'Dustbin', 'Fridge', 'Oven', 'Microwave', 'Kettle', 'Toaster', 'Tap', 'Sink', 'Pot', 'Cup', 'Bowl', 'Toilet', 'Dishwasher', 'Cloth', 'Jug', 'Bottle', 'Coffee Machine', 'Couch', 'Television', 'Pillow', 'Fire Place', 'Chair', 'Table', 'Bed', 'Draws', 'Desk', 'Computer', 'Flowers']
    for csv_file in listdir(path_to_run_metrics):
        if (csv_file.endswith(".csv")):
            image_path = path_to_run_metrics + '/' + csv_file
            metric_df = pd.read_csv(image_path)
            metric_df['Class'] = metric_df['Class'].astype(str)

            for count in range(len(metric_df)):
                metric_df.at[count, 'Class'] = my_classes[count]

            output_path = path_to_run_metrics + '/' + csv_file
            print(metric_df)
            metric_df.to_csv(output_path, index=False, mode='w')

################################################################################################################
