import pandas as pd 
import os
import datetime

metric_df = pd.DataFrame(columns=('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95'))
for i in range(7):
    print(i)
    metric_df.loc[i] = [i, i * 1, i * 2, i*3, i*4, i*5, i*6]

current_dir = os.getcwd()
print(current_dir + '/Run_metrics')
current_time = datetime.datetime.now()
metric_df.to_csv(current_dir + '/Run_metrics/' + "metrics " + str(current_time) + ".csv")

print(metric_df)