textfile_to_proocess = open("Normal_Dataset/Training_Data/labels.txt", "r")
label_data = textfile_to_proocess.read()
label_list = label_data.split("\n")
print(label_list)
textfile_to_proocess.close()