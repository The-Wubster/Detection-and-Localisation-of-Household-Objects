import os

my_path = "Volumes/Ryan_Extern/Normal_Dataset_Full/Validation_Data/images"
dirListing = os.listdir(my_path)
image_list = []

for item in dirListing:
    if ".jpeg" in item:
        image_list.append(item)

my_path1 = "Volumes/Ryan_Extern/Normal_Dataset_Full/Validation_Data/labels"
dirListing1 = os.listdir(my_path1)
label_list = []

for item in dirListing1:
    if ".txt" in item:
        label_list.append(item)

image_list = [z[4:8] for z in image_list]
label_list = [z[4:8] for z in label_list]
main_list = list(set(label_list) - set(image_list))
main_list.sort()

print(main_list)

# Remove labels without images:
for x in range(len(main_list)):
    main_list[x] = "/IMG_" + main_list[x] + ".txt"
    os.remove(my_path1 + main_list[x])

"""# Remove images without labels:
for x in range(len(main_list)):
    main_list[x] = "/IMG_" + main_list[x] + ".jpeg"
    os.remove(my_path + main_list[x])"""