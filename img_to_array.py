import glob
import cv2

img_List = [cv2.imread(file) for file in glob.glob("Validation_Data/Whytebank/*.jpeg")]

print(img_List)