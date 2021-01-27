import cv2
import os
import numpy as np
import glob

PROCESS = True

directories = glob.glob("./experimental/*")
directories.extend(glob.glob("./classic/*"))
for directory in directories:
    if os.listdir(directory) != 0:
        files = glob.glob(os.path.join(directory, "*"))
        for f in files:
            os.remove(f)

final_height = 512
final_width = 1024

image = cv2.imread("./source/Img_001.tif")
label_1 = cv2.imread("./source/Img_001_label_1.tif", 0)
_, label_1 = cv2.threshold(label_1, 127, 255, cv2.THRESH_BINARY_INV)
label_2 = cv2.imread("./source/Img_001_label_2.tif", 0)
_, label_2 = cv2.threshold(label_2, 127, 255, cv2.THRESH_BINARY_INV)
label_3 = cv2.imread("./source/Img_001_label_3.tif", 0)
_, label_3 = cv2.threshold(label_3, 127, 255, cv2.THRESH_BINARY_INV)

height, width = image.shape[:2]
print(height, width)
crop_height = height//final_height
crop_width = width//final_width
print(crop_height)
print(crop_width)

# Experimental
label_1 = np.where(label_1 == 0, 1, label_1)
label_1 = np.where(label_1 == 255, 0, label_1)
label_2 = np.where(label_2 == 0, 2, label_2)
label_2 = np.where(label_2 == 255, 0, label_2)
label_3 = np.where(label_3 == 0, 3, label_3)
label_3 = np.where(label_3 == 255, 0, label_3)

# Classical
label = label_1 + label_2
label = np.where(label == 3, 2, label)
label = label + label_3
label = np.where(label == 4, 3, label)
label = np.where(label == 5, 3, label)

if PROCESS:
    index = 0
    for i in range(crop_height):
        for j in range(crop_width):
            index += 1
            path_img_ex = os.path.join("./experimental/Img/", str(index).zfill(3) + ".png")
            path_img_cl = os.path.join("./classic/Img/", str(index).zfill(3) + ".png")
            image_crop = image[i*final_height:(i+1) * final_height, j * final_width:(j+1)*final_width]

            # experimental

            path_label_1 = os.path.join("./experimental/Gt/", str(index).zfill(3) + "_1.png")
            path_label_2 = os.path.join("./experimental/Gt/", str(index).zfill(3) + "_2.png")
            path_label_3 = os.path.join("./experimental/Gt/", str(index).zfill(3) + "_3.png")
            label_1_crop = label_1[i*final_height:(i+1)*final_height, j*final_width:(j+1)*final_width]
            label_2_crop = label_2[i*final_height:(i+1)*final_height, j*final_width:(j+1)*final_width]
            label_3_crop = label_3[i*final_height:(i+1)*final_height, j*final_width:(j+1)*final_width]

            # classical
            path_label = os.path.join("./classic/Gt/", str(index).zfill(3) + ".png")
            label_crop = label[i*final_height:(i+1)*final_height, j*final_width:(j+1)*final_width]

            # Save
            cv2.imwrite(path_img_ex, image_crop)
            cv2.imwrite(path_img_cl, image_crop)
            cv2.imwrite(path_label_1, label_1_crop)
            cv2.imwrite(path_label_2, label_2_crop)
            cv2.imwrite(path_label_3, label_3_crop)
            cv2.imwrite(path_label, label_crop)
            print("image ", index)
