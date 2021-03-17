import cv2
import os
import numpy as np
import glob


# TODO Get class repartition : return a vector of the percentage representation of each class
# TODO Refactoring

def compute_class_proportion():
    create_tree_structure()
    dictionaries = create_dict_from_sources()

    index = 0

    for dictionary in dictionaries:
        print(dictionary["source"])
        image = cv2.imread(dictionary["source"])
        labels_img_threshold = []
        labels = list(dictionary.keys())
        labels.remove('source')
        for i, label in enumerate(labels):
            label_img = cv2.imread(dictionary[label], 0)
            _, label_img = cv2.threshold(label_img, 127, 255, cv2.THRESH_BINARY_INV)
            label_img = np.where(label_img == 255, i+1, 0)
            labels_img_threshold.append(label_img)

        # create label image
        labels = labels_img_threshold
        label = labels[0]
        for i in range(1, len(labels)):
            label = label + labels[i]
            for j in range(i+i+1, i+1, -1):
                label = np.where(label == j, i+1, label)

        hist, bins = np.histogram(label.ravel(), 256, [0, 256])
        hist = hist[hist != 0]
        print(hist)
        total = sum(hist)
        proportion = hist/total
        print(proportion)


def create_tree_structure():
    root = "dataset"
    directories = ["train/GT", "train/Img", "val/GT", "val/Img"]

    for directory in directories:
        path = os.path.join(root, directory)
        try:
            os.makedirs(path)
        except os.error:
            files = glob.glob(os.path.join(path, "*"))
            for f in files:
                os.remove(f)
        finally:
            pass


def create_dict_from_sources():
    path = "source"
    images = glob.glob(os.path.join(path, "Img_*"))
    label = os.path.join(path, "label")
    dictionaries = []

    images.sort()
    last_index = int(os.path.splitext(images[-1])[0].split("_")[1])
    print(images)

    for index in range(1, last_index + 1):
        element = {}
        path_by_index = os.path.join(path, "Img_" + str(index).zfill(3))
        images_by_index = [x for x in images if x.startswith(path_by_index)]
        for label in images_by_index:
            if len(label.split("_")) == 2:
                element["source"] = label
            else:
                label_number = int(os.path.splitext(label)[0].split("_")[-1])
                element["label" + str(label_number)] = label
        dictionaries.append(element)
    return dictionaries


def create_dataset():
    PROCESS = True
    final_height = 512
    final_width = 1024

    create_tree_structure()
    dictionaries = create_dict_from_sources()

    index = 0

    for dictionary in dictionaries:
        print(dictionary["source"])
        image = cv2.imread(dictionary["source"])
        labels_img_threshold = []
        labels = list(dictionary.keys())
        labels.remove('source')
        for i, label in enumerate(labels):
            label_img = cv2.imread(dictionary[label], 0)
            _, label_img = cv2.threshold(label_img, 127, 255, cv2.THRESH_BINARY_INV)
            label_img = np.where(label_img == 255, i+1, 0)
            labels_img_threshold.append(label_img)

        height, width = image.shape[:2]
        crop_height = height // final_height
        crop_width = width // final_width

        # create label image
        labels = labels_img_threshold
        label = labels[0]
        print(len(labels))
        for i in range(1, len(labels)):
            print(i, " : i")
            label = label + labels[i]
            for j in range(i+i+1, i+1, -1):
                print(j, " : j")
                label = np.where(label == j, i+1, label)

        # create images
        for i in range(crop_height):
            for j in range(crop_width):
                index += 1
                path_img_cl = os.path.join("dataset/train/Img/", str(index).zfill(3) + ".png")
                image_crop = image[i * final_height:(i + 1) * final_height, j * final_width:(j + 1) * final_width]

                # classical
                path_label = os.path.join("dataset/train/GT/", str(index).zfill(3) + ".png")
                label_crop = label[i * final_height:(i + 1) * final_height, j * final_width:(j + 1) * final_width]

                # Save
                cv2.imwrite(path_img_cl, image_crop)
                cv2.imwrite(path_label, label_crop)
                print("image ", index)


if __name__ == "__main__":
    compute_class_proportion()
    create_dataset()
