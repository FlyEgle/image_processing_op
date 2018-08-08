import os
import numpy as np
import shutil
from tqdm import tqdm

TRAIN_VALIDATION_RATE = 0.8
TOTAL_DATASET_PATH = "/home/jiangmingchao/Gan_tensorflow/flower_photos/"
TRAIN_DATA_PATH = "/home/jiangmingchao/Gan_tensorflow/flower_dataset/train_dataset/"
VALIDATION_DATA_PATH = "/home/jiangmingchao/Gan_tensorflow/flower_dataset/validation_dataset/"


def evaluate_images_nums(images_path):
    total_images_counts = 0
    image_format = ['jpg', 'png']
    for image_folder in os.listdir(images_path):
        if len(image_folder.split('.')) == 1:
            folder_images_count = 0
            for images in os.listdir(images_path + image_folder):
                if images.split('.')[-1] == image_format[0] or images.split('.')[-1] == image_format[1]:
                    folder_images_count += 1
            print('{} images count :{}'.format(image_folder, folder_images_count))
            total_images_counts += folder_images_count
    print("total images count : {}".format(total_images_counts))


def make_dirs(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def split_train_validation_images(path, train_path, validation_path):
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(validation_path):
        os.mkdir(validation_path)

    for image_folder in os.listdir(path):
        # train_path
        make_dirs(train_path + image_folder)
        # validation_path
        make_dirs(validation_path + image_folder)
        print('process folder: ', image_folder)
        move_number_count = 1
        for images in tqdm(os.listdir(path + image_folder)):
            # split with the rate = 0.8, than move 1 images to validation after 4 images
            step = int(1 / (1 - TRAIN_VALIDATION_RATE))
            if move_number_count % step == 0:
                shutil.copy(path + image_foler + '/' + images,
                            validation_path + image_foler + '/' + images)
            move_number_count += 1
    print("move images have been done")


def move_remain_train_images(path, train_path):
    for image_folder in os.listdir(path):
        print('process folder: ', image_folder)
        for images in tqdm(os.listdir(path + image_folder)):
            shutil.copy(path + image_folder + '/' + images,
                        train_path + image_folder + '/' + images)
    print("move images have been done")


if __name__ == "__main__":
    evaluate_images_nums(TRAIN_DATA_PATH)
    evaluate_images_nums(VALIDATION_DATA_PATH)


