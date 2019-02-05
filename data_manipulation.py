# Rafael Pires de Lima
# December 2018
# file managing thin section images

import os
import shutil
import numpy as np
from PIL import Image
import random

# to color balance the images
import cv2
import simple_cb as cb


def multi_crop(path_in, path_out, input_shape=(1292, 968), target_shape=(644, 644), bottom_right=False):
    """
    Function makes a three crop (top left, top center, top right, bottom left, bottom center)
    and saves images in path_out.
    We discard bottom right as it might contain the scale and we do not want to add that to the training data.
    :param path_in: string, path to folder containing subfolders with images
    :param path_out: string, path to folder to save the images
    :param target_shape: image shape
    :param bottom_right: create/keep bottom right crop (not using for training as it might contain scale)
    :return:
    """

    print('Starting multi_crop')
    # Create the folder that will hold all images:
    if os.path.exists(path_out):
        shutil.rmtree(path_out, ignore_errors=True)
    os.makedirs(path_out)

    # get the classes
    folders = os.listdir(path_in)

    # get center point
    x_c = np.int(input_shape[0] / 2.)

    # create dictionary to be used in cropping loop:
    # values define the cropping position
    if bottom_right:
        # if user wants to keep bottom right crop, we add it to the dictionary
        new_imgs = {'tl': (0, 0, target_shape[0], target_shape[1]),
                    'tc': (x_c - np.int(target_shape[0] / 2.), 0,
                           x_c + np.int(target_shape[0] / 2.), target_shape[1]),
                    'tr': (input_shape[0] - target_shape[0], 0,
                           input_shape[0], target_shape[1]),
                    'bl': (0, input_shape[1] - target_shape[1],
                           target_shape[0], input_shape[1]),
                    'bc': (x_c - np.int(target_shape[0] / 2.), input_shape[1] - target_shape[1],
                           x_c + np.int(target_shape[0] / 2.), input_shape[1]),
                    'br': (input_shape[0] - target_shape[0], input_shape[1] - target_shape[1],
                           input_shape[0], input_shape[1])}
    else:
        new_imgs = {'tl': (0, 0, target_shape[0], target_shape[1]),
                    'tc': (x_c - np.int(target_shape[0] / 2.), 0,
                           x_c + np.int(target_shape[0] / 2.), target_shape[1]),
                    'tr': (input_shape[0] - target_shape[0], 0,
                           input_shape[0], target_shape[1]),
                    'bl': (0, input_shape[1] - target_shape[1],
                           target_shape[0], input_shape[1]),
                    'bc': (x_c - np.int(target_shape[0] / 2.), input_shape[1] - target_shape[1],
                           x_c + np.int(target_shape[0] / 2.), input_shape[1])}

    # uses the path_in and walks in folders to crop images
    for folder in folders:
        print('----{}'.format(folder))
        os.mkdir(path_out + os.sep + folder)
        lst = os.listdir(path_in + os.sep + folder)

        images = [item for item in lst if item.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for file in images:

            # open image
            ori = Image.open(path_in + os.sep + folder + os.sep + file)

            for k in new_imgs:
                new_name = '{}_{}{}'.format(os.path.splitext(file)[0], k, os.path.splitext(file)[1])
                # crop image
                cropped = ori.crop(new_imgs[k])
                # save cropped image with new resolution
                img = cropped.resize(target_shape, Image.ANTIALIAS)
                img.save(path_out + os.sep + folder + os.sep + new_name)
    print('multi_crop complete\n')


def wbalance(path_in, path_out, list_folders):
    """
    Receives path in with subfolders. Saves white balanced images in path_in_wb
    :param path_in: the "root" folder. Folders listed in list_folders need to be here.
    :param list_folders: the list of folders to be analyzed.
    :return: No returns, creates folders with white balanced images
    """

    print('Starting white balance')
    # Create the folder that will hold all images:
    if os.path.exists(path_out):
        shutil.rmtree(path_out, ignore_errors=True)
    os.makedirs(path_out)

    # uses the path_in and walks in folders to square crop and reduce image
    for folder in list_folders:
        print('----{}'.format(folder))
        os.mkdir(path_out + os.sep + folder)
        lst = os.listdir(path_in + os.sep + folder)

        images = [item for item in lst if item.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for file in images:
            # open image
            img = cv2.imread(path_in + os.sep + folder + os.sep + file)
            # use simplest white balance
            out = cb.simplest_cb(img, 1)
            # save file
            cv2.imwrite(path_out + os.sep + folder + os.sep + file, out)
    print('White balance complete\n')


def train_val_test(path, val_p=0.1, test_p=0.1):
    """Splits a single root folder with subfolderss (classes) containing images of different classes
    into train/validation/test sets.
      Args:
        path: String path to a root folder containing subfolders of images.
        val_p: Float proportion of the images to be reserved for validation.
        test_p: Float proportion of the images to be reserved for test.
      Returns:
        No returns.
      """

    print('Starting train_val_test (splitting data)')
    # for reproducibility
    random.seed(1234)
    # print(random.randint(1, 10000))  # 7221

    print("Splitting data in " + path + " into training/validation/test")

    # uses the file path and walks in folders to select training and test data
    lst = os.listdir(path)
    # assume elements with "." are files and remove from list
    folders = [item for item in lst if "." not in item]

    # create folder to save training/validation/test data:
    path_train = os.path.dirname(path) + os.sep + path.split(os.sep)[-1] + '_train'
    if os.path.exists(path_train):
        shutil.rmtree(path_train, ignore_errors=True)
    os.makedirs(path_train)

    path_valid = os.path.dirname(path) + os.sep + path.split(os.sep)[-1] + '_validation'
    if os.path.exists(path_valid):
        shutil.rmtree(path_valid, ignore_errors=True)
    os.makedirs(path_valid)

    if test_p > 0:
        path_test = os.path.dirname(path) + os.sep + path.split(os.sep)[-1] + '_test'
        if os.path.exists(path_test):
            shutil.rmtree(path_test, ignore_errors=True)
        os.makedirs(path_test)

    # for each one of the folders
    for this_folder in folders:
        print('----{}'.format(this_folder))

        # create folder to save cropped image:
        if os.path.exists(path_train + os.sep + this_folder):
            shutil.rmtree(path_train + os.sep + this_folder, ignore_errors=True)
        os.makedirs(path_train + os.sep + this_folder)

        if os.path.exists(path_valid + os.sep + this_folder):
            shutil.rmtree(path_valid + os.sep + this_folder, ignore_errors=True)
        os.makedirs(path_valid + os.sep + this_folder)

        if test_p > 0:
            if os.path.exists(path_test + os.sep + this_folder):
                shutil.rmtree(path_test + os.sep + this_folder, ignore_errors=True)
            os.makedirs(path_test + os.sep + this_folder)

        # separate training and test data:
        # get pictures in this folder:
        lst = os.listdir(path + os.sep + this_folder)

        if len(lst) < 3:
            print("      Not enough data for automatic separation")
        else:
            # shuffle the indices:
            np.random.shuffle(lst)

            # number of test images:
            n_valid = np.int(np.round(len(lst) * val_p))
            n_test = np.int(np.round(len(lst) * test_p))
            if n_valid == 0:
                print("Small amount of data, only one sample forcefully selected to be part of validation data")
                n_valid = 1
            if test_p > 0 and n_test == 0:
                print("Small amount of data, only one sample forcefully selected to be part of test data")
                n_test = 1

            # copy all pictures to appropriate folder
            for t in range(0, n_test):
                shutil.copy2(path + os.sep + this_folder + os.sep + lst[t], path_test + os.sep + this_folder)

            for t in range(n_test, n_test + n_valid):
                shutil.copy2(path + os.sep + this_folder + os.sep + lst[t], path_valid + os.sep + this_folder)
                # print("copied " + lst[t] + " to validation folder");

            for t in range(n_test + n_valid, len(lst)):
                shutil.copy2(path + os.sep + this_folder + os.sep + lst[t], path_train + os.sep + this_folder)
                # print("copied " + lst[t] + " to train folder");

    print('Split data complete\n')


def same_original(path_in, path_search):
    """
    Walks in path_in and searches crops from the same image in path_search. Moves the crops to path_in
    :param path_in: string, path to folder (validation or test) that contains multicrop images
    :param path_search: string, path to folder (training) that might contain other multicrop
    :return:
    """
    print('Starting same original')

    # uses the file path and walks in folders to select training and test data
    lst = os.listdir(path_in)
    # assume elements with "." are files and remove from list
    folders = [item for item in lst if "." not in item]

    # for each one of the folders
    for this_folder in folders:
        print('----{}'.format(this_folder))

        # gets filename
        file_list = os.listdir(path_in + os.sep + this_folder)

        for file in file_list:
            same = [x for x in os.listdir(path_search + os.sep + this_folder) if os.path.splitext(file)[0][:-3] in x]
            for s in same:
                # print(s)
                shutil.move(path_search + os.sep + this_folder + os.sep + s,
                            path_in + os.sep + this_folder + os.sep + s)

    print('Same original complete\n')


def image_augment(path_in):
    """
    Creates rotations of images in subfolders of path_in. Saves image in the same folder
    :param path_in: Root path
    :return:
    """

    print('Starting image augmentation')

    # get the classes
    folders = os.listdir(path_in)

    # uses the path_in and walks in folders to square crop and reduce image
    for folder in folders:
        print('----{}'.format(folder))
        lst = os.listdir(path_in + os.sep + folder)

        for file in lst:
            # open the image:
            # open image
            ori = Image.open(path_in + os.sep + folder + os.sep + file)

            # save rotated versions of original image:
            img = ori.rotate(90)
            img.save(path_in + os.sep + folder + os.sep + 'aug_rot090_' + file)
            img = ori.rotate(180)
            img.save(path_in + os.sep + folder + os.sep + 'aug_rot180_' + file)
            img = ori.rotate(270)
            img.save(path_in + os.sep + folder + os.sep + 'aug_rot270_' + file)

            img = ori.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(path_in + os.sep + folder + os.sep + 'aug_tb_rot000_' + file)
            img = ori.transpose(Image.FLIP_TOP_BOTTOM).rotate(90)
            img.save(path_in + os.sep + folder + os.sep + 'aug_tb_rot090_' + file)
            img = ori.transpose(Image.FLIP_TOP_BOTTOM).rotate(180)
            img.save(path_in + os.sep + folder + os.sep + 'aug_tb_rot180_' + file)
            img = ori.transpose(Image.FLIP_TOP_BOTTOM).rotate(270)
            img.save(path_in + os.sep + folder + os.sep + 'aug_tb_rot270_' + file)

    print('Image augmentation complete\n')


if __name__ == '__main__':

    # multicrop the images
    path_in = '../Data/PP'
    path_out = '../Data/PP_mcrop'

    multi_crop(path_in, path_out)

    # apply white balance
    path_in = '../Data/PP_mcrop'
    path_out = '../Data/PP_mc_wb'
    list_folders = ['Ambiguous',
                    'Argillaceous_siltstone',
                    'Bioturbated_siltstone',
                    'Massive_calcareous_siltstone',
                    'Massive_calcite-cemented_siltstone',
                    'Porous_calcareous_siltstone']

    wbalance(path_in, path_out, list_folders)

    # separate training, validation, test data
    path_in = '../Data/PP_mc_wb'
    train_val_test(path_in, val_p=0.075, test_p=0.075)

    # move the crops from the same original image to validation and test folders
    path_search = '../Data/PP_mc_wb_train'
    path_in = '../Data/PP_mc_wb_validation'
    same_original(path_in, path_search)

    path_in = '../Data/PP_mc_wb_test'
    same_original(path_in, path_search)

    path_search = '../Data/PP_mc_wb_validation'
    same_original(path_in, path_search)

    # use image augmentation for training
    path_in = '../Data/PP_mc_wb_train'
    image_augment(path_in)

    # print number of samples in final data
    # remember ambiguous will not be used
    list_folders = ['Argillaceous_siltstone',
                    'Bioturbated_siltstone',
                    'Massive_calcareous_siltstone',
                    'Massive_calcite-cemented_siltstone',
                    'Porous_calcareous_siltstone']

    list_roots = ['../Data/PP_mc_wb_train',
                  '../Data/PP_mc_wb_validation',
                  '../Data/PP_mc_wb_test']
    for r in list_roots:
        print(r.split('/')[-1])
        for folder in list_folders:
            print('----{}: {} images'.format(folder, len(os.listdir(r + os.sep + folder))))
