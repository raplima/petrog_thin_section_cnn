# Rafael Pires de Lima
# February 2019
# Evaluation of CNN models generated with cnn_processing.py
# This file includes functions to:
# predict the resulting classification of a single file using the CNN model provided and generate a figure
# predict the resulting classification of a folder with subfolders using the CNN model provided and generate a csv file
# analyse the predictions

import pickle

import matplotlib.image as mpimg
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from matplotlib import style

style.use("seaborn")


def make_fig(res, model_labels, im):
    """Makes a matplotlib figure with image and classification results.
        Args:
            res: np.array with shape (1, number of labels) containing the results of the classification
                performed by a CNN model.
            im: String path to a single image (the image that generated res)
            model_labels: A python list with the real name of the classes
        Returns:
            fig: a matplotlib figure.
    """

    # set up figure
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    ax[0].axis("off")

    ax[1].set_xlabel("Probability")

    fig.set_size_inches(w=14, h=6)

    # get the class position for later plotting:
    x_pos = [elem for elem, _ in enumerate(model_labels)]

    # read image for plotting:
    img = mpimg.imread(im)
    ax[0].axis("off")
    ax[0].imshow(img)

    ax[1].barh(x_pos, res[0][:], color='grey')
    ax[1].set_xlabel("Probability", fontsize=16)
    ax[1].tick_params(labelsize=14)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].yaxis.grid(False)
    ax[1].set_yticks(x_pos)
    ax[1].set_yticklabels('')

    for y, lab in enumerate(model_labels):
        ax[1].text(0, y, lab.replace('_', ' '), verticalalignment='center', fontsize=18)

    plt.show(fig)


def label_one(path_img, path_model):
    """Labels (classifies) a single image based on a retrained CNN model.
      Args:
        path_img: String path to a single image.
        path_model: String path to the model to be used for classification.
      Returns:
        pred: the resulting prediction using the model
      """
    # load the model:
    model = load_model(path_model)

    # get model input parameters:
    img_height = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    img_width = model.layers[0].get_output_at(0).get_shape().as_list()[2]

    # load the image
    img = image.load_img(path_img, target_size=(img_height, img_width))

    # save as array and rescale
    x = image.img_to_array(img) * 1. / 255

    # predict the value
    pred = model.predict(x.reshape(1, img_height, img_width, 3))
    return pred


def label_folder(path_folder, path_model):
    """Labels (classifies) a folder containing subfloders based on a retrained CNN model.
      Args:
        path_folder: String path to a folder containing subfolders of images.
        path_model: String path to the model to be used for classification.
      Returns:
        List: a numpy array with predictions (pred) and the file names of the images classified (generator.filenames)
      """
    # load the model:
    model = load_model(path_model)

    # get model input parameters:
    img_height = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    img_width = model.layers[0].get_output_at(0).get_shape().as_list()[2]

    datagen = ImageDataGenerator(rescale=1. / 255)

    # flow from directory:
    generator = datagen.flow_from_directory(
        path_folder,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    if len(generator) > 0:
        # if data file is structured as path_folder/classes, we can use the generator:
        pred = model.predict_generator(generator, steps=len(generator), verbose=1)
    else:
        # the path_folder contains all the images to be classified
        # TODO: if problems arise
        pass

    return [pred, generator.filenames]


if __name__ == '__main__':
    print("Starting...")

    # for model selection parameters
    options_dict = {
        'Xception': (299, 299, 3),
        'VGG16': (224, 224, 3),
        'VGG19': (224, 224, 3),
        'ResNet50': (224, 224, 3),
        'InceptionV3': (299, 299, 3),
        'InceptionResNetV2': (299, 299, 3),
        'MobileNet': (224, 224, 3),
        'MobileNetV2': (224, 224, 3),
        'DenseNet121': (224, 224, 3),
        'NASNetLarge': (331, 331, 3)
    }

    # train and validation data folders
    train_data_dir = '../Data/PP_mc_wb_train'
    validation_data_dir = '../Data/PP_mc_wb_validation'
    test_data_dir = '../Data/PP_mc_wb_test'

    test_image = '../Data/PP_mc_wb_test/Argilaceous_siltstone/23_10X PP 2_bl.jpg'

    # model folder
    model_dir = './runs/models/'
    ####################################################
    # choose model architecture with weights coming from ImageNet training:

    models_list = ['MobileNetV2', 'VGG19',
                   'InceptionV3', 'ResNet50']

    for m in models_list:
        print(m)

        # model path:
        m_path = '{}{}{}'.format(model_dir, m, '.hdf5')
        m_dict = '{}{}{}'.format(model_dir, m, '_dict_l')
        # open the label dictionary
        with open(m_dict, 'rb') as f:
            m_labels = pickle.load(f)

        # image height and width ("picture size for the model"):
        height = options_dict[m][0]
        width = options_dict[m][1]

        # calling the functions:
        #predicted = label_one(test_image, m_path)
        #make_fig(predicted, m_labels, test_image)

        # classify folder
        res = label_folder(test_data_dir, m_path)

        # save results as dataframe
        df = pd.DataFrame(res[0], columns=m_labels)
        df['file'] = res[1]

        df.to_csv('{}/{}{}'.format(test_data_dir, m, '.csv'))
