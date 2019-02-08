# Rafael Pires de Lima
# February 2019
# Generate figures of CNN models generated with cnn_processing.py
# This file includes functions to:
# predict the resulting classification of a single file using the CNN model provided and generate a figure

import os
import pickle

import matplotlib.image as mpimg
import pandas as pd
import seaborn as sn
from keras.models import load_model
from keras.preprocessing import image
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.metrics import confusion_matrix

import data_manipulation as dm

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
    #fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    fig = plt.figure(constrained_layout=True)
    spec = plt.GridSpec(ncols=3, nrows=1, figure=fig, wspace=0.1)
    ax = [None]*2
    ax[0] = fig.add_subplot(spec[0, 0])
    ax[1] = fig.add_subplot(spec[0, 1:])

    ax[0].axis("off")

    ax[1].set_xlabel("Probability")

    fig.set_size_inches(w=7, h=3)

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

    # plt.show(fig)
    plt.savefig(
        '{}{}{}{}'.format(os.path.split(os.path.split(im)[0])[0], os.sep, os.path.splitext(os.path.basename(im))[0],
                          '.tif'),
        dpi=600)
    plt.close()


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


if __name__ == '__main__':
    print("Starting...")

    # for model selection parameters
    options_dict = {
        'ResNet50_fine_tuned': (224, 224, 3),
    }

    # train and validation data folders
    train_data_dir = '../Data/PP_mc_wb_train'
    validation_data_dir = '../Data/PP_mc_wb_validation'
    test_data_dir = '../Data/PP_mc_wb_test' + os.sep

    test_image_list = ['{}{}{}'.format('Argillaceous_siltstone', os.sep, '25_10X PP 2_tc.jpg'),
                       '{}{}{}'.format('Argillaceous_siltstone', os.sep, '23_10X PP 2_bc.jpg'),

                       '{}{}{}'.format('Bioturbated_siltstone', os.sep, '10_10X PP 2_tr.jpg'),
                       '{}{}{}'.format('Bioturbated_siltstone', os.sep, '38_10X PP 2_bc.jpg'),

                       '{}{}{}'.format('Massive_calcareous_siltstone', os.sep, '30_10X PP S_bc.jpg'),
                       '{}{}{}'.format('Massive_calcareous_siltstone', os.sep, '31_10X PP S_tr.jpg'),

                       '{}{}{}'.format('Massive_calcite-cemented_siltstone', os.sep, '14_10X PP 1_bc.jpg'),
                       '{}{}{}'.format('Massive_calcite-cemented_siltstone', os.sep, '100_10X PP 2_bc.jpg'),

                       '{}{}{}'.format('Porous_calcareous_siltstone', os.sep, '18_10X PP 1_bl.jpg'),
                       '{}{}{}'.format('Porous_calcareous_siltstone', os.sep, '21_10X PP 2_bl.jpg'),
                       ]

    # model folder
    model_dir = './runs/models/'
    ####################################################
    # choose models to be evaluated

    models_list = ['ResNet50_fine_tuned']

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

        # plot predictions
        for test_image in test_image_list:
            predicted = label_one('{}{}'.format(test_data_dir, test_image), m_path)
            make_fig(predicted, m_labels, '{}{}'.format(test_data_dir, test_image))

        # for confusion matrix:
        # check the cropped and the thin section csv files created
        for df_csv in [m, '{}{}'.format(m, '_thin_section_combined')]:
            print('Checking {}'.format(df_csv))
            df = pd.read_csv('{}/{}{}'.format(test_data_dir, df_csv, '.csv'))

            y_true = df['TrueLabel']
            y_pred = df['PredLabel']
            # Compute confusion matrix
            cnf_matrix = confusion_matrix(y_true, y_pred)

            # remove _ from m_labels
            m_labels = [word.replace('_', ' ') for word in m_labels]
            df_cm = pd.DataFrame(cnf_matrix, m_labels, m_labels)

            plt.figure(figsize=(15, 15))
            sn.set(font_scale=1.8)
            sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 14}, cmap="Greens")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig('{}/{}{}'.format(test_data_dir, df_csv, '_confusion_matrix.tif'),
                        dpi=600)
            plt.close()


    ############
    # create a figure to show multicrop
    fig_folder = '../Fig'
    figout_folder = '../Figout'
    dm.multi_crop(fig_folder, figout_folder, bottom_right=True)
