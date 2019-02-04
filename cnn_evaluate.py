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
from sklearn.metrics import accuracy_score

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

def multi_crop_class(df_input):
    """
    Function receives a pandas df containing predictions assigned to different files coming from the same
    thin section image.
    :param df_input: a pandas df with columns containing the probabilities given by a CNN model and the file name
    (last column)
    :return: df_out: a pandas df with the name of the original thin section image and the count of each one of the
    crops for each one of the classes
    """
    # create the df_out based on input columns:

    # split the values in file column
    # for this test, we know the real label:
    new_col = df_input['file'].str.split("\\", n=1, expand=True)
    df_input['filename'] = new_col[1]

    df_input['ts_name'] = [x[:-7] for x in df_input['filename']]

    # save thin section name in the output df
    df_out['ts_name'] = df_input['ts_name'].unique()



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
        'NASNetLarge': (331, 331, 3),

        'VGG19fine_tuned': (224, 224, 3),
        'ResNet50fine_tuned': (224, 224, 3),
        'InceptionV3fine_tuned': (299, 299, 3),
        'MobileNetV2fine_tuned': (224, 224, 3)

    }

    # train and validation data folders
    train_data_dir = '../Data/PP_mc_wb_train'
    validation_data_dir = '../Data/PP_mc_wb_validation'
    test_data_dir = '../Data/PP_mc_wb_test'

    test_image = '../Data/PP_mc_wb_test/Argilaceous_siltstone/23_10X PP 2_bl.jpg'

    # model folder
    model_dir = './runs/models/'
    ####################################################
    # choose models to be evaluated

    models_list = ['MobileNetV2', 'MobileNetV2fine_tuned',
                   'VGG19', 'VGG19fine_tuned',
                   'InceptionV3', 'InceptionV3fine_tuned',
                   'ResNet50', 'ResNet50fine_tuned']

    # we will change the threshold and save the accuracy results
    df_acc = pd.DataFrame(columns=['model',
                                   'acc_30',
                                   'acc_40',
                                   'acc_50',
                                   'acc_60',
                                   'acc_70',
                                   'acc_80',
                                   'acc_90',
                                   'acc_argmax'])

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
        # predicted = label_one(test_image, m_path)
        # make_fig(predicted, m_labels, test_image)

        # classify folder
        res = label_folder(test_data_dir, m_path)

        # save results as dataframe
        df = pd.DataFrame(res[0], columns=m_labels)
        df['file'] = res[1]

        # for this test, we know the real label:
        new_col = df['file'].str.split("\\", n=1, expand=True)
        df['TrueLabel'] = new_col[0]
        df['filename'] = new_col[1]

        # save the predicted label (the argmax)
        df['PredLabel'] = df[['Argillaceous_siltstone',
                              'Bioturbated_siltstone',
                              'Massive_calcareous_siltstone',
                              'Massive_calcite-cemented_siltstone',
                              'Porous_calcareous_siltstone']].idxmax(axis=1)

        # save the highest probability assigned:
        df['MaxPred'] = df[['Argillaceous_siltstone',
                            'Bioturbated_siltstone',
                            'Massive_calcareous_siltstone',
                            'Massive_calcite-cemented_siltstone',
                            'Porous_calcareous_siltstone']].max(axis=1)

        # save file
        df.to_csv('{}/{}{}'.format(test_data_dir, m, '.csv'))

        ##########
        # combine the results assigned to one thin section
        df['ts_name'] = [x[:-7] for x in df['filename']]
        df_comb = df.groupby(by=['ts_name', 'TrueLabel'])['PredLabel'].value_counts().unstack().fillna(0)

        # save the predicted label (the argmax)
        df_comb['TSPredLabel'] = df_comb[['Argillaceous_siltstone',
                              'Bioturbated_siltstone',
                              'Massive_calcareous_siltstone',
                              'Massive_calcite-cemented_siltstone',
                              'Porous_calcareous_siltstone']].idxmax(axis=1)

        # compute this accuracy:
        y_pred = df_comb['TSPredLabel']
        y_true = df_comb['TrueLabel']
        acc_combined = accuracy_score(y_true, y_pred)

        # save file
        df_comb.to_csv('{}/{}{}'.format(test_data_dir, m, '_thin_section_combined.csv'))
        #########

        # compute the accuracy using argmax:
        y_pred = df['PredLabel']
        y_true = df['TrueLabel']
        acc_argmax = accuracy_score(y_true, y_pred)

        # to populate the accuracy df later, save the name of the model
        acc_list = [m]
        # calculate the accuracy changing the threshold of acceptance:
        for i in range(30, 100, 10):
            # save a new df keeping only rows where the prediction is higher than the set threshold
            df_thresh = df[df.MaxPred >= i *0.01]
            print(len(df_thresh.index))
            # compute the accuracy with threshold data:
            y_pred = df_thresh['PredLabel']
            y_true = df_thresh['TrueLabel']
            acc = accuracy_score(y_true, y_pred)
            acc_list.append(acc)

        # append the argmax value computed previously (with complete data)
        acc_list.append(acc_argmax)
        # append the combined accuracy
        acc_list.append(acc_combined)

        # add new entry to df
        new_acc = pd.DataFrame([acc_list],
                               columns=['model',
                                        'acc_30',
                                        'acc_40',
                                        'acc_50',
                                        'acc_60',
                                        'acc_70',
                                        'acc_80',
                                        'acc_90',
                                        'acc_argmax',
                                        'acc_combined'])
        df_acc = pd.concat([df_acc, new_acc])

    df_acc.to_csv('{}/{}{}'.format(test_data_dir, 'accuracy_thresh', '.csv'))

    ####

    for m in models_list:
        print(m)

        # load the df
        df = pd.read_csv('{}/{}{}'.format(test_data_dir, m, '.csv'), index_col=0)

