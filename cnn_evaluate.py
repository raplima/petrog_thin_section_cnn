# Rafael Pires de Lima
# February 2019
# Evaluation of CNN models generated with cnn_processing.py
# This file includes functions to:
# predict the resulting classification of a folder with subfolders using the CNN model provided and generate a csv file
# analyse the predictions

import os
import pickle

import matplotlib.image as mpimg
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.metrics import accuracy_score

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
        'NASNetLarge': (331, 331, 3),

        'VGG19_fine_tuned': (224, 224, 3),
        'ResNet50_fine_tuned': (224, 224, 3),
        'InceptionV3_fine_tuned': (299, 299, 3),
        'MobileNetV2_fine_tuned': (224, 224, 3)

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

    models_list = ['MobileNetV2', 'MobileNetV2_fine_tuned',
                   'VGG19', 'VGG19_fine_tuned',
                   'InceptionV3', 'InceptionV3_fine_tuned',
                   'ResNet50', 'ResNet50_fine_tuned']

    # we will change the threshold and save the accuracy results
    df_acc = pd.DataFrame(columns=['model',
                                   'acc_30', 'unks_30', 'acc_35', 'unks_35',
                                   'acc_40', 'unks_40', 'acc_45', 'unks_45',
                                   'acc_50', 'unks_50', 'acc_55', 'unks_55',
                                   'acc_60', 'unks_60', 'acc_65', 'unks_65',
                                   'acc_70', 'unks_70', 'acc_75', 'unks_75',
                                   'acc_80', 'unks_80', 'acc_85', 'unks_85',
                                   'acc_90', 'unks_90', 'acc_95', 'unks_95',
                                   'acc_argmax', 'n_tot',
                                   'acc_combined'])

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

        # save and load
        df_comb.to_csv('{}/{}{}'.format(test_data_dir, m, '_temp.csv'))
        df_comb = pd.read_csv('{}/{}{}'.format(test_data_dir, m, '_temp.csv'))
        os.remove('{}/{}{}'.format(test_data_dir, m, '_temp.csv'))

        # save the predicted label (the argmax)
        df_comb['PredLabel'] = df_comb.iloc[0:, 2:].idxmax(axis=1)

        # compute this accuracy:
        y_pred = df_comb['PredLabel']
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
        for i in range(30, 100, 5):
            # save a new df keeping only rows where the prediction is higher than the set threshold
            df_thresh = df[df.MaxPred >= i * 0.01]
            # print(len(df_thresh.index))
            # compute the accuracy with threshold data:
            y_pred = df_thresh['PredLabel']
            y_true = df_thresh['TrueLabel']
            acc = accuracy_score(y_true, y_pred)
            acc_list.append(acc)
            acc_list.append(len(df.index) - len(df_thresh.index))

        # append the argmax value computed previously (with complete data)
        acc_list.append(acc_argmax)
        # append the number of samples analyzed
        acc_list.append(len(df.index))
        # append the combined accuracy
        acc_list.append(acc_combined)

        # add new entry to df
        new_acc = pd.DataFrame([acc_list],
                               columns=['model',
                                        'acc_30', 'unks_30', 'acc_35', 'unks_35',
                                        'acc_40', 'unks_40', 'acc_45', 'unks_45',
                                        'acc_50', 'unks_50', 'acc_55', 'unks_55',
                                        'acc_60', 'unks_60', 'acc_65', 'unks_65',
                                        'acc_70', 'unks_70', 'acc_75', 'unks_75',
                                        'acc_80', 'unks_80', 'acc_85', 'unks_85',
                                        'acc_90', 'unks_90', 'acc_95', 'unks_95',
                                        'acc_argmax', 'n_tot',
                                        'acc_combined'])
        # print(acc_list)
        df_acc = pd.concat([df_acc, new_acc])

    df_acc.to_csv('{}/{}{}'.format(test_data_dir, 'accuracy_thresh', '.csv'))
    ####
