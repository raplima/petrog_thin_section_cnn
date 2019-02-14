# Rafael Pires de Lima
# February 2019
# Test the fine-tuned model on public available (thin section) images

import os
import pickle
import shutil

import pandas as pd
import seaborn as sn
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import cnn_evaluate  # for label_folder
import data_manipulation as dm  # for color balance, multicrop

if __name__ == '__main__':

    print("Starting...")
    # model folder
    model_dir = './runs/models/'

    # for model selection parameters
    options_dict = {
        'ResNet50_fine_tuned': (224, 224, 3)
    }
    m = 'ResNet50_fine_tuned'

    # base folder:
    bs = '../Public'
    # test data folder
    test_data_dir_in = '../Public/Test_data'
    # save files with same dimensions as training model data into
    test_data_dir_size = '../Public/Test_data_sz/size'
    # then multicrop images and save into
    test_mc = '../Public/Test_mc'
    # then color balance and save into
    test_mc_wb = '../Public/Test_mc_wb'

    # make sure all these folders exist:
    for path_out in [test_data_dir_size, test_mc, test_mc_wb]:
        if os.path.exists(path_out):
            shutil.rmtree(path_out, ignore_errors=True)
        os.makedirs(path_out)

    # get all image files from initial directory
    images = os.listdir(test_data_dir_in)

    # loop through all images to save them with 1292 x 968 dimensions (to match training data parameters)
    for img in images:
        ori = Image.open(test_data_dir_in + os.sep + img)
        rs = ori.resize((1292, 968), Image.ANTIALIAS)
        rs.save(test_data_dir_size + os.sep + img)

    print('Image resize complete')

    # crop the images
    dm.multi_crop(os.path.dirname(test_data_dir_size), test_mc, bottom_right=True, random_crop=3)

    # apply white balance
    dm.wbalance(test_mc, test_mc_wb, ['size'])

    # this data can be used for testing with the CNN model.
    ###################################################################
    # model path:
    m_path = '{}{}{}'.format(model_dir, m, '.hdf5')
    m_dict = '{}{}{}'.format(model_dir, m, '_dict_l')
    # open the label dictionary
    with open(m_dict, 'rb') as f:
        m_labels = pickle.load(f)

    # image height and width ("picture size for the model"):
    height = options_dict[m][0]
    width = options_dict[m][1]

    # classify folder
    res = cnn_evaluate.label_folder(test_mc_wb, m_path)

    # save results as dataframe
    df = pd.DataFrame(res[0], columns=m_labels)
    df['file'] = res[1]

    # save the filename
    new_col = df['file'].str.split("\\", n=1, expand=True)
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
    df.to_csv('{}/{}{}'.format(bs, m, '.csv'))

    ##########
    # combine the results assigned to one thin section
    df['ts_name'] = [x[:-7] for x in df['filename']]
    df_comb = df.groupby(by=['ts_name'])['PredLabel'].value_counts().unstack().fillna(0)

    # save and load
    df_comb.to_csv('{}/{}{}'.format(bs, m, '_temp.csv'))
    df_comb = pd.read_csv('{}/{}{}'.format(bs, m, '_temp.csv'))
    os.remove('{}/{}{}'.format(bs, m, '_temp.csv'))

    # save the predicted label (the row argmax)
    df_comb['PredLabel_1'] = df_comb.iloc[0:, 2:].idxmax(axis=1)

    # select only the predictions
    df_comb_pred = df_comb.iloc[0:, 1:6]
    # sort the results:
    pred_sort = df_comb_pred.values.argsort(1)
    # save the label with the "first" argmax
    df_comb['PredLabel_1'] = df_comb_pred.columns[pred_sort[:, -1]]
    # save the label with the "second" argmax
    df_comb['PredLabel_2'] = df_comb_pred.columns[pred_sort[:, -2]]

    # the predicted label is the PredLabel_1...
    df_comb['PredLabel'] = df_comb['PredLabel_1']
    # ... as long as there is no tie between first and second "argmax"

    # maybe inefficient, but just check whether argmax1 is actually bigger than argmax2:
    for i in range(0, len(df_comb)):
        if df_comb_pred.iloc[i][pred_sort[i, -1]] == df_comb_pred.iloc[i][pred_sort[i, -2]]:
            print('tie {}'.format(i))
            df_comb['PredLabel'][i] = 'Tie'

    # save file
    df_comb.to_csv('{}/{}{}'.format(bs, m, '_thin_section_combined.csv'))

    ####################################
    ####################################

    # later on, geologist provided the labels for the public data
    # generate a confusion matrix:
    df_csv = 'ResNet50_fine_tuned_thin_section_combined_pred_true'
    print('Checking {}'.format(df_csv))
    df = pd.read_csv('{}/{}{}'.format(bs, df_csv, '.csv'))

    y_true = df['TrueLabel']
    y_pred = df['PredLabel']
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)

    # set the labels
    yt_l = ['Bioturbated siltstone', 'Massive calcareous siltstone', 'Massive calcite-cemented siltstone',
            'Porous calcareous siltstone', 'Tied', 'Unknown']
    df_cm = pd.DataFrame(cnf_matrix, yt_l, yt_l)

    plt.figure(figsize=(15, 15))
    sn.set(font_scale=1.8)
    sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 14}, cmap="Greens")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('{}/{}{}'.format(bs, df_csv, '_confusion_matrix.tif'),
                dpi=600)
    plt.close()

    # compute the accuracy ignoring ties and unknowns
    df = df[df['TrueLabel'] != 'Unknown']
    df = df[df['PredLabel'] != 'Tie']

    y_true = df['TrueLabel']
    y_pred = df['PredLabel']

    print(accuracy_score(y_true, y_pred))
