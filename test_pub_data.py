# Rafael Pires de Lima
# February 2019
# Test the fine-tuned model on public available (thin section) images

import os
import pickle
import shutil

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

import cnn_evaluate  # for label_folder
import data_manipulation as dm  # for color balance, multicrop

if __name__ == '__main__':

    print("Starting...")
    # model folder
    model_dir = './runs/models/'

    # for model selection parameters
    options_dict = {
        'ResNet50_fine_tuned': (224, 224, 3),
        'VGG19_fine_tuned': (224, 224, 3),
        'InceptionV3_fine_tuned': (299, 299, 3),
        'MobileNetV2_fine_tuned': (224, 224, 3),
    }
    models = ['ResNet50_fine_tuned', 
              'VGG19_fine_tuned',
              'InceptionV3_fine_tuned',
              'MobileNetV2_fine_tuned'
              ]

    # base folder:
    bs = '../Data/Public'
    # test data folder
    test_data_dir_in = '../Data/Public/Test_data'
    # save files with same dimensions as training model data into
    test_data_dir_size = '../Data/Public/Test_data_sz/size'
    # then multicrop images and save into
    test_mc = '../Data/Public/Test_mc'
    # then color balance and save into
    test_mc_wb = '../Data/Public/Test_mc_wb'

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
    # load the dataframe containing filename labels:
    df_labels = pd.read_csv(os.path.join(bs, 'public_filename_label.csv'))
    df_labels = df_labels.set_index('ts_name')
    
    class_names = ['Argillaceous_siltstone',
                  'Bioturbated_siltstone',
                  'Massive_calcareous_siltstone',
                  'Massive_calcite-cemented_siltstone',
                  'Porous_calcareous_siltstone']
    
    # create file to save accuracy and kappa not considering unknown values
    with open(os.path.join(bs, 'accuracy_kappa.csv'), 'w') as outfile:
        print('model, accuracy, kappa', file=outfile)
    
    # model path:
    for m in models:
        m_path = os.path.join(model_dir, f'{m}.hdf5')
        m_dict = os.path.join(model_dir, f'{m}_dict_l')
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
        df['PredLabel'] = df[class_names].idxmax(axis=1)

        # save the highest probability assigned:
        df['MaxPred'] = df[class_names].max(axis=1)

        # combine the results assigned to one thin section
        df['ts_name'] = df['filename'].str.rsplit('_', n=1, expand=True)[0]
        df_comb = df.groupby(by=['ts_name'])['PredLabel'].value_counts().unstack().fillna(0)

        # save the predicted label (the row argmax)
        df_comb['PredLabel_1'] = df_comb[class_names].idxmax(axis=1)

        # select only the predictions
        df_comb_pred = df_comb[class_names].copy()
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

        ####################################
        # combine predictions with labels provided 
        df_comb = df_comb.merge(df_labels, left_index=True, right_index=True)
        
        # save file:
        df_comb.to_csv(os.path.join(bs, f'{m}_combined.csv'), index=False)
    
        # compute the accuracy ignoring unknowns
        df_comb = df_comb[df_comb['TrueLabel'] != 'Unknown']
    
        y_true = df_comb['TrueLabel']
        y_pred = df_comb['PredLabel']
        
        with open(os.path.join(bs, 'accuracy_kappa.csv'), 'a') as outfile:
            acc = accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            print(f'{m}, {acc}, {kappa}', file=outfile)
