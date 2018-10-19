import pandas as pd
import numpy as np

def weighted_averaging(pred_list, weights=[], submission_name=''):

    cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    pred_ens = pd.DataFrame(np.zeros(79726*10).reshape(79726,10), columns=cols)
    for w, i in enumerate(pred_list):
        a = pd.read_csv('submission/' + i)
        pred_ens[cols] += a[cols] * weights[w]

    pred_ens['img'] = a['img'].values
    pred_ens.to_csv('submission/' + submission_name + 'weightedaveraging.csv', index=False)

pred_lists = ['predictions_resnet.csv','predictions_vgg.csv']

weights = [0.9, 0.1]
weighted_averaging(pred_lists, weights, 'ensemble1_pred_')