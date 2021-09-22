import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, make_scorer


def gini_score(y_true, y_pred):
    return -1+2*roc_auc_score(y_true, y_pred)


def mape_score(y_true, y_pred):
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except ZeroDivisionError:
        mape = None
        print('\nImpossible to calculate MAPE due to zero values in the input data!')
    return mape


def mae_mean_score(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)/np.abs(y_true.mean()) * 100


gini = make_scorer(gini_score, needs_threshold=True)
mape = make_scorer(mape_score, greater_is_better=False)
mea_mean = make_scorer(mae_mean_score, greater_is_better=False)
