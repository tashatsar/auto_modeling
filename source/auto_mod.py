import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy
import sklearn

from sklearn.utils import resample
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from lightgbm import LGBMRegressor, LGBMClassifier

import warnings
from pandas.core.common import SettingWithCopyWarning
from tqdm import tqdm_notebook

import auto_mod_metrics as metrics

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


DEFAULT_METRICS = {'regression': 'r2',
                   'binary': 'roc_auc',
                   'multiclass': 'f1_macro'}

SCORERS = sklearn.metrics.SCORERS
SCORERS['gini'] = metrics.gini
SCORERS['mape'] = metrics.mape
SCORERS['mae_mean'] = metrics.mea_mean


def identify_task_type(y, cat_cutoff=15):
    """
    Identification of ML task type depending on target values

    :param y: target values in Series format
    :param cat_cutoff: maximum number of values the target can take to be considered as categorical
    :returns: type of model, possible values are 'binary', 'regression' and 'multiclass'
    """
    y = pd.Series(y)
    if y.nunique() == 2:
        model_type = 'binary'
    elif y.nunique() > cat_cutoff:
        model_type = 'regression'
    else:
        model_type = 'multiclass'
    return model_type


class AutoModeling:
    def __init__(self, X_train, y_train, X_oos=None, y_oos=None, X_oot=None, y_oot=None,
                 model_type=None, metric=None, cat_cutoff_target=15, cat_cutoff=None,
                 random_state=13):

        self.X_train, self.cat_feats = self.detect_cat_feats(pd.DataFrame(X_train),
                                                             cat_cutoff=cat_cutoff) if cat_cutoff is not None else X_train, None
        self.y_train = y_train
        self.X_oos = pd.DataFrame(X_oos) if X_oos is not None else None
        self.y_oos = y_oos
        self.X_oot = pd.DataFrame(X_oot) if X_oot is not None else None
        self.y_oot = y_oot

        self.random_state = random_state

        self.metric = None
        self.model = None
        self.model_lr = None

        self.feats_selected = None
        self.final_feats_selected = None
        self.res_cv_best_set = None
        self.feats_imp_cv = None

        self.model_type = identify_task_type(self.y_train,
                                             cat_cutoff=cat_cutoff_target) if model_type is None else model_type
        self.set_metric(metric_name=metric)
        self.set_model()

    def set_train(self, X_train, y_train):
        self.X_train = pd.DataFrame(X_train)
        self.y_train = y_train

    def set_oos(self, X_oos, y_oos):
        self.X_oos = pd.DataFrame(X_oos)
        self.y_oos = y_oos

    def set_oot(self, X_oot, y_oot):
        self.X_oot = pd.DataFrame(X_oot)
        self.y_oot = y_oot

    @staticmethod
    def generate_data(n_features=10,
                      n_informative=5,
                      n_train=1000,
                      n_oos=200,
                      n_oot=300,
                      model_type='regression',
                      n_classes=5,
                      n_cat_features=0,
                      label_list_cat=None,
                      random_state=13,
                      bias=0,
                      noise=0):
        if model_type == 'binary':
            n_classes = 2

        if model_type == 'regression':
            X_train, y_train = make_regression(n_samples=n_train + n_oos + n_oot,
                                               n_features=n_features,
                                               n_informative=n_informative,
                                               n_targets=1,
                                               random_state=random_state,
                                               bias=bias,
                                               noise=noise)
        else:
            X_train, y_train = make_classification(n_samples=n_train + n_oos + n_oot,
                                                   n_features=n_features,
                                                   n_informative=n_informative,
                                                   n_classes=n_classes,
                                                   random_state=random_state,
                                                   shift=bias, flip_y=noise)

        feature_names = ['att_num_' + str(i) for i in range(n_features)]
        X_train = pd.DataFrame(data=X_train, columns=feature_names)
        for i in range(0, n_cat_features):
            X_train.iloc[:, i] = pd.cut(X_train.iloc[:, i], bins=len(label_list_cat), labels=label_list_cat)
        y_train = pd.Series(y_train)
        X_train, X_oot, y_train, y_oot = train_test_split(X_train, y_train,
                                                          train_size=n_train + n_oos,
                                                          random_state=random_state)
        X_train, X_oos, y_train, y_oos = train_test_split(X_train, y_train,
                                                          train_size=n_train,
                                                          random_state=random_state)
        return X_train, y_train, X_oos, y_oos, X_oot, y_oot

    @staticmethod
    def detect_cat_feats(X, cat_cutoff=20):
        """
        Detection of categorical features with further conversion to 'category' type

        :param X: input features in DataFrame format
        :param cat_cutoff: maximum number of values a variable can take to be considered as categorical
        :returns: DataFrame with columns converted to 'category' type
        """
        for feat in X.select_dtypes('integer').columns:
            if X[feat].nunique() <= cat_cutoff:
                X[feat] = X[feat].astype('category')  # is a number with few number of values
        for feat in X.select_dtypes(exclude='number').columns:
            X[feat] = X[feat].astype('category')  # is a letter
        cat_feats = X.select_dtypes('category').columns
        return X, cat_feats

    @staticmethod
    def encode_cat_feats(X_train, X_oos=None, X_oot=None):
        """
        Encoding of categorical features (of 'category' type) to a dummy variables.
        NaN values have a separated category.

        :param X_train: input features in DataFrame format
        :param X_oos: out-of-sample input features in DataFrame format
        :param X_oot: out-of-time input features in DataFrame format
        :returns: tuple with three DataFrames with encoded X, X_oos and X_oot samples
        """
        X = X_train.copy(deep=True)
        X['sample'] = 'train'
        if X_oos is not None:
            X_oos['sample'] = 'oos'
            X_sample = pd.concat([X, X_oos])
            X_oos.drop('sample', axis=1, inplace=True)
        else:
            X_sample = X.copy(deep=True)
        X.drop('sample', axis=1, inplace=True)

        if X_oot is not None:
            X_oot['sample'] = 'oot'
            X_sample = pd.concat([X_sample, X_oot])
            X_oot.drop('sample', axis=1, inplace=True)

        cols_list = X_sample.select_dtypes(include='category').columns

        for feat in cols_list:
            dummy = pd.get_dummies(X_sample[feat], prefix=feat, dummy_na=True, drop_first=True)
            X_sample = pd.concat([X_sample, dummy], axis=1)
            X_sample.drop(feat, axis=1, inplace=True)
        res = ()
        for sample in ('train', 'oos', 'oot'):
            df = X_sample[X_sample['sample'] == sample].drop('sample', axis=1)
            res += (df,) if len(df) > 0 else (None,)
        return res

    def set_metric(self, metric_name=None):
        """Identification of metric"""
        self.metric = DEFAULT_METRICS.get(self.model_type, 'roc_auc') if metric_name is None else metric_name

    def set_model(self):
        """Identification of model class"""
        kwargs = {'random_state': self.random_state}
        if self.model_type in ['binary', 'multiclass']:
            kwargs['objective'] = self.model_type
        model_class = LGBMRegressor if self.model_type == 'regression' else LGBMClassifier
        self.model = model_class(**kwargs)
        self.model_lr = LinearRegression() if self.model_type == 'regression' else LogisticRegression()

    def feature_imp_fast(self, verbose=True, plt_show=True):
        """
        Fast calculation of feature importance based on LightGBM model

        :return: DataFrame with feature importance of every feature and list of the most important features
        """
        if verbose:
            print('\nLGBM feature importance calculation is in process...')
        model = self.model
        model.fit(self.X_train, self.y_train)
        feats_imp = pd.DataFrame(zip(model.feature_importances_, self.X_train.columns),
                                 columns=['score', 'feature']).sort_values(by='score', ascending=False)
        c_v = np.std(feats_imp['score']) / np.mean(feats_imp['score'])  # coefficient of variation
        part_pred = 136.82 * np.exp(c_v * (-1.01))  # predicted zbs percent of informative features
        part_pred = np.clip(part_pred, 1, 100)
        top = int(len(feats_imp) * part_pred / 100)
        feats_selected = feats_imp['feature'].head(top)
        if verbose:
            print('Fast feature selection is finished! {} features out of {} are selected.'.format(len(feats_selected),
                                                                                                   len(feats_imp)))

        if plt_show:
            palette = sns.cubehelix_palette(len(feats_selected), start=2, reverse=True, rot=0, dark=0.5, light=0.95)
            plt.figure(figsize=(14, 8))
            sns.barplot(x='score', y='feature', data=feats_imp.head(top), palette=palette, saturation=1)
            plt.title('LGBM feature importances for {} selected features'.format(len(feats_selected)), fontsize=18)
            plt.xlabel('Number of features in the model', fontsize=16)
            plt.ylabel(self.metric + ' score', fontsize=16)
            plt.show()
        return feats_imp, feats_selected

    def feature_imp_stepwise(self, reduce_sample_times=10, cv_folds=3, round_derivative_digits=2,
                             feats_selected=None, verbose=True):
        """
        Calculation of feature importance based on forward selection using LightGBM model
        and feature selection according to gain of model performance metric

        :param reduce_sample_times: times to reduce X_test for faster execution
        :param cv_folds: number of cross validation folds for average CV metric calculation
        :param round_derivative_digits: number of digits of derivative round for the check of equality of derivative
        to zero. Higher the parameter -- softer the criteria and more features are potentially selected. The opposite is
        also correct. NOTE: with big values all of features can be selected.
        :param feats_selected: list of features to check
        :param verbose: print performance metric and number of features selected at each step
        :return: DataFrame with number of included in model features and metric of model performance
        """
        print('\nStepwise forward feature selection is in process...')
        feats = list(feats_selected) if feats_selected is not None else self.X_train.columns
        len_feats = len(feats)
        iter_feats = []
        derivative_mean = 1
        feats_imp_cv = pd.DataFrame(columns=['n_f', 'feature', 'score'])

        # keep checking features until the mean derivative of last three values of performance metric
        # does not become significantly close to zero or lower then zero
        while round(derivative_mean, round_derivative_digits) > 0:
            result = pd.DataFrame(columns=['feature', 'score'])
            # score on cross validation with different number of features
            sign = sklearn.metrics.SCORERS.get(self.metric)._sign
            for f in feats:
                score_cv = cross_val_score(self.model,
                                           self.X_train[iter_feats + [f]][::reduce_sample_times],
                                           self.y_train[::reduce_sample_times],
                                           cv=cv_folds,
                                           scoring=self.metric).mean() * sign
                result.loc[len(result)] = f, score_cv

            # adding the best feature according to cross validation metric to the list of features
            best_ind = result['score'].idxmax()  # if sign == 1 else result['score'].idxmin()
            best_feat = result.loc[best_ind]['feature']

            iter_feats.append(best_feat)

            score = float(result[result['feature'] == best_feat]['score'])
            feats_imp_cv.loc[len(feats_imp_cv)] = len(iter_feats), best_feat, score
            feats.remove(best_feat)

            if verbose:
                ending = 's' if len(feats_imp_cv) > 1 else ''
                verb = 'are' if len(feats_imp_cv) > 1 else 'is'
                print('{} feature{} out of {} {} selected, {} = {}'.format(len(feats_imp_cv),
                                                                           ending, len_feats, verb,
                                                                           self.metric, round(score, 3)))
            # checking the derivative as stopping criteria
            if len(iter_feats) >= 3:
                derivative_mean = np.mean(np.gradient(feats_imp_cv['score'][-3:]))
        self.feats_imp_cv = pd.DataFrame(feats_imp_cv)
        print(
            'Forward feature selection is finished! {} features out of {} are selected.'.format(len(self.feats_imp_cv),
                                                                                                len_feats))
        return feats_imp_cv

    def metric_gain_plot(self, cv_folds=5, plt_show=True):
        """
        :param cv_folds: number of cross validation folds for average CV metric calculation
        :param plt_show: flag of plot output
        :return: DataFrame with performance metrics of the model
        """
        res_cv_best_set = pd.DataFrame(columns=['n_f', 'score_cv', 'score_oos', 'score_oot'])
        for n_f in range(0, len(self.feats_imp_cv) + 1):
            f = list(self.feats_imp_cv.head(n_f)['feature'])
            score_cv = cross_val_score(self.model,
                                       self.X_train[f],
                                       self.y_train,
                                       cv=cv_folds,
                                       scoring=self.metric).mean()
            self.model.fit(self.X_train[f], self.y_train)
            score = sklearn.metrics.SCORERS.get(self.metric)
            score_oos = (None if self.X_oos is None else score(self.model, self.X_oos[f], self.y_oos))
            score_oot = (None if self.X_oot is None else score(self.model, self.X_oot[f], self.y_oot))

            res_cv_best_set.loc[len(res_cv_best_set)] = n_f, score_cv, score_oos, score_oot
        if plt_show:
            plt.figure(figsize=(14, 6))
            plt.plot(res_cv_best_set['score_cv'], label='score_cv')
            plt.plot(res_cv_best_set['score_oos'], label='score_oos')
            plt.plot(res_cv_best_set['score_oot'], label='score_oot')
            plt.legend()
            plt.title('Performance of LGBM model varying the number of features selected', fontsize=18)
            plt.xlabel('Number of features in the model', fontsize=16)
            plt.ylabel(self.metric + ' score', fontsize=16)
            plt.show()
        return res_cv_best_set

    def lgbm_mod(self, n_iter=10, feats=None, params=None):
        """
        Model based on LGBM model for regression and classification

        :param n_iter: number of parameter settings that are sampled
        :param feats: list of features to check
        :param params: hyperparameters to tune in a dictionary format
        :return: DataFrame with performance metrics of the best choosen model and best model
        """
        random_state = 13
        feats = list(feats) if feats is not None else self.X_train.columns

        print('\nHyperparameters tuning for LGBM model...')
        if params is None:
            params = {'num_leaves': np.linspace(10, 200, 8, dtype=int),
                      'reg_alpha': np.linspace(0, 0.99, 20),
                      'max_depth': np.linspace(2, 63, 15, dtype=int),
                      'learning_rate': np.logspace(-4, 1, 20),
                      'reg_lambda': np.linspace(0, 0.99, 20),
                      # 'n_estimators': np.linspace(20, 300, 25, dtype=int),
                      'min_child_samples': np.linspace(5, 50, 10, dtype=int)
                      }
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        model = RandomizedSearchCV(self.model, param_distributions=params,
                                   scoring=self.metric, cv=kf, random_state=random_state,
                                   n_iter=n_iter, n_jobs=-1, verbose=10, error_score=0)

        model.fit(self.X_train[feats], self.y_train)
        best_model = model.best_estimator_

        print('LGBM model best_params:', model.best_params_)

        score = sklearn.metrics.SCORERS.get(self.metric)
        score_oos = ('OOS is not presented' if self.X_oos is None else score(model, self.X_oos[feats], self.y_oos))
        score_oot = ('OOT is not presented' if self.X_oot is None else score(model, self.X_oot[feats], self.y_oot))

        alt_mod_result = pd.DataFrame({'num_feats': len(feats),
                                       'score_cv': model.best_score_,
                                       'score_train': score(model, self.X_train[feats], self.y_train),
                                       'score_oos': score_oos,
                                       'score_oot': score_oot
                                       }, index=['LGBM'])
        return alt_mod_result, best_model

    def lr_alt_mod(self, feats=None, cv_folds=3):
        """
        Model based on linear or logistic regression model

        :param feats: list of features to check
        :param cv_folds: number of cross validation folds for average CV metric calculation
        :return: DataFrame with performance metrics of the model
        """
        feats = list(feats) if feats is not None else self.X_train.columns
        model, scoring = self.model_lr, self.metric

        X_lr, X_lr_oos, X_lr_oot = self.encode_cat_feats(self.X_train[feats],
                                                         None if self.X_oos is None else self.X_oos[feats],
                                                         None if self.X_oot is None else self.X_oot[feats])

        X_lr = X_lr.fillna(0)
        X_lr_oos = None if self.X_oos is None else X_lr_oos.fillna(0)
        X_lr_oot = None if self.X_oot is None else X_lr_oot.fillna(0)

        model.fit(np.array(X_lr), self.y_train.values.ravel())

        score = sklearn.metrics.SCORERS.get(scoring)
        score_oos = ('OOS is not presented' if self.X_oos is None else score(model, np.array(X_lr_oos),
                                                                             self.y_oos.values.ravel()))
        score_oot = ('OOT is not presented' if self.X_oot is None else score(model, np.array(X_lr_oot),
                                                                             self.y_oot.values.ravel()))
        score_train = score(model, np.array(X_lr), self.y_train.values.ravel())
        score_cv = cross_val_score(model,
                                   np.array(X_lr),
                                   self.y_train.values.ravel(),
                                   cv=cv_folds,
                                   scoring=scoring).mean()
        alt_mod_result = pd.DataFrame({'num_feats': len(feats),
                                       'score_cv': score_cv,
                                       'score_train': score_train,
                                       'score_oos': score_oos,
                                       'score_oot': score_oot
                                       }, index=['LR'])
        return alt_mod_result

    def modeling(self,
                 n_iter_lgbm=50,
                 params_lgbm=None,
                 feats_selection=True,
                 reduce_sample_times_stepwise=2,
                 round_derivative_digits=2,
                 metric_gain_calc=True,
                 n_feats_init=None,
                 recalc=True,
                 plt_show=True,
                 path_save=None):
        """
        Modeling function

        :param n_iter_lgbm: number of parameter settings that are sampled
        :param params_lgbm: parameters for LGBM model
        :param feats_selection: necessity of feature selection
        :param reduce_sample_times_stepwise: times to reduce test sample for faster execution
        :param metric_gain_calc: flag of calculation metric gain
        :param n_feats_init: number of features in the initial model
        :param recalc: whether or not to recalculate statistics
        :param plt_show: whether show plots or not
        :return: selected features, metrics of model performance
        """
        if recalc:
            self.feats_selected = None
            self.final_feats_selected = None
            self.res_cv_best_set = None
            self.feats_imp_cv = None

        print('Solving {} problem, key performance metric is {}.'.format(self.model_type, self.metric))

        # categorical features according to analysis of train data set
        for sample in (self.X_oos, self.X_oot):
            if sample is not None and self.cat_feats is not None:
                for feat in self.cat_feats:
                    sample[feat] = sample[feat].astype('category')

        if feats_selection:
            if self.feats_selected is None:
                _, self.feats_selected = self.feature_imp_fast(plt_show=plt_show)

            if self.final_feats_selected is None:
                self.feats_imp_cv = self.feature_imp_stepwise(feats_selected=self.feats_selected,
                                                              round_derivative_digits=round_derivative_digits,
                                                              reduce_sample_times=reduce_sample_times_stepwise)
                self.final_feats_selected = self.feats_imp_cv.feature
                if metric_gain_calc:
                    self.res_cv_best_set = self.metric_gain_plot(plt_show=plt_show)
                else:
                    self.res_cv_best_set = None
        else:
            self.feats_imp_cv, _ = self.feature_imp_fast(verbose=False, plt_show=plt_show)
            self.final_feats_selected = self.X_train.columns
            self.feats_selected, self.res_cv_best_set = None, None

        if path_save is not None:
            {'feats_selected': self.feats_selected,
             'feats_imp_cv': self.feats_imp_cv,
             'res_cv_best_set': self.res_cv_best_set}.to_csv(path_save + 'features_info.csv')

        alt_mod_lgbm, best_model_lgbm = self.lgbm_mod(n_iter=n_iter_lgbm,
                                                      feats=self.final_feats_selected,
                                                      params=params_lgbm)
        try:
            alt_mod_lr = self.lr_alt_mod(feats=self.final_feats_selected)
        except ValueError:
            alt_mod_lr = None
            print('\nLinear model was not fitted due to ValueError')

        if n_feats_init is not None and n_feats_init - 2 < len(self.final_feats_selected):
            final_feats_selected_n_f = list(self.final_feats_selected.head(n_feats_init - 2))

            alt_mod_lgbm_n_f, best_model_lgbm_n_f = self.lgbm_mod(n_iter=n_iter_lgbm,
                                                                  feats=final_feats_selected_n_f,
                                                                  params=params_lgbm)
            alt_mod_lr_n_f = self.lr_alt_mod(feats=final_feats_selected_n_f)
            alt_mod_result = pd.concat([alt_mod_lgbm, alt_mod_lr, alt_mod_lgbm_n_f, alt_mod_lr_n_f])
        else:
            alt_mod_result = pd.concat([alt_mod_lgbm, alt_mod_lr])

        result = {'feats_selected': self.feats_selected,
                  'feats_imp_cv': self.feats_imp_cv,
                  'mod_result': alt_mod_result,
                  'res_cv_best_set': self.res_cv_best_set,
                  'best_model_lgbm': best_model_lgbm}

        if path_save is not None:
            result.to_csv(path_save + 'modeling_results.csv')

        return result


class ModelComparison:
    def __init__(self,
                 X_vld, y_vld, vld_model,
                 X_alt, y_alt, alt_model,
                 metric=None,
                 model_type=None,
                 cat_cutoff_target=15):

        self.X_vld = X_vld
        self.y_vld = y_vld
        self.vld_model = vld_model

        self.X_alt = X_alt
        self.y_alt = y_alt
        self.alt_model = alt_model

        self.model_type = identify_task_type(self.y_alt,
                                             cat_cutoff=cat_cutoff_target) if model_type is None else model_type
        self.set_metric(metric=metric)
        self.score = sklearn.metrics.SCORERS.get(self.metric)

        self.alt_metrics = None
        self.vld_metrics = None
        self.n_iter = None

    def set_metric(self, metric=None):
        """Identification of metric"""
        self.metric = DEFAULT_METRICS.get(self.model_type, 'roc_auc') if metric is None else metric

    def stat_calc(self, X, y, model, n_iter=200, test_size=0.3, bootstrap=False):
        """Calculation of statistics using either resampling or bootstrap"""
        stats = []
        sign = sklearn.metrics.SCORERS.get(self.metric)._sign
        if bootstrap:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=13)
            model.fit(X_train, y_train)
            for i in tqdm_notebook(range(n_iter)):
                X_test_bs = resample(X_test, n_samples=int(len(X) * test_size), random_state=i)
                y_test_bs = y_test.loc[X_test_bs.index]
                stats.append(self.score(model, X_test_bs, y_test_bs) * sign)
        else:
            for i in tqdm_notebook(range(n_iter)):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
                model.fit(X_train, y_train)
                stats.append(self.score(model, X_test, y_test) * sign)
        return np.array(stats)

    @staticmethod
    def quant(dist, q):
        return sorted(dist)[int(q * len(dist))]

    def hist_plots(self, vld_metrics, alt_metrics, bins=10, quantile=0.95):
        """Plotting distribution of metrics for two models"""
        plt.figure(figsize=(16, 10))
        plt.hist(vld_metrics, bins=bins, alpha=0.4, label='Initial model')
        plt.hist(alt_metrics, bins=bins, alpha=0.4, facecolor='green', label='Alternative model')
        _, top = plt.ylim()
        plt_quantile_height = top * 0.1

        self.left_q = 1 / 2 - quantile / 2
        self.right_q = 1 / 2 + quantile / 2

        self.left_quantile_vld = self.quant(vld_metrics, self.left_q)
        self.right_quantile_vld = self.quant(vld_metrics, self.right_q)
        plt.plot([self.left_quantile_vld, self.left_quantile_vld, self.right_quantile_vld, self.right_quantile_vld],
                 [0, plt_quantile_height, plt_quantile_height, 0],
                 label='Initial %0.2f confidence interval' % quantile, color='blue', linestyle='dashed')
        plt.axvline(vld_metrics.mean(), 0, top, label='Initial model mean metric')

        self.left_quantile_alt = self.quant(alt_metrics, self.left_q)
        self.right_quantile_alt = self.quant(alt_metrics, self.right_q)
        plt.plot([self.left_quantile_alt, self.left_quantile_alt, self.right_quantile_alt, self.right_quantile_alt],
                 [0, plt_quantile_height, plt_quantile_height, 0],
                 label='Alternative %0.2f confidence interval' % quantile, color='green', linestyle='dashed')
        plt.axvline(alt_metrics.mean(), 0, top, label='Alternative model mean metric', color='green')

        plt.xlabel(self.metric, fontsize=16)
        plt.ylabel('% of evidence', fontsize=16)
        plt.title('Distribution of {} score'.format(self.metric), fontsize=18)
        plt.legend()
        plt.show()

    def stats_output(self, vld_metrics, alt_metrics):
        """Calculation of means, p-value and confidence intervals"""
        # p_value = round(scipy.stats.ttest_rel(alt_metrics, vld_metrics)[1], 5)
        # if p_value < 0.0001:
        #     sign_level = 'любом разумном'
        # elif p_value < 0.01:
        #     sign_level = '1%-ом'
        # elif p_value < 0.05:
        #     sign_level = '5%-ом'

        # if p_value >= 0.05:
        #     summary = 'Гипотеза об отсутствии различий в распределениях не отвергается (p_value={}).'.format(p_value)
        # else:
        #     summary = 'Статистический тест показывает значимое различие в распределениях на {} уровне значимости ' \
        #               '(p_value = {}). '.format(sign_level, p_value)
        summary = '\nMean value of {} metric for initial model is {},' \
                  ' and {} for the alternative one.'.format(self.metric,
                                                            round(vld_metrics.mean(), 4),
                                                            round(alt_metrics.mean(), 4))
        print(summary)
        stats_diff_abs = 100 * (-vld_metrics + alt_metrics)
        stats_diff_rel = 100 * (-vld_metrics + alt_metrics) / vld_metrics

        left_quantile_abs = self.quant(stats_diff_abs, self.left_q)
        right_quantile_abs = self.quant(stats_diff_abs, self.right_q)
        left_quantile_rel = self.quant(stats_diff_rel, self.left_q)
        right_quantile_rel = self.quant(stats_diff_rel, self.right_q)

        res = {"mean": [alt_metrics.mean(), vld_metrics.mean(), stats_diff_abs.mean(), stats_diff_rel.mean()],
               "std": [alt_metrics.std(), vld_metrics.std(), stats_diff_abs.std(), stats_diff_rel.std()],
               "left bound": [self.left_quantile_alt, self.left_quantile_vld, left_quantile_abs, left_quantile_rel],
               "right bound": [self.right_quantile_alt, self.right_quantile_vld, right_quantile_abs,
                               right_quantile_rel]}

        res = pd.DataFrame(res, index=["Alternative model", "Initial model", "Absolute difference (p.p.)",
                                       "Relative difference (%%)"])
        self.res = res
        self.summary = summary

    def compare_models(self, n_iter=200, test_size=0.3, hist_bins=10, recalc=False, bootstrap=False):
        """
        Method for comparison of two models: calculation of statistics, means and p-value, plotting a histogram
        :param n_iter: number of iterations for resampling or bootstrap
        :param test_size: part of sample to be used as test in train_test_split from sklearn
        :param hist_bins: number of bins for histogram
        :param recalc: whether or not to recalculate statistics
        :param bootstrap: use bootstrapping instead of resampling
        :returns: DataFrame with means and p-value
        """
        if self.n_iter != n_iter:
            self.n_iter = n_iter
            recalc = True

        if recalc:
            self.alt_metrics = None
            self.vld_metrics = None

        if self.alt_metrics is None:
            print('Calculating statistics for alternative model...')
            self.alt_metrics = self.stat_calc(self.X_alt, self.y_alt, self.alt_model,
                                              n_iter=self.n_iter, test_size=test_size, bootstrap=bootstrap)
        else:
            print('Statistics for alternative model have been already calculated!')

        if self.vld_metrics is None:
            print('Calculating statistics for initial model...')
            self.vld_metrics = self.stat_calc(self.X_vld, self.y_vld, self.vld_model,
                                              n_iter=self.n_iter, test_size=test_size, bootstrap=bootstrap)
        else:
            print('Statistics for initial model have been already calculated!')

        self.hist_plots(self.vld_metrics, self.alt_metrics, bins=hist_bins)

        self.stats_output(self.vld_metrics, self.alt_metrics)

        return self.res, self.summary
