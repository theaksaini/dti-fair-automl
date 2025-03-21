import openml
import tpot
from tpot.search_spaces.pipelines import ChoicePipeline, SequentialPipeline
from tpot.search_spaces.nodes import *
import sklearn.metrics
import sklearn
import traceback
import dill as pickle
import os
import time
import numpy as np
import sklearn.model_selection
from functools import partial
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder

def encode_categorical_features(X_train, X_test):
    boolean_cols = X_train.select_dtypes(include=['bool']).columns
    X_train[boolean_cols] = X_train[boolean_cols].astype(int)
    X_test[boolean_cols] = X_test[boolean_cols].astype(int)

    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    # Initialize and fit the encoder on X_train
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    
    # Transform X_test using the same encoder
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])
    
    return X_train, X_test


def check_xgboost_compatibility(X, y=None):
    """
    Checks if the dataset is ready for XGBoost in Sklearn.
    
    Parameters:
    - X: Feature matrix (Pandas DataFrame or NumPy array)
    - y: Target variable (optional, Pandas Series or NumPy array)
    
    Returns:
    - Boolean indicating if the dataset is compatible
    - List of issues (if any)
    """
    issues = []
    
    # Check if X is a Pandas DataFrame or NumPy array
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        issues.append("X must be a Pandas DataFrame or NumPy array.")
    
    # Check for missing values
    if isinstance(X, pd.DataFrame):
        if X.isnull().any().any():
            issues.append("X contains missing values. Consider imputing them.")
    elif isinstance(X, np.ndarray):
        if np.isnan(X).any():
            issues.append("X contains missing values. Consider imputing them.")

    # Check for infinite values
    if np.isinf(X).values.sum() > 0:
        issues.append("X contains infinite values. Replace them with finite values.")

    # Check if all features are numeric
    if isinstance(X, pd.DataFrame):
        if not all(X.dtypes.apply(lambda dtype: np.issubdtype(dtype, np.number))):
            issues.append("X contains non-numeric features. Encode categorical features before using XGBoost.")

    # Check target variable (if provided)
    if y is not None:
        if not isinstance(y, (pd.Series, np.ndarray)):
            issues.append("y must be a Pandas Series or NumPy array.")
        if isinstance(y, pd.Series) and y.isnull().any():
            issues.append("y contains missing values.")
        if isinstance(y, np.ndarray) and np.isnan(y).any():
            issues.append("y contains missing values.")

    if issues:
        return False, issues
    return True, ["Dataset is ready for XGBoost!"]


def retrieve_data(data_file):
    # Load pickled data as pandas dataframe
    with open(data_file, 'rb') as f:
        data = pd.read_pickle(f)
    X_train, X_test, y_train, y_test, X_train_groups, X_test_groups = data['X_train'], data['X_test'], data['y_train'], data['y_test'], data['X_train_groups'], data['X_test_groups']
    X_train, X_test = encode_categorical_features(X_train=X_train, X_test=X_test)
    return X_train, X_test, y_train, y_test, X_train_groups, X_test_groups

def FPR(y_true, y_pred):
    """Returns False Positive Rate.

    Parameters
    ----------
    y_true: array-like, bool 
        True labels. 
    y_pred: array-like, float or bool
        Predicted labels. 

    If y_pred is floats, this is the "soft" false positive rate 
    (i.e. the average probability estimate for the negative class)
    """
    # if there are no negative labels, return zero
    if np.sum(y_true) == len(y_true):
        return 0
    yt = y_true.astype(bool)
    return np.sum(y_pred[~yt])/np.sum(~yt)

def FNR(y_true, y_pred):
    """Returns False Negative Rate.

    Parameters
    ----------
    y_true: array-like, bool 
        True labels. 
    y_pred: array-like, float or bool
        Predicted labels. 

    If y_pred is floats, this is the "soft" false negative rate 
    (i.e. the average probability estimate for the negative class)
    """
    # if there are no postive labels, return zero
    if np.sum(y_true) == 0:
        return 0
    yt = y_true.astype(bool)
    return np.sum(1-y_pred[yt])/np.sum(yt)

def subgroup_loss(y_true, y_pred, X_protected, metric, grouping = 'intersectional', abs_val = False, gamma = True):
    assert isinstance(X_protected, pd.DataFrame), "X should be a dataframe"
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true, index=X_protected.index)
    else:
        y_true.index = X_protected.index

    y_pred = pd.Series(y_pred, index=X_protected.index)

    if (grouping == 'intersectional'):
        groups = list(X_protected.columns)
        categories = X_protected.groupby(groups).groups  
        #print("Categories: ", categories)
    else:
        categories = {}
        for col in X_protected.columns:
            unique_values = X_protected[col].unique()
            for val in unique_values:
                category_key = f'{col}_{val}'
                mask = X_protected[col] == val
                indices = X_protected[mask].index
                categories[category_key] = indices

    if isinstance(metric,str):
        loss_fn = FPR if metric=='FPR' else FNR
    elif callable(metric):
        loss_fn = metric
    else:
        raise ValueError(f'metric={metric} must be "FPR", "FNR", or a callable')

    base_loss = loss_fn(y_true, y_pred)
    max_loss = 0.0
    for c, idx in categories.items():
        # for FPR and FNR, gamma is also conditioned on the outcome probability
        if metric=='FPR' or loss_fn == FPR: 
            g = 1 - np.sum(y_true.loc[idx])/len(X_protected)
        elif metric=='FNR' or loss_fn == FNR: 
            g = np.sum(y_true.loc[idx])/len(X_protected)
        else:
            g = len(idx) / len(X_protected)

        category_loss = loss_fn(
            y_true.loc[idx].values, 
            y_pred.loc[idx].values
        )
        
        deviation = category_loss - base_loss

        if abs_val:
            deviation = np.abs(deviation)
        
        if gamma:
            deviation *= g

        if deviation > max_loss:
            max_loss = deviation

    return max_loss

def subgroup_FNR_loss(X_prime, y, y_pred):
    # Since it would be used as a scorer, we will assume est if already fitted
    #X_prime = X.loc[:, sens_features] 
    
    # Both y_val and y_proba should be pd.Series; Also checks whether they are 1D and have the same length as X_prime
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X_prime.index)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=X_prime.index)
    return subgroup_loss(y, y_pred, X_prime, 'FNR', grouping = 'intersectional', abs_val = True, gamma = True)

def evaluate_objective_functions(est, X, X_prime, y, objectives):
    score ={}
    y_prob = est.predict_proba(X)[:,1]
    y_pred = y_prob > 0.55

    for obj in objectives:
        if obj == 'balanced_accuracy':
            if not isinstance(y, np.ndarray):
                score[obj] = sklearn.metrics.balanced_accuracy_score(y.to_numpy(), y_pred)
            else:
                score[obj] = sklearn.metrics.balanced_accuracy_score(y, y_pred)
        elif obj == 'fpr':
            score[obj] = FPR(y, y_pred)
        elif obj == 'subgroup_fnr_loss':
            score[obj] = subgroup_FNR_loss(X_prime, y, y_pred)
        else:
            raise ValueError(f"Unknown objective: {obj}")
    return score


# generate traditional cross validation scores for tournament selection
def custom_objectives(est,X, X_prime ,y,cv, objectives, groups):
    # hold all the scores
    scores = []
    for train_index, test_index in cv.split(X, y, groups):
        # make a copy of the estimator
        this_fold_pipeline = sklearn.base.clone(est)
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                X_train, X_p, X_test = X.iloc[train_index], X_prime.iloc[test_index], X.iloc[test_index]
        else:
                X_train, X_p, X_test = X[train_index], X_prime[test_index], X[test_index]

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
                y_train, y_test = y[train_index], y[test_index]

        # fit model
        this_fold_pipeline.fit(X_train, y_train)
        this_fold_score = evaluate_objective_functions(this_fold_pipeline, X_test, X_p, y_test, objectives)
        scores.append(list(this_fold_score.values()))

        del this_fold_pipeline
        del X_train
        del X_test
        del y_train
        del y_test

    # make sure we have the right number of scores
    return np.mean(scores, 0)

# pipeline search space: selector(optional) -> transformer(optional) -> regressor/classifier(required)
def get_pipeline_space(n_features, seed):
    return tpot.search_spaces.pipelines.SequentialPipeline([
        GeneticFeatureSelectorNode(n_features=n_features),
        tpot.config.get_search_space('RandomForestClassifier', n_classes=2, random_state=seed)])
