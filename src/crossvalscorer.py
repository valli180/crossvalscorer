import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score



def datasplit(df, target, split_size):
    """
    Returns X_train, y_train, X_val and y_val

    Parameters
    ----------
    df :
        training data after preprocessing
    target : 
        target column in the dataframe
    split_size :
        split size required

    Returns
    ----------
        training set split further into X_train, y_train for training and X_val, y_val for validation
    """

    train_df_small, val_df = train_test_split(df, train_test_split=split_size, random_state=123)
    X_train, y_train = train_df_small.drop(columns=[target]), df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]
    return X_train, y_train, X_val, y_val


def crossvalscore(model, X_train, y_train, X_val, y_val, return_train_score=True):
     """
    Returns scores of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data
    X_val:
        X in the validation data
    y_val:
        y in the training data
    return_train_score:
        True/False 

    Returns
    ----------
        pandas Series with training and validation scores from cross_validation
    """

    start_time_fit = time.time()
    model.fit(X_train, y_train)
    end_time_fit = time.time()
    start_time_score = time.time()
    y_val_pred = model.predict(X_val)
    
    
    scores_dict = {
        "roc_auc_test":roc_auc_score(y_val, y_val_pred, average="weighted"),
        "f1_test":f1_score(y_val, y_val_pred, average="weighted"),
        "recall_test":recall_score(y_val, y_val_pred, average="weighted"),
        "precision_test": precision_score(y_val, y_val_pred, average="weighted")
    }
    end_time_score = time.time()
    scores_dict["score_time"] = end_time_score - start_time_score
    
    if return_train_score:
        y_train_pred = model.predict(X_train)
        scores_dict["roc_auc_train"] = roc_auc_score(y_train, y_train_pred, average="weighted"),
        scores_dict["f1_train"] = f1_score(y_train, y_train_pred, average="weighted"),
        scores_dict["recall_train"] = recall_score(y_train, y_train_pred, average="weighted"),
        scores_dict["precision_test"] = precision_score(y_train, y_train_pred, average="weighted")
        scores_dict["fit_time"] = end_time_fit - start_time_fit
        
    
    scores_results = pd.Series(scores_dict)
    
    return model, scores_results


    if __name__=="__main__":
        