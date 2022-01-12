# import pandas as pd
# import numpy as np
# import time

# from sklearn.metrics import (
#     f1_score,
#     roc_auc_score,
#     recall_score,
#     precision_score,
#     mean_absolute_percentage_error,
#     mean_squared_error,
#     make_scorer
# )

# def cross_val_score_classfication(model, X_train, y_train, X_val, y_val, return_train_score=True):
#     """
#     Returns scores of cross validation

#     Parameters
#     ----------
#     model :
#         scikit-learn model
#     X_train : numpy array or pandas DataFrame
#         X in the training data
#     y_train :
#         y in the training data
#     X_val:
#         X in the validation data
#     y_val:
#         y in the training data
#     return_train_score:
#         True/False 

#     Returns
#     ----------
#         pandas Series with training and validation scores from cross_validation
#     """
#     start_time_fit = time.time()
#     model.fit(X_train, y_train)
#     end_time_fit = time.time()
#     start_time_score = time.time()
#     y_val_pred = model.predict(X_val)
    
    
#     scores_dict = {
#         "roc_auc_test":roc_auc_score(y_val, y_val_pred, average="weighted"),
#         "f1_test":f1_score(y_val, y_val_pred, average="weighted"),
#         "recall_test":recall_score(y_val, y_val_pred, average="weighted"),
#         "precision_test": precision_score(y_val, y_val_pred, average="weighted")
#     }
#     end_time_score = time.time()
#     scores_dict["score_time"] = end_time_score - start_time_score
    
#     if return_train_score:
#         y_train_pred = model.predict(X_train)
#         scores_dict["roc_auc_train"] = roc_auc_score(y_train, y_train_pred, average="weighted"),
#         scores_dict["f1_train"] = f1_score(y_train, y_train_pred, average="weighted"),
#         scores_dict["recall_train"] = recall_score(y_train, y_train_pred, average="weighted"),
#         scores_dict["precision_test"] = precision_score(y_train, y_train_pred, average="weighted")
#         scores_dict["fit_time"] = end_time_fit - start_time_fit
        
    
#     scores_results = pd.Series(scores_dict)
    
#     return model, scores_results



# def cross_val_score_regression(model, X_train, y_train, X_val, y_val, return_train_score=True):
#     """
#     Returns scores of cross validation

#     Parameters
#     ----------
#     model :
#         scikit-learn model
#     X_train : numpy array or pandas DataFrame
#         X in the training data
#     y_train :
#         y in the training data
#     X_val:
#         X in the validation data
#     y_val:
#         y in the training data
#     return_train_score:
#         True/False 

#     Returns
#     ----------
#         pandas Series with training and validation scores from cross_validation
#     """

#     model.fit(X_train, y_train)
#     y_val_pred = model.predict(X_val)

#     score_dict = {
#         "r2_test": model.score(X_val, y_val),
#         "mse_test": mean_squared_error(y_val, y_val_pred),
#         "mape_test": mean_absolute_percentage_error(y_val, y_val_pred)
#     }

#     if return_train_score:
#         y_train_pred = model.predict(X_train)

#         score_dict["r2_train"] = model.score(X_train, y_train)
#         score_dict["mse_train"] = mean_squared_error(y_train, y_train_pred)
#         score_dict["mape_train"] = mean_absolute_percentage_error(y_train, y_train_pred)

#     scores_result = pd.Series(score_dict)

#     return model, scores_result



