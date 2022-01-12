from sklearn.model_selection import train_test_split


def datasplit(df, target, split_size):
    """
    Returns X_train, y_train, X_val and y_val
    
    Parameters
    ----------
    df : pandas dataframe
        training data after preprocessing
    target : str
        name of the target column in the dataframe
    split_size : float
        split size required

    Returns
    ----------
        training set split further into X_train, y_train for training and X_val, y_val for validation
    """

    train_df_small, val_df = train_test_split(df, train_test_split=split_size, random_state=123)
    X_train, y_train = train_df_small.drop(columns=[target]), df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]
    return X_train, y_train, X_val, y_val

