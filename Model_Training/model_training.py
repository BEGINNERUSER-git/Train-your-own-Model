from sklearn.model_selection import train_test_split

def split_data(data, feature_columns, target_column, test_size=0.2, random_state=42):
    X = data[feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
