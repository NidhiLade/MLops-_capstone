import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(csv_path, target='species', test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()>1 else None
    )
    return X_train, X_test, y_train, y_test