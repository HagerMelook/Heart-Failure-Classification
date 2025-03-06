import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def split_features_target(df, target_column):
    """Split the dataset into features (X) and target (y)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def split_train_validation_test(X, y, train_size=0.7, test_size=0.2, val_size=0.1, random_state=42):
    """Split dataset into training, validation, and test sets while maintaining class distribution."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_size), stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(test_size / (test_size + val_size)), stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def print_class_distribution(y_train, y_val, y_test):
    """Print class distributions for training, validation, and test sets."""
    print("Training Class Distribution:\n", y_train.value_counts(normalize=True))
    print("Validation Class Distribution:\n", y_val.value_counts(normalize=True))
    print("Test Class Distribution:\n", y_test.value_counts(normalize=True))

def encode_categorical_columns(df, categorical_columns):
    """Perform one-hot encoding for categorical columns."""
    return pd.get_dummies(df, columns=categorical_columns, dtype='uint8')

def standardize_features(df):
    """Standardize numerical features using StandardScaler."""
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def main():
    # Load dataset
    df = load_data("heart.csv")
    print(df.head())
    
    # Split features and target
    X, y = split_features_target(df, 'HeartDisease')
    
    # Define categorical columns to encode
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_encoded = encode_categorical_columns(X, categorical_columns)
    print("Encoded Columns:", df_encoded.columns)

    # Split data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(df_encoded, y)
    
    # Validate class distribution
    print_class_distribution(y_train, y_val, y_test)
        
    # if needed, Standardize dataset
    df_standardized = standardize_features(df_encoded)
    print(df_standardized.head())

if __name__ == "__main__":
    main()
