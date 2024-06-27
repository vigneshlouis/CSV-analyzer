import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
# Function to load and preprocess the CSV dataset
'''
def load_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    # Handle missing values, encoding of categorical features, and other preprocessing steps if needed
    # Check if the dataset has any missing values and handle them if needed
    if df.isnull().values.any():
        df.dropna(inplace=True)

    # Check if the dataset contains categorical features and encode them if needed
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        label_encoders = {}
        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
    return df

# Function to train a Random Forest classifier
def train_random_forest(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Plot feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify the path to your CSV file
    
    # Specify the name of the target column in your CSV dataset
    target_column_name = "ane"

    # Load and preprocess the CSV dataset
    df = load_and_preprocess_csv(csv_file_path)

    # Train Random Forest and plot feature importances
    train_random_forest(df, target_column_name)
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to load and preprocess the CSV dataset
def load_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    # Handle missing values, encoding of categorical features, and other preprocessing steps if needed
    if df.isnull().values.any():
        df.dropna(inplace=True)

    # Check if the dataset contains categorical features and encode them if needed
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        label_encoders = {}
        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
    return df

# Function to train a Random Forest classifier
def train_random_forest(df, target_column):

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create and train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Plot feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()


    return clf

# Function to make predictions using the trained Random Forest classifier
def make_predictions(trained_rf, input_string):
    # Split the input string into a list of feature values
    input_values = input_string.split(',')

    # Convert the input values to a NumPy array and reshape it to match the input shape
    input_array = np.array(input_values).reshape(1, -1)

    # Make predictions
    prediction = trained_rf.predict(input_array)

    return prediction[0]

if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_file_path = r"C:\Users\ELCOT\PycharmProjects\CSV Analyzer\website\uploads\diabetes.csv"

    # Specify the name of the target column in your CSV dataset
    target_column_name = "Outcome"

    # Load and preprocess the CSV dataset
    df = load_and_preprocess_csv(csv_file_path)

    # Train Random Forest
    trained_rf = train_random_forest(df, target_column_name)

    # Example: Make predictions for new input values (comma-separated string)
    input_string = "1,197,70,45,543,30.5,0.158,53"  # Replace with your own input

    # Make a prediction
    prediction = make_predictions(trained_rf, input_string)
    print("Prediction:", prediction)
