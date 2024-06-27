import numpy as np
from skimage import feature, io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage import color
# Load a toy dataset of images and labels (you should replace this with your dataset)
# The images should be preprocessed and converted into feature vectors.
# In this example, we'll use HOG features for simplicity.
data = []  # Each element of data should be a tuple (feature_vector, label)

# Load and preprocess images
for i in range(0, 10):
    image = io.imread(f'C:\\Users\\ELCOT\\Downloads\\dataset-master\\dataset-master\\a\BloodImage_0000{i}.jpg')


    # Load a color image (replace 'your_image.jpg' with the actual image file path)


    # Convert to grayscale
    grayscale_image = color.rgb2gray(image)
    feature_vector = feature.hog(grayscale_image, pixels_per_cell=(16, 16))
    data.append((feature_vector, 'ClassA'))

for i in range(0, 10):
    image = io.imread(f'C:\\Users\\ELCOT\\Downloads\\dataset-master\\dataset-master\\b\BloodImage_0010{i}.jpg')
    feature_vector = feature.hog(image, pixels_per_cell=(16, 16))
    data.append((feature_vector, 'ClassB'))

# Split the dataset into training and testing sets
X = [item[0] for item in data]
y = [item[1] for item in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# You can also analyze feature importances if needed
feature_importances = clf.feature_importances_
print("Feature Importances:")
print(feature_importances)
