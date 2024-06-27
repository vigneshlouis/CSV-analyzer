
'''
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("website/uploads/diabetes.csv")

d = {'yes':1,'no':0}
df['Pregnancies'] = df['Pregnancies'].map(d)




X = df.drop(columns='Outcome',axis=1)
y = df['Outcome']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)



#Two  lines to make our compiler able to draw:

plt.figure(figsize=(12, 8))  # You can adjust the figure size as needed
tree.plot_tree(dtree,filled=True, feature_names=X.columns.tolist(),class_names=['No Diabetes', 'Diabetes'])
plt.show()  # This will display the plot
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# Load the Iris dataset
data=pd.read_csv("website/uploads/diabetes.csv")
X = data.drop(columns='Outcome') # Use only the first two features for visualization
y = data['Outcome']

# Create an SVM classifier with a linear kernel
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)

# Create a mesh grid to plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('SVM Decision Boundaries for Iris Dataset')
plt.show()
