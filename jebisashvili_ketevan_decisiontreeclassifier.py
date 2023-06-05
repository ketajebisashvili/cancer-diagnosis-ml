# Load libraries

import pandas as pd

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

data = pd.read_csv("Cancer_Data.csv")

X = data.iloc[:, 1:31] 

y = data.iloc[:, 0]

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)
importances = tree.feature_importances_
sorted_importances = sorted(zip(importances, X.columns), reverse=True)
for importance, feature in sorted_importances:
    print(f"{feature}: {importance}")

# Create a mask to select the features with importance > 0.1
mask = importances > 0.01

# Select the features based on the mask
selected_features = X.loc[:, mask]

X = selected_features

# Retrieve the feature names
feature_names = X.columns.tolist()

# Print the feature names
print(feature_names)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_names,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('cancers.png')
Image(graph.create_png())
