# Load libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from jebisashvili_ketevan_neuralnetwork import Network, mse_prime, mse



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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) # 70% training and 30% test
X_train= X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an instance of the Network class
num_features = X_train.shape[1]
nn = Network(num_features)

# Set the loss function
nn.use(mse, mse_prime)

# Train the network
epochs = 100
learning_rate = 0.1
nn.fit(X_train, y_train, epochs, learning_rate)

# Make predictions
y_pred = nn.predict(X_test)
y_pred_binary = np.round(y_pred).flatten()



# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# Show the classification report
print(classification_report(y_test, y_pred_binary))