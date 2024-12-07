import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2]) #to see the changes in the dataset we can print the dataset before and after the transformation

#for the country column we have to use one hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##################################building the ANN#####################################
#initializing the ANN
ann = tf.keras.models.Sequential()
#adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #for binary classification we use sigmoid activation function

#compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#training the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#predicting the test set results
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5) #we have to scale the input values before predicting and use the threshold of 0.5 to get the prediction in the form of True or False
