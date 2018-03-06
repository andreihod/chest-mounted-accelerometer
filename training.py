# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#tf.logging.set_verbosity(tf.logging.DEBUG)

print "Carregando o dataset..."

# carrega os dados
data = pd.read_csv("dataset.csv", sep=",")
#data = pd.read_csv("15.csv", sep=",")

classes = data["label"].value_counts()

train, test = train_test_split(data, test_size=0.3)

features = ["x_acceleration","y_acceleration","z_acceleration"]
# carrega as features
x_train = train[features]
x_test = test[features]
# converte para formato do numpy
x_train = x_train.astype(np.float32).values
x_test = x_test.astype(np.float32).values

# carrega o label
y_train = train["label"]
y_test = test["label"]
y_train = y_train.astype(np.int32).values
y_test = y_test.astype(np.int32).values

# instancia os parâmetros do random forest
params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
  num_classes=classes.count(), num_features=3, regression=False,
  num_trees=100, max_nodes=10000).fill()

# cria o classifier
classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)

# treina o classifier
print "Treinando..."
classifier.fit(x=x_train, y=y_train, steps=10)

# testa o modelo
print "Testando o modelo..."

# pega somente as classes
y_predicted = list(classifier.predict(x=x_test))
y_out = list(y['classes'] for y in y_predicted)

print("Accuracy = %s" % accuracy_score(y_test, y_out))