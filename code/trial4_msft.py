import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math 
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
%matplotlib inline 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

msft = pd.read_excel('MSFT_with_technical_data.xlsx')

msft_wo_1 = msft.drop('Unnamed: 27',axis=1)
msft_wo_2 = msft_wo_1.drop('Unnamed: 28', axis=1)
msft_wo_3 = msft_wo_2.drop('Unnamed: 29', axis=1)
msft_wo_4 = msft_wo_3.drop('Unnamed: 30', axis=1)
msft_wo_4 = msft_wo_4.drop('Names Date', axis=1)
msft_wo_4 = msft_wo_4.drop(0)

imp = SimpleImputer(strategy='mean')

X = msft_wo_4
y = msft_wo_4['Closing Price']

imp.fit(X)
X = imp.transform(X)

X_train_full, X_test, y_train_full, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
X_train,X_valid,y_train,y_valid = train_test_split(X_train_full,y_train_full, random_state = 42)

sc = StandardScaler()
X_train  = sc.fit_transform(X_train)
X_valid = sc.transform(X_test)
X_test = sc.transform(X_test)

X_train = pd.DataFrame(X_train)
X_valid = pd.DataFrame(X_valid)
X_test = pd.DataFrame(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(26, use_bias=True, activation='elu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(26,use_bias=True,activation='elu',kernel_initializer="he_normal", kernel_regularizer = regularizers.l2(0.01)),
    keras.layers.Dense(1)
])
opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss="mean_squared_error", optimizer=opt)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,10000)
plt.show()
