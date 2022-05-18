# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:10:52 2022

@author: User
"""

#scientific math
import numpy as np
import pandas as pd
import datetime
import os

#Visualization

import matplotlib.pyplot as plt
import seaborn as sns

#Deep learning
from tensorflow.keras import Sequential, Input
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

#Data preprocessing
from sklearn.preprocessing import LabelEncoder
#One hot Encoder
from sklearn.preprocessing import OneHotEncoder

# #features selection


#fit and validate
from sklearn.model_selection import train_test_split

#Reporting
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report

#%% PATH

TRAIN_PATH = os.path.join(os.getcwd(),'train.csv')

#the new customers csv file has been filled with the Segmentation column
#with Vlookup and saved under test.csv for not violating the original dataset
TEST_PATH = os.path.join(os.getcwd(),'test.csv')

MODEL_PATH = os.path.join(os.getcwd(),'cust_segmentation.h5')
LOG_PATH = os.path.join(os.getcwd(),'Assessment2')
nb_classes=10
# %%

def training_process(hist):
    
    keys = [i for i in hist.history.keys()]
    
    training_loss = hist.history[keys[0]]
    training_metric = hist.history[keys[1]]
    validation_loss = hist.history[keys[2]]
    validation_metric = hist.history[keys[3]]
    
    plt.figure()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title('training loss and validation loss')
    plt.xlabel('epoch')
    plt.ylabel(keys[0])
    plt.legend(['training loss', 'validation loss'])
    plt.show()
    
    plt.figure()
    plt.plot(training_metric)
    plt.plot(validation_metric)
    plt.title('training acc and validation acc')
    plt.xlabel('epoch')
    plt.ylabel(keys[1])
    plt.legend(['training acc', 'validation acc'])
    plt.show()
    
    return (hist)

def report_generation(X_test, y_test):
    
    #x_test contains the features
    # pred_x contains the predict result from the model
    # y_test has the true label
    
    pred_x = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(pred_x, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    # cr = classification_report(y_true,y_pred)

    print(classification_report(y_true, y_pred))
    
    #confusion matrix display
    
    # labels = [str(i) for i in range (10)]
    # or you can list the array as below
    # labels = [str(i) for i in range(10)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                                
    # metrics.f1_score(y_test, y_pred, , labels=np.unique(y_pred))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    
#%% EDA

#Step 1: Load data

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

#to concatenate both df_train and df_test
df_train['y_train'] = 1
df_test['y_train'] = 0
#Our new dataset is now df_combine
df_combine = pd.concat([df_train,df_test])

#Step 2: Data inspection/visualization

#checking on the first 10 rows
df_combine.head(10)

#checking on the last 10 rows
df_combine.tail(10)

#now we have extra column feeded named: y_train
df_combine.info()
df_combine.describe()

#to check null values
#Ever married, Graduated, profession, work experience, family size and var 1 has
#large null values
df_combine.isnull().sum()

#replace NaN values with median --> work experiene & family size
df_combine[['Work_Experience','Family_Size']] = df_combine[['Work_Experience',
                                                        'Family_Size']].fillna(df_combine[['Work_Experience',
                                                                                         'Family_Size']].median())
#checking on duplicate values :no duplicate value
df_combine.duplicated().sum()


# STEP 3: Data cleaning

#1: to convert column category (string to number)
label = LabelEncoder()
df_combine.Ever_Married = label.fit_transform(df_combine.Ever_Married.astype(str))
df_combine.Gender = label.fit_transform(df_combine.Gender.astype(str))
df_combine.Graduated = label.fit_transform(df_combine.Graduated.astype(str))
df_combine.Profession = label.fit_transform(df_combine.Profession.astype(str))
df_combine.Spending_Score = label.fit_transform(df_combine.Spending_Score.astype(str))
df_combine.Var_1 = label.fit_transform(df_combine.Var_1.astype(str))
df_combine.Segmentation = label.fit_transform(df_combine.Segmentation.astype(str))


#Since we are looking at customer's buying pattern (spending score)
# then the possibilities of correlation could be related to gender,age,
# family size and even profession
# Lets look at th edata visualizing to interperate 

f, ax = plt.subplots(1,1, figsize=(20,10))
corr = df_combine.corr()
ax = sns.heatmap(corr, annot=True, cmap='Reds')

# STEP 4: Features selection

x1 = df_combine['Gender']
x2 = df_combine['Age']
x3 = df_combine['Profession']
x4 = df_combine['Family_Size']

X = [x1,x2,x3,x4]
X = np.array(X).T
y = df_combine['Spending_Score']

enc = OneHotEncoder(sparse=False)
X_enc = enc.fit_transform(X)
y = enc.fit_transform(np.expand_dims(y,axis=-1)) 
# STEP 5: Data preprocessing

#split test 
X_train, X_test, y_train, y_test = train_test_split(X_enc,y,test_size=0.3,stratify=y, 
                                                    random_state=2)

#4.model training

# 1. Create a sequential model

model = Sequential()

#2. placing items into container 
model.add(Input(shape=(88)))
model.add(Dense(60, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))

#3.compile model
model.compile(optimizer = 'adam',
                            loss='categorical_crossentropy',
                            metrics='acc')
model.summary()


# to stop at over-fitting point
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

#Tensorboard

log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_files)

hist = model.fit(X_train,y_train, epochs=30, validation_data=(X_test, y_test), 
                  callbacks=[early_stopping_callback, tensorboard_callback])


#to visualize training loss using matploblib 
training_process(hist)

# %% Reporting

report_generation(X_test, y_test)
