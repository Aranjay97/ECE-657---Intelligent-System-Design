# Importing Libraries
import numpy as np # LIBRARY IMPORT FOR LINEAR ALGEBRA
import pandas as pd # LIBRARY IMPORT FOR DATA PROCESSING
from sklearn.model_selection import train_test_split # MODULE IMPORT FOR DATA SPLITTING

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

if __name__ == "__main__": 
    # 1. load your training data
    #Loading Data set
    Stock_Data = pd.read_csv("./data/q2_dataset.csv")
    store_data=np.zeros((1258,13))

    # The dataset was created in such a way to predict the next day opening price using the past 3 days Open, High, and Low prices and volume. 
    # So each sample contains 12 features and 1 target. 

    for i in range(len(store_data)-2):
        
        store_data[i][12]=Stock_Data.iloc[i+3][3] #target

        store_data[i][0]=Stock_Data.iloc[i+2][3] #open -1
        store_data[i][1]=Stock_Data.iloc[i+1][3] #open -2
        store_data[i][2]=Stock_Data.iloc[i][3] #open -3

        store_data[i][3]=Stock_Data.iloc[i+2][4] #High -1
        store_data[i][4]=Stock_Data.iloc[i+1][4] #High -2
        store_data[i][5]=Stock_Data.iloc[i][4] #High -3

        store_data[i][6]=Stock_Data.iloc[i+2][5] #Low -1
        store_data[i][7]=Stock_Data.iloc[i+1][5] #Low -2
        store_data[i][8]=Stock_Data.iloc[i][5] #Low -3

        store_data[i][9]=Stock_Data.iloc[i+2][2] #Volume -1
        store_data[i][10]=Stock_Data.iloc[i+1][2] #Volume -2
        store_data[i][11]=Stock_Data.iloc[i][2] #Volume -3

    col_names=['Open-1','Open-2','Open-3','High-1','High-2','High-3','Low-1','Low-2','Low-3','Volume-1','Volume-2','Volume-3','Target']

    df=pd.DataFrame(store_data[:-2,:],columns=col_names)
    data=df.drop(['Target'],axis=1)
    ran = 0
    #the dataset was randomized to create ‘train_data_RNN.csv’ and ‘test_data_RNN.csv.
    X_train, X_test, y_train, y_test = train_test_split(data, df['Target'], test_size=0.3, random_state = ran) 
    train_data=pd.concat([X_train,y_train],axis=1)
    test_data=pd.concat([X_test,y_test],axis=1)
    # Commenting the exporting of ‘train_data_RNN.csv’ and ‘test_data_RNN.csv’
    # train_data.to_csv(r'./data/train_data_RNN.csv', index = False, header=True)
    # test_data.to_csv(r'./data/test_data_RNN.csv', index = False, header=True)

    #reading the train data
    Train_Data = pd.read_csv("./data/train_data_RNN.csv")
    #creating the X_train
    X_train=Train_Data.drop(['Target'],axis=1)
    #creating y_train
    y_train=Train_Data['Target']

    #scaling the dataset using minmaxscaler
    scaler=MinMaxScaler(feature_range=(0,1))
    X_train=scaler.fit_transform(X_train)

    #numpy array conversion
    X_train=np.array(X_train)
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)



    # 2. Train your network
    model = Sequential()
    #adding LSTM layer with 50 LSTM units
    model.add(LSTM(50,input_shape=(X_train.shape[1],1),return_sequences=True))
    #adding LSTM layer with 150 LSTM units
    model.add(LSTM(150))
    #adding dense layer
    model.add(Dense(1,activation='linear'))

    #'mean_squared_error' has been used as loss function
    # Optimizer: Here adam optimizer has been used.
    # Adam is an adaptive learning rate optimization algorithm that’s been designed specifically for
    # training deep neural networks.

    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])

    History = model.fit(X_train,y_train,epochs=600,batch_size=64,verbose=2)
    # 		Make sure to print your training loss within training to show progress
    # 		Make sure you print the final training loss
    print('The final training loss is ',History.history['loss'][-1])



    # 3. Save your model
    # Please uncomment the following line to save the model in the models directory
    model.save('./models/Group25_RNN_model.h5')