# import required packages
from glob import glob
import os,re,string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading data from folder
def load_texts_labels_from_folders(path, folders):
    texts,labels = [],[]
    for idx,label in enumerate(folders):
        for fname in glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r',encoding="utf8").read())
            labels.append(idx)
    
    return texts, np.array(labels).astype(np.int64)

# Preprocessing
def preprocess_reviews(reviews):
    tokens = re.compile("[.;:!#\'?,\"()\[\]]|(<br\s*/><br\s*/>)|(\-)|(\/)")
    
    return [tokens.sub("", line.lower()) for line in reviews]

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
if __name__ == "__main__":
  
	# 1. load your training data  
  PATH='./data/aclImdb/'
  names = ['neg','pos']
  x_train,y_train = load_texts_labels_from_folders(f'{PATH}train',names)

  x_train_clean = preprocess_reviews(x_train)

  # Tokenizer
  tok = keras.preprocessing.text.Tokenizer()
  tok.fit_on_texts(x_train_clean) 
  X_train = tok.texts_to_sequences(x_train_clean)

  from sklearn.model_selection import train_test_split
  X_train = keras.preprocessing.sequence.pad_sequences(X_train,padding='post',maxlen=1000)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy
  word_size = len(tok.word_index)+1

  model = keras.Sequential()
  model.add(keras.layers.Embedding(word_size, 16))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Conv1D(filters=16,kernel_size=2,padding='valid',activation='relu'))
  model.add(keras.layers.GlobalAveragePooling1D())
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(32, activation='relu'))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  model.summary()

  model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
  history = model.fit(X_train,y_train,epochs=20,validation_data=(X_val, y_val),verbose=1,batch_size=512)

  print('The final training accuracy is ',history.history['acc'][-1]*100)

  # 3. Save your model
  model.save("./models/Group25_NLP_model.h5")