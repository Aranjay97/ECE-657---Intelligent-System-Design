# import required packages
from glob import glob
import os,re,string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

#Load data from folder
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
	# 1. Load your saved model
    model=tf.keras.models.load_model('./models/Group25_NLP_model.h5')

    # 2. Load your testing data
    PATH='./data/aclImdb/'
    names = ['neg','pos']
    x_test,y_test = load_texts_labels_from_folders(f'{PATH}test',names)
    x_train,y_train = load_texts_labels_from_folders(f'{PATH}train',names)

    x_train_clean = preprocess_reviews(x_train)
    x_test_clean = preprocess_reviews(x_test)

    # Tokenizer
    tok = keras.preprocessing.text.Tokenizer()
    tok.fit_on_texts(x_train_clean)
    X_test = tok.texts_to_sequences(x_test_clean)

    X_test = keras.preprocessing.sequence.pad_sequences(X_test,padding='post',maxlen=1000)

    scores = model.evaluate(X_test,y_test)
    test_accuracy = scores[1]
    print('accuracy on testing set:',test_accuracy*100)