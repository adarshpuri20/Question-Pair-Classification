from __future__ import print_function
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.models import Sequential,Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence, text
from keras.layers import merge
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.layers import LSTM
from keras.callbacks import Callback, ModelCheckpoint

import gensim
from gensim.utils import simple_preprocess

#Read data and drop null vals
df = pd.read_csv("train.csv")
df.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True)

#Simple preprocess before word2Vec
def read_questions(row,column_name):
    return simple_preprocess(str(row[column_name]).encode('utf-8'))
    
documents = []
q1 = []
q2 = []
for index, row in df.iterrows():
    question1 = read_questions(row,"question1")
    question2 = read_questions(row,"question2")
    documents.append(question1)
    documents.append(question2)
    q1.append(question1)
    q2.append(question2)

#Create the word2vec model
model = gensim.models.Word2Vec(size=300, window=10, min_count=2, sg=1, workers=10)
model.build_vocab(documents)
model.train(sentences=documents, total_examples=len(documents), epochs=model.iter)

word_vectors = model.wv

#Pad the input sequences to be of equal size
tk = text.Tokenizer(num_words=200000)

max_len = 30

tk.fit_on_texts(documents)
x1 = tk.texts_to_sequences(q1)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(q2)
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tk.word_index
y = df.is_duplicate.values
ytrain_enc = np_utils.to_categorical(y)

#Creaating the embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, 300))
i = 0
for word in word_index:
    if word in word_vectors:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    i += 1

#Defining the training and testing data
RNG_SEED = 13371447
X = np.stack((x1, x2), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RNG_SEED)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]

#Defining the model
embedding_layer = Embedding(len(word_index) + 1,
        300,
        weights=[embedding_matrix],
        input_length=30,
        trainable=False)

lstm_layer = LSTM(200, dropout=0.20, recurrent_dropout=0.20)

sequence_1_input = Input(shape=(30,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1_layer = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(30,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1_layer = lstm_layer(embedded_sequences_2)

merged = concatenate([x1_layer, y1_layer])
merged = Dropout(0.25)(merged)
merged = BatchNormalization()(merged)

merged = Dense(300, activation='relu')(merged)
merged = Dropout(0.25)(merged)
merged = BatchNormalization()(merged)

merged = Dense(100, activation='relu')(merged)
merged = Dropout(0.25)(merged)
merged = BatchNormalization()(merged)

out = Dense(1, activation='sigmoid')(merged)


#Model Training
model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=out)

model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
model_path = 'lstm.h5'
model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([Q1_train, Q2_train], y_train, 
        validation_split=0.1, 
        epochs=200, batch_size=2048, shuffle=True, 
        callbacks=[early_stopping, model_checkpoint])

model.load_weights(model_path)
min_val_score = min(hist.history['val_loss'])
print('Minimum validation loss achived while training = {:.4f}'.format(min_val_score))


score, acc = model.evaluate([Q1_test, Q2_test], y_test,batch_size=2048)
print('loss = {0:.4f}, accuracy = {1:.4f}'.format(score, acc))

