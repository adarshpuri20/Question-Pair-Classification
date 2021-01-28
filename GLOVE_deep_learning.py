from __future__ import print_function
from keras.layers import LSTM
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.models import Sequential,Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import datetime, time
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.utils.vis_utils import plot_model
from keras import backend as K
#from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from keras.layers import merge
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.layers import PReLU
from keras.callbacks import Callback, ModelCheckpoint
import os
MODEL_WEIGHTS_FILE = 'C:/ALDA/Project/data/question_pairs_weights.h5' #weight parameter file to save
data = pd.read_csv("C:/ALDA/Project/data/train.csv") #test adn train data to read
y = data.is_duplicate.values
print(data.head(1))
#GloVe Embeddings
#
tk = text.Tokenizer(num_words=200000)

max_len = 40

tk.fit_on_texts(list(data.question1.values.astype(str)) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values.astype(str))
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tk.word_index
ytrain_enc = np_utils.to_categorical(y)


embeddings_index = {}
f = open('C:/ALDA/Project/data/glove.840B.300d.txt',encoding="utf-8")
for line in tqdm(f):
    values = line.split(" ")
    word = ''.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))
###
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
#####

RNG_SEED = 13371447
type(y)

# Partition the dataset into train and test sets
X = np.stack((x1, x2), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RNG_SEED)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]

# Define the model
MAX_SEQUENCE_LENGTH = 40 # can be set
question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
print(question2.shape)
#######

NB_EPOCHS = 18
MODEL_WEIGHTS_FILE = "question_pairs_weights/cp.ckpt"
VALIDATION_SPLIT=0.1
BATCH_SIZE = 32
#######
#embedding TD CNN layer
EMBEDDING_DIM=300
MAX_NB_WORDS=200000
nb_words = min(MAX_NB_WORDS, len(word_index))
q1 = Embedding(nb_words + 1, 
                  EMBEDDING_DIM, 
                  weights=[embedding_matrix], 
                  input_length=MAX_SEQUENCE_LENGTH, 
                  trainable=False)(question1)

q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)
print(q1.shape)
q2 = Embedding(nb_words + 1, 
                  EMBEDDING_DIM, 
                  weights=[embedding_matrix], 
                  input_length=MAX_SEQUENCE_LENGTH, 
                  trainable=False)(question2)

q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)
print(q2.shape)
# #######

DROPOUT=0.2
 #uncomment Line (117 to 163)  to use RELU activation and a TD CNN only model
 
# merged = concatenate([q1,q2])
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)

# is_duplicate = Dense(1, activation='sigmoid')(merged)
# ################

# OPTIMIZER = 'adam'
# model = Model(inputs=[question1,question2], outputs=is_duplicate)
# model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
# ###

# model.summary()
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

###

# # Train the model, checkpointing weights with best validation accuracy
# print("Starting training at", datetime.datetime.now())
# t0 = time.time()
# callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_accuracy', save_best_only=True)]
# #cp_callbacks=ModelCheckpoint((filepath=MODEL_WEIGHTS_FILE,
#                           #save_best_only=True,
#                           #verbose=1))
# history = model.fit([Q1_train, Q2_train],
#                     y_train,
#                     epochs=NB_EPOCHS,
#                     validation_split=VALIDATION_SPLIT,
#                     verbose=2,
#                     batch_size=BATCH_SIZE,
#                     callbacks=callbacks)
# t1 = time.time()
# print("Training ended at", datetime.datetime.now())
# print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

#embedding LSTM layer
m1 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=True)(question1)
m1 = LSTM(EMBEDDING_DIM,activation='relu',dropout=0.2,recurrent_dropout=0.2)(m1)#dropout=0.2

print(m1.shape)
m2 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=True)(question2)

m2 = LSTM(EMBEDDING_DIM,activation='relu',dropout=0.2,recurrent_dropout=0.2)(m2)#dropout=0.2

print(m2.shape)


mod1 = Embedding(nb_words + 1, 
                  EMBEDDING_DIM, 
                  weights=[embedding_matrix], 
                  input_length=MAX_SEQUENCE_LENGTH, 
                  trainable=False)(question1)
mod2 = Embedding(nb_words + 1, 
                  EMBEDDING_DIM, 
                  weights=[embedding_matrix], 
                  input_length=MAX_SEQUENCE_LENGTH, 
                  trainable=False)(question1)

filter_length = 5
nb_filter = 64
#embedding for 1d CNN (mod1 and mod2)
mod1=Convolution1D(filters=nb_filter,
                    kernel_size=filter_length,
                    padding='valid',
                    activation='relu',
                    dilation_rate=1)(mod1)

mod1= Dropout(DROPOUT)(mod1)
mod1=Convolution1D(filters=nb_filter,
                    kernel_size=filter_length,
                    padding='valid',
                    activation='relu',
                    dilation_rate=1)(mod1)


mod1=GlobalMaxPooling1D()(mod1)
mod1= Dropout(DROPOUT)(mod1)
mod1 = Dense(300, activation='relu')(mod1)
mod1= Dropout(DROPOUT)(mod1)
mod1 = BatchNormalization()(mod1)

mod2=Convolution1D(filters=nb_filter,
                    kernel_size=filter_length,
                    padding='valid',
                    activation='relu',
                    dilation_rate=1)(mod2)

mod2= Dropout(DROPOUT)(mod2)
mod2=Convolution1D(filters=nb_filter,
                    kernel_size=filter_length,
                    padding='valid',
                    activation='relu',
                    dilation_rate=1)(mod2)


mod2=GlobalMaxPooling1D()(mod2)
mod2= Dropout(DROPOUT)(mod2)
mod2 = Dense(300, activation='relu')(mod2)
mod2= Dropout(DROPOUT)(mod2)
mod2 = BatchNormalization()(mod2)

DROPOUT=0.2
# add the vector [mod1,mod2,q1,q2,m1,m2] as concatenate input to use LSTM+TDCNN+1DCNN
# add the vector [mod1,mod2,q1,q2] as concatenate input to use TDCNN+1DCNN
# add the vector [ ] as concatenate input to use combination of models

merged = concatenate([mod1,mod2,q1,q2,])#[mod1,mod2,q1,q2,m1,m2]

# uncomment below and comment above to use RELU instead of PRELU(to be commented)

# merged = BatchNormalization()(merged)
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)
# merged = Dense(300, activation='relu')(merged)
# merged = Dropout(DROPOUT)(merged)
# merged = BatchNormalization()(merged)

# ### Using Prelu#######
# uncomment below and comment above to use PRELU instead of RELU(have to comment)

merged = BatchNormalization()(merged)
merged = Dense(300)(merged)
merged=PReLU()(merged)
merged = Dropout(DROPOUT)(merged)

merged = BatchNormalization()(merged)
merged = Dense(300)(merged)
merged=PReLU()(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(300)(merged)
merged=PReLU()(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(300)(merged)
merged=PReLU()(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(300)(merged)
merged=PReLU()(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)

############
is_duplicate_super = Dense(1, activation='sigmoid')(merged)

OPTIMIZER = 'adam'
model_super= Model(inputs=[question1,question2], outputs=is_duplicate_super)
model_super.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

##
model_super.summary()
plot_model(model_super, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

##
MODEL_WEIGHTS_FILE = 'C:/ALDA/Project/data/quepair_weights_super(32_25_prelu).h5'
print("Starting training at", datetime.datetime.now())
t0 = time.time()
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_accuracy', save_best_only=True)]
history = model_super.fit([Q1_train, Q2_train],#,Q1_train, Q2_train,Q1_train, Q2_train],
                    y_train,
                    epochs=25,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1,
                    batch_size=32,
                    callbacks=callbacks)
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_accuracy']))
print('Maximum accuracy at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(max_val_acc))

model_super.load_weights(MODEL_WEIGHTS_FILE)
loss, accuracy = model_super.evaluate([Q1_test, Q2_test], y_test, verbose=0)
print('loss = {0:.4f}, accuracy = {1:.4f}'.format(loss, accuracy))