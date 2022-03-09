import numpy as np
import pandas as pd
import sys, os

from rdkit import Chem
from rdkit.Chem import PandasTools

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

from gensim.models import word2vec

import tensorflow as tf
#tf.seed(123)
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

model = word2vec.Word2Vec.load('model_300dim.pkl')

#[7]
bbbp_df= pd.read_csv('new_full_bbbp_trng.csv')
bbbp_test_df= pd.read_csv('ghosh_non_cns_test.csv')

#This prints the shape of the array
print(bbbp_df.shape)

#print the first few instances of the csv file as read by the pandas array
bbbp_df.head()


#[8]
#set the target or the values to be predicted/trained 
#this is just the p_np column of your csv file - the BBB values - 0/1
target = bbbp_df['p_np']
#test_target = bbbp_test_df['p_np']

print(target.shape)

#[9]
#Convert the smiles to their molecular representations and store in a 
#new pandas column called 'mol'
bbbp_df['mol'] = bbbp_df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
bbbp_test_df['mol'] = bbbp_test_df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

print('Conversion to mol done')

#check one molecular sentence at radius=1 
#print some other details
#these are optional and can be removed
#print('Molecular sentence:', mol2alt_sentence(bbbp_df['mol'][1], radius=1))
#print('\nMolSentence object:', MolSentence(mol2alt_sentence(bbbp_df['mol'][1], radius=1)))
#print('\nDfVec object:',DfVec(sentences2vec(MolSentence(mol2alt_sentence(bbbp_df['mol'][1], radius=1)), model, unseen='UNK')))

#[10]
#Some mol2vec setup
mols = MolSentence(mol2alt_sentence(bbbp_df['mol'][1], radius=1))
keys = set(model.wv.vocab.keys())
#mnk = set(mols)&keys

#Create the mol2vec sentences - each molecule is actually a sentence
#consisting of words which are the representations of atoms and their surrounding bound at radius=1
bbbp_df['sentence'] = bbbp_df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
bbbp_test_df['sentence'] = bbbp_test_df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

#Now create the MOL2VEC values for each molecule
#each molecule will have a 300-dim vector representation
bbbp_df['mol2vec'] = [DfVec(x) for x in sentences2vec(bbbp_df['sentence'], model, unseen='UNK')]
bbbp_test_df['mol2vec'] = [DfVec(x) for x in sentences2vec(bbbp_test_df['sentence'], model, unseen='UNK')]

#print the 300-dim representation of the first molecule to check
print(bbbp_df['mol2vec'][1])

#[11]
#Convert the pandas array to numpy arrays for training/prediction
X = np.array([x.vec for x in bbbp_df['mol2vec']])
print(X.shape)
y = np.array([x for x in bbbp_df['p_np']])
print(y.shape)

Xt = np.array([x.vec for x in bbbp_test_df['mol2vec']])
print(Xt.shape)
yt = np.array([x for x in bbbp_test_df['p_np']])
print(yt.shape)

#[12]
#Import scikit modules for training/prediction
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

#[13]
#Setting the training/test sets by splitting the original data
#0.8, 0.1, 0.1 train/test/val split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=.5, random_state=1)

#####################################################
#For the final CNS test set

Xtvecall = np.zeros((Xt.shape[0],101,300))
ytvecall = np.zeros((yt.shape))

model1 = word2vec.Word2Vec.load('model_300dim.pkl')
#mols = MolSentence(mol2alt_sentence(bbbp_test_df['mol'][1], radius=1))
keys = set(model1.wv.vocab.keys())
#mnk = set(mols)&keys

#For the external BBB file
for i in range(1,Xt.shape[0]):
  print (i)
  print (bbbp_test_df['sentence'][i])

  mols = MolSentence(mol2alt_sentence(bbbp_test_df['mol'][i], radius=1))

  xvecapp = np.zeros((101,300))
  ii = 0
  # carry the mol2vec on the 0th index of the sequence
  Xtvecall[i][0] = Xt[i]

  for yy in set(mols):
     if yy in keys:
        s2v1 = model1.wv.word_vec(yy)
        xvecapp[ii] = s2v1
        ii = ii + 1

  for j in range(100):
    for k in range(300):
        Xtvecall[i][j+1][k] = xvecapp[j][k]

  ytvecall[i] = yt[i]

#######################################################
#actual train/test of 7K molecules from LightBBB paper

Xvecall = np.zeros((X.shape[0],101,300))
yvecall = np.zeros((y.shape))

print(Xvecall.shape)
print(yvecall.shape)

#train
for i in range(1,X.shape[0]):
  #print (i)
  #print (bbbp_df['sentence'][i])

  mols = MolSentence(mol2alt_sentence(bbbp_df['mol'][i], radius=1))

  xvecapp = np.zeros((101,300))
  ii = 0
  # carry the mol2vec on the 0th index of the sequence
  Xvecall[i][0] = X[i]

  for yy in set(mols):
     if yy in keys:
        s2v1 = model.wv.word_vec(yy)
        xvecapp[ii] = s2v1
        ii = ii + 1

  for j in range(100):
    for k in range(300):
        Xvecall[i][j+1][k] = xvecapp[j][k]

  yvecall[i] = y[i]

#Xvec_train, Xvec_test, yvec_train, yvec_test = train_test_split(Xvecall, yvecall, test_size=.11, random_state=1)
#Xvec_test, Xvec_val, yvec_test, yvec_val = train_test_split(Xvec_test, yvec_test, test_size=.5, random_state=1)

#Xvec_train, Xvec_test, yvec_train, yvec_test = train_test_split(Xvecall, yvecall, test_size=.1, random_state=1)

#Try to use the entire set for training
Xvec_train, yvec_train= Xvecall, yvecall

print(Xvec_train.shape)
#print(Xvec_test.shape)

#seed(123)
#Accuracy: 90.25%
#('auc_roc=', 0.9316452777991239)

#Not much diff between 30 and 40 epochs
#n_epoch=30
n_epoch=35
n_batch=64

seqsize = 100
seq_inputs = layers.Input(shape=(101,300,), dtype='float32')
sin1 = seq_inputs[:,1:101,:]
sin2 = seq_inputs[:,0:1,:]
#conv1 = layers.Conv1D(1000, 2, activation='relu')(sin1)
conv1 = layers.Conv1D(800, 2, activation='relu')(sin1)
pool1 = layers.GlobalMaxPooling1D()(conv1)
#conv2 = layers.Conv1D(1000, 1, activation='relu')(sin1)
conv2 = layers.Conv1D(800, 1, activation='relu')(sin1)
pool2 = layers.GlobalMaxPooling1D()(conv2)
pool12 = layers.concatenate([pool1, pool2])
fcoutput1  = (layers.Dense(300,activation="relu"))(pool12)
fcoutput1  = (layers.Dense(200,activation="relu"))(fcoutput1)
fcoutput1  = (layers.Dense(100,activation="relu"))(fcoutput1)
lstm = layers.Bidirectional(layers.LSTM(500, return_sequences=False, implementation=1, name="lstm_1"))(sin2)
d1  = (layers.Dense(500,activation="relu"))(lstm)
d1  = (layers.Dense(700,activation="relu"))(d1)
fcoutput2  = (layers.Dense(100,activation="relu"))(d1)
fc = layers.concatenate([fcoutput1, fcoutput2])
fcoutput  = (layers.Dense(1,activation="sigmoid"))(fc)
model = tf.keras.Model(inputs=seq_inputs, outputs=fcoutput)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
print(model.summary())

#Accuracy: 90.25%
#('auc_roc=', 0.9350759130978911)

#history = model.fit(Xvec_train, yvec_train, epochs=n_epoch, batch_size=n_batch, verbose=2,shuffle=True, validation_split=0.11)

#Dont shuffle and dont keep anything for validation
history = model.fit(Xvec_train, yvec_train, epochs=n_epoch, batch_size=n_batch, verbose=2, validation_split=0.01)

ghosh_scores = model.evaluate(Xtvecall, ytvecall, verbose=0)
print("Gupta n-CNS Accuracy: %.2f%%" % (ghosh_scores[1]*100))

print("pred = ",np.round(model.predict(Xtvecall)))
print("pred = ",(model.predict(Xtvecall)))

ghoshpreds = np.round(model.predict(Xtvecall))

accr = 0
for i in range(ghoshpreds.shape[0]):
    if ( ghoshpreds[i] == ytvecall[i] ):
        accr = accr + 1

pcent = (accr*100)/(ghoshpreds.shape[0]*1.0)
print("Calculated Ghosh n-CNS Accuracy with 0.1 threshold : %.2f%%" % (pcent))

