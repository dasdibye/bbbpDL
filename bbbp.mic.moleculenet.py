import numpy as np
import pandas as pd
import sys, os

from rdkit import Chem
from rdkit.Chem import PandasTools

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

from gensim.models import word2vec

import tensorflow as tf
#seed(123)
from tensorflow.keras import layers

model = word2vec.Word2Vec.load('model_300dim.pkl')

#[7]
#Now read the csv file via pandas interface and store in a pandas array
bbbp_df= pd.read_csv('BBBP.csv')
bbbp_test_df= pd.read_csv('y_indices_external.csv')

#This prints the shape of the array
print(bbbp_df.shape)

#print the first few instances of the csv file as read by the pandas array
bbbp_df.head()


#[8]
#set the target or the values to be predicted/trained 
#this is just the p_np column of your csv file - the BBB values - 0/1
target = bbbp_df['p_np']
test_target = bbbp_test_df['p_np']

print(target.shape)

#for i in range(7161):
    #print(i+1)
    #print(bbbp_df['smiles'][i])
    #print(Chem.MolFromSmiles(bbbp_df['smiles'][i]))

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
mnk = set(mols)&keys

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

#[14]
#use the MLPClassifier from scikit to train
#currently this uses a simple multi-layer-perceptron(MLP) having 4 layers of hidden neurons with sizes
#hidden-layer-1 = 300 neurons, hidden-layer-2 = 500 neurons, hidden-layer-3 = 100 neurons, hidden-layer-4 = 50 neurons

#MLClass = MLPClassifier(hidden_layer_sizes=(300,500,100,50),alpha=0.00005, batch_size=16, random_state=1, max_iter=300)

#0.854 auc-roc with batch=128
MLClass = MLPClassifier(hidden_layer_sizes=(300,500,100,50),alpha=0.00005, batch_size=128, random_state=1, max_iter=300)
MLClass.fit(X_train, y_train)

#[15]
#Now predict using the evaluation function
auc_roc=roc_auc_score(y_test,MLClass.predict(X_test))

#This is the accuracy score which lies between 0 and 1
#Higher means we have better prediction accuracy
print('auc_roc=',auc_roc)

#[16]
from sklearn.ensemble import GradientBoostingClassifier
GBclf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
GBclf.fit(X_train,y_train)

#[17]
#Now predict using the evaluation function
auc_roc=roc_auc_score(y_test,GBclf.predict(X_test))

#This is the accuracy score which lies between 0 and 1
#Higher means we have better prediction accuracy
print('auc_roc=',auc_roc)

#[18]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

#Now predict using the evaluation function
auc_roc=roc_auc_score(y_test,neigh.predict(X_test))

#This is the accuracy score which lies between 0 and 1
#Higher means we have better prediction accuracy
print('auc_roc=',auc_roc)

#[19]
from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier(max_depth=2, random_state=0)
RFclf.fit(X_train, y_train)

#[20]
#Now predict using the evaluation function
auc_roc=roc_auc_score(y_test,RFclf.predict(X_test))

#This is the accuracy score which lies between 0 and 1
#Higher means we have better prediction accuracy
print('auc_roc=',auc_roc)


#####################################################
#For the final CNS test set

Xtvecall = np.zeros((Xt.shape[0],101,300))
ytvecall = np.zeros((yt.shape))

model1 = word2vec.Word2Vec.load('model_300dim.pkl')
mols = MolSentence(mol2alt_sentence(bbbp_test_df['mol'][1], radius=1))
keys = set(model1.wv.vocab.keys())
mnk = set(mols)&keys
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

Xvec_train, Xvec_test, yvec_train, yvec_test = train_test_split(Xvecall, yvecall, test_size=.1, random_state=1)

print(Xvec_train.shape)
print(Xvec_test.shape)

#seed(123)
#Accuracy: 90.25%
#('auc_roc=', 0.9316452777991239)

#Not much diff between 30 and 40 epochs
n_epoch=30
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
#fcoutput1  = (layers.Dense(300,activation="relu"))(pool12)
#fcoutput1  = (layers.Dense(200,activation="relu"))(fcoutput1)
#fcoutput1  = (layers.Dense(100,activation="relu"))(fcoutput1)
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
#scores = model.evaluate(Xvec_test, yvec_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
#auc_roc=roc_auc_score(yvec_test,model.predict(Xvec_test))
#print('auc_roc=',auc_roc)

def confusion_matrix_scorer(clf, X, y):
      y_pred = clf.predict(X)
      cm = confusion_matrix(y, np.round(y_pred))
      tn = cm[0, 0]
      fp = cm[0, 1]
      fn = cm[1, 0]
      tp = cm[1, 1]

      sens = (1.0*tp)/(1.0*tp+1.0*fn)
      spec = (1.0*tn)/(1.0*tn+1.0*fp)
      print("Sensitivity = ",sens)
      print("Specificity = ",spec)
      return sens,spec

#splits=10 gives 0.98106 on average
skf = StratifiedKFold(n_splits=10,random_state=1)
skf.get_n_splits(Xvecall, yvecall)
print(skf)

totAccuracy = 0.
totAucRoc = 0.

totCNSAccuracy = 0.
totCNSAucRoc = 0.

totSens = 0.
totSpec = 0.

for train_index, test_index in skf.split(Xvecall, yvecall):
     print("TRAIN:", train_index, "TEST:", test_index)
     Xvec_train, Xvec_test = Xvecall[train_index], Xvecall[test_index]
     yvec_train, yvec_test = yvecall[train_index], yvecall[test_index]

     history = model.fit(Xvec_train, yvec_train, epochs=n_epoch, batch_size=n_batch, verbose=2,shuffle=True, validation_split=0.11)
     scores = model.evaluate(Xvec_test, yvec_test, verbose=0)
     accuracy = (scores[1]*100)
     totAccuracy = totAccuracy + accuracy
     print("Accuracy LightBBB dataset: %.2f%%" % (scores[1]*100))

     auc_roc=roc_auc_score(yvec_test,model.predict(Xvec_test))
     totAucRoc = totAucRoc + auc_roc
     print('auc_roc LightBBB dataset =',auc_roc)
     print("Precision Score LightBBB dataset : ", precision_score(yvec_test,np.round(model.predict(Xvec_test))))
     print("Recall Score LightBBB datatset: ", recall_score(yvec_test,np.round(model.predict(Xvec_test))))
     sens1, spec1 = confusion_matrix_scorer(model,Xvec_test,yvec_test)
     totSens = totSens + sens1
     totSpec = totSpec + spec1

     cns_scores = model.evaluate(Xtvecall, ytvecall, verbose=0)
     print("Accuracy CNS test set: %.2f%%" % (scores[1]*100))
     cns_accuracy = (cns_scores[1]*100)
     totCNSAccuracy = totCNSAccuracy + accuracy

     cns_auc_roc=roc_auc_score(ytvecall,model.predict(Xtvecall))
     print('auc_roc CNS test set =',cns_auc_roc)
     totCNSAucRoc = totCNSAucRoc + cns_auc_roc

print("Average accuracy for stratified 10-fold = ",totAccuracy/10.)
print("Average auc_roc for stratified 10-fold = ",totAucRoc/10.)
print("Average sensitivity for stratified 10-fold = ",totSens/10.)
print("Average specificity for stratified 10-fold = ",totSpec/10.)

print("Average accuracy for CNS test = ",totCNSAccuracy/10.)
print("Average auc_roc for CNS test = ",totCNSAucRoc/10.)



#################################
#('Average accuracy for stratified 10-fold = ', 91.7766135931015)
#('Average auc_roc for stratified 10-fold = ', 0.9634442382711068)

#('Average accuracy for CNS test = ', 91.7766135931015)
#('Average auc_roc for CNS test = ', 0.8531639928698752)
#################################

#####
#Orig model
#####
#('Average accuracy for stratified 10-fold = ', 92.11637556552887)
#('Average auc_roc for stratified 10-fold = ', 0.9612481581216908)
#('Average accuracy for CNS test = ', 92.11637556552887)
#('Average auc_roc for CNS test = ', 0.8457219251336898)
#('Average sensitivity for CNS test = ', 0.9505658971936191)
#('Average specificity for CNS test = ', 0.8279463974939087)


##########################################
#conv1 = layers.Conv1D(800, 2, activation='relu')(sin1)
#pool1 = layers.GlobalMaxPooling1D()(conv1)
#conv2 = layers.Conv1D(800, 1, activation='relu')(sin1)
#pool2 = layers.GlobalMaxPooling1D()(conv2)
#pool12 = layers.concatenate([pool1, pool2])
#fcoutput1  = (layers.Dense(300,activation="relu"))(pool12)
#fcoutput1  = (layers.Dense(200,activation="relu"))(fcoutput1)
#fcoutput1  = (layers.Dense(100,activation="relu"))(fcoutput1)
#lstm = layers.Bidirectional(layers.LSTM(500, return_sequences=False, implementation=1, name="lstm_1"))(sin2)
#d1  = (layers.Dense(500,activation="relu"))(lstm)
#d1  = (layers.Dense(700,activation="relu"))(d1)
#fcoutput2  = (layers.Dense(100,activation="relu"))(d1)
#fc = layers.concatenate([fcoutput1, fcoutput2])
#fcoutput  = (layers.Dense(1,activation="sigmoid"))(fc)

##########################################
#('Average accuracy for stratified 10-fold = ', 92.49763190746307)
#('Average auc_roc for stratified 10-fold = ', 0.9620068626665578)
#('Average accuracy for CNS test = ', 92.49763190746307)
#('Average auc_roc for CNS test = ', 0.849643493761141)
#('Average sensitivity for CNS test = ', 0.9496427364457105)
#('Average specificity for CNS test = ', 0.8467629655412461)

