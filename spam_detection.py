import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import wget


url = "https://github.com/fahmi54321/spam_detector/raw/refs/heads/main/spam.csv"
wget.download(url, 'spam.csv')


###############################################################################################################
# Notice that we specify the encoding ISO-8859-1. Why?
# 1. The CSV file contains some invalid characters, so using the default encoding (UTF-8) would produce errors.
# 2. In text processing, especially today with emojis and non-standard symbols, this issue is very common.
###############################################################################################################
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.head()



##############################
# Removing Unnecessary Columns
##############################
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.head()



###############################
# Renaming Columns for Clarity
###############################
df.columns = ['labels', 'data']
df.head()



#################################################
# Checking Class Distribution (Imbalanced or Not)
#################################################
df['labels'].hist()



######################################################
# Creating a Numeric Label Column (b_labels)
# 0 for ham
# 1 for spam
# We will also extract this column as a NumPy array.
######################################################
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].to_numpy()
Y
df.head()



######################################################
# Splitting the Data Into Train and Test Sets
######################################################
df_train, df_test, Ytrain, Ytest = train_test_split(
    df['data'], Y, test_size=0.33)


####################################################################################################################
# Building the Feature Matrix (X)
# There are two common feature extraction methods:
# 1. CountVectorizer → counts word frequencies
# 2. TfidfVectorizer → computes TF-IDF weights    

# The decode_error='ignore' parameter ensures that invalid characters are ignored, as mentioned earlier.
####################################################################################################################

# try multiple ways of calculating features
# featurizer = TfidfVectorizer(decode_error='ignore')
# Xtrain = featurizer.fit_transform(df_train)
# Xtest = featurizer.transform(df_test)

featurizer = CountVectorizer(decode_error='ignore')
Xtrain = featurizer.fit_transform(df_train)
Xtest = featurizer.transform(df_test)
Xtrain

######################################################
# Training the Model and Measuring Accuracy
######################################################
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train acc:", model.score(Xtrain, Ytrain))
print("test acc:", model.score(Xtest, Ytest))

######################################################
# Checking the F1 Score (Imbalanced Classes Issue)
######################################################
Ptrain = model.predict(Xtrain)
Ptest = model.predict(Xtest)
print("train F1:", f1_score(Ytrain, Ptrain))
print("test F1:", f1_score(Ytest, Ptest))

######################################################
# Checking AUC (Area Under the ROC Curve)
######################################################
Prob_train = model.predict_proba(Xtrain)[:,1]
Prob_test = model.predict_proba(Xtest)[:,1]
print("train AUC:", roc_auc_score(Ytrain, Prob_train))
print("test AUC:", roc_auc_score(Ytest, Prob_test))


###############################################################
# Viewing the Confusion Matrix for More Detailed Evaluation
###############################################################
cm = confusion_matrix(Ytrain, Ptrain)
cm


######################################################
# Plotting the Confusion Matrix
######################################################
def plot_cm(cm):
  classes = ['ham', 'spam']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  ax = sn.heatmap(df_cm, annot=True, fmt='g')
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Target")
plot_cm(cm)


###################################################
# Plotting the Confusion Matrix for the Test Data
###################################################
cm_test = confusion_matrix(Ytest, Ptest)
plot_cm(cm_test)




###################################################
# Creating Word Clouds for Spam and Ham
###################################################
def visualize(label):
  words = ''
  for msg in df[df['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()
  
visualize('spam')
visualize('ham')




#####################
# Error Analysis
#####################
X = featurizer.transform(df['data'])
df['predictions'] = model.predict(X)




###############################################################
# Detecting Spam That “Slipped Through” (False Negatives)
###############################################################
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
  print(msg)
  
  
  
  
  
###############################################################
# Detecting Normal Messages Mistaken as Spam (False Positives)
###############################################################
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
  print(msg)