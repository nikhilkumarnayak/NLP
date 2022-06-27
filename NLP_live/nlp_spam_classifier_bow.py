# Spam Classifier

import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=["label","message"])

print(messages)

## Data Cleaning and Preprocessing

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

corpus = []
## Stemming
for i in range(len(messages)):
    reg_exp = '[^a-zA-Z]'
    review = re.sub(reg_exp,' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

print("corpus :- ", corpus)

## Bag Of Word
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
bow = cv.fit_transform(corpus)
print("BOW size :- ",bow.shape)
bow = bow.toarray()
print("BOW :- ",bow)

## Get the X & y
X = bow
print("X size is {0} ".format(X.shape))
y = pd.get_dummies(messages['label']) ## Converting the column value into categorial i.e ham & spam
# print(y)
y = y.iloc[:,1].values ## keepung one column i.e spam for train & test the model
print(y)

## Train Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("X_train size is {0} & X_test size (1)".format(X_train.shape,X_test.shape))
print("y_train size is {0} & y_test size (1)".format(y_train.shape,y_test.shape))

## Training model using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB().fit(X_train,y_train)
y_pred = spam_detection_model.predict(X_test)

print(y_pred)

## Confustion matrix
from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test,y_pred)
print("confusion_matrix :- ",con_mat)

## Check Prediction accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Prediction accuracy :- ",accuracy)