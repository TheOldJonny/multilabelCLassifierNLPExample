import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import spacy

nlp=spacy.load('en_core_web_sm')



from nltk.corpus import stopwords
import spacy

en_stopwords = stopwords.words("english")
nlp = spacy.load('en_core_web_sm')

def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)

# Database downloaded from Kaggle at https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset/data
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Exploring the data
print(df.head())

# list of the available classes
list_classes = [df.columns[i] for i in range(3, len(df.columns))]
# print(df[list_classes])

y = df[list_classes].values

# Checking for missing values
print(df.isnull().sum())
print(df.describe())

df["full_text"] = df["TITLE"] + df["ABSTRACT"]
df['full_text'] = df['full_text'].apply(lemmatize_text)

df_test["full_text"] = df_test["TITLE"] + df_test["ABSTRACT"]
df_test['full_text'] = df_test['full_text'].apply(lemmatize_text)

X_train, X_valid, y_train, y_valid = train_test_split(df["full_text"], y, test_size=0.3)

vectorizer = TfidfVectorizer(stop_words=en_stopwords, analyzer="word")
X_train = vectorizer.fit_transform(X_train)
X_valid = vectorizer.transform(X_valid)

X_test = vectorizer.transform(df_test['full_text'])

print("Preprocessing finished")

clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid)
f1 = f1_score(y_valid, y_pred, average='micro')
print("F1-score Logistic Regression on the validation set:", f1)

clf_forest = OneVsRestClassifier(RandomForestClassifier())
clf_forest.fit(X_train, y_train)

y_pred_rf = clf_forest.predict(X_valid)
f1 = f1_score(y_valid, y_pred_rf, average='micro')
print("F1-score Random Forest on the validation set:", f1)

clf_svc = OneVsRestClassifier(SVC()) # default method for multilabel classification
clf_svc.fit(X_train, y_train)

y_pred_svc = clf_svc.predict(X_valid)
f1 = f1_score(y_valid, y_pred_svc, average='micro')
print("F1-score SVC on the validation set:", f1)