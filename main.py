
#dataset handling
import pandas as pd
df=pd.read_csv("dataset.csv",encoding="latin-1",header=None)
df=df[[0,5]]
df.columns=["polarity","text"]
print(df.head())
print(df['polarity'].value_counts())
df=df[df['polarity'].isin([0,4])]
df['polarity'] = df['polarity'].map({0:0, 4:1})
print(df.head())
#data cleaning
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text
df['clean_text'] = df['text'].apply(clean_text)
print(df[['text','clean_text']].head())
#Import Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['polarity'],
    test_size=0.2,
    random_state=42
)
print(len(X_train))
print(len(X_test))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english",max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(X_train_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
print(df['polarity'].value_counts())
model.fit(X_train_tfidf, y_train)
print("Naive Bayes Model trained successfully")
predictions = model.predict(X_test_tfidf)
print(predictions[:20])
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
sample = ["I hate this movie"]
sample_vector = vectorizer.transform(sample)
result = model.predict(sample_vector)
print(result)
y_pred = model.predict(X_test_tfidf)
#Model Evaluation
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
#Get classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
