import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Conv1D, GlobalMaxPooling1D, Dropout

data = {
    "text": [
        "I love this movie",
        "This film was terrible",
        "Amazing acting",
        "Worst movie ever",
        "I enjoyed it",
        "I hate this movie",
        "Best film I have seen",
        "Awful movie",
        "Great plot and acting",
        "Terrible film"
    ],
    "label": [1,0,1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_counts, y_train)
y_pred_nb = nb_model.predict(X_test_counts)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 10
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_len))
rnn_model.add(LSTM(64))
rnn_model.add(Dense(1, activation='sigmoid'))

rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(X_train_pad, y_train, epochs=10, batch_size=2, verbose=0)
rnn_acc = rnn_model.evaluate(X_test_pad, y_test, verbose=0)[1]
print("RNN (LSTM) Accuracy:", rnn_acc)

cnn_model = Sequential()
cnn_model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_len))
cnn_model.add(Conv1D(128, 3, activation='relu'))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.fit(X_train_pad, y_train, epochs=10, batch_size=2, verbose=0)
cnn_acc = cnn_model.evaluate(X_test_pad, y_test, verbose=0)[1]
print("CNN Accuracy:", cnn_acc)
