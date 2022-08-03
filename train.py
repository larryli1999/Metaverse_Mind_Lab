from unicodedata import name
import pandas as pd
import numpy as np
import pre_process
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pickle
import os


def main():
    # preprocess the text data
    print('Start data processing ...')
    tqdm.pandas()
    # df_train = pd.read_csv('./fake_news/train.csv')
    df_train = pd.read_csv('./fake_news/cleaned_train.csv')
    if 'clean_text' not in df_train.columns:
        df_train = df_train.dropna(subset=['text'])
        df_train['clean_text'] = df_train['text'].progress_apply(lambda x: pre_process.process_text(x))
        df_train.to_csv('./fake_news/cleaned_train.csv') # save to csv to save time for next iteration of training

    # train test split for model training   
    df_train = df_train.dropna(subset=['text','clean_text'])
    X_train, X_test, y_train, y_test = train_test_split(df_train["clean_text"],df_train["label"],test_size=0.3,shuffle=True)
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    print('Done data processing ...\n')

    # Train the classification model
    print('Start model training ...')
    model=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
    model.fit(X_train_vectors_tfidf, y_train)
    print('Done model training ...\n')

    # evaluate the model on dev set
    print('Start evaluation ...')
    y_predict = model.predict(X_test_vectors_tfidf)
    print('Evaluation result on dev set:\n')
    print(classification_report(y_test,y_predict))

    # save model
    model_name = 'model.sav'
    pickle.dump(model, open(model_name,'wb'))
    print('Model has been save to: {}\n'.format(os.getcwd()+'/'+model_name))

    # save tfidf vectorizer
    feature_name = 'feature.pkl'
    pickle.dump(tfidf_vectorizer.vocabulary_,open(feature_name,'wb'))
    print('TFIDF vectorizer has been save to: {}\n'.format(os.getcwd()+'/'+feature_name))

if __name__ == '__main__':
    main()