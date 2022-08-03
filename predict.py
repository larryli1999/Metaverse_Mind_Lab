import pandas as pd
import numpy as np
import pre_process
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import pickle
import os

def main():
    # preprocess test data
    print('Start data processing ...')
    tqdm.pandas()
    df_test = pd.read_csv('./fake_news/test.csv')
    df_label = pd.read_csv('./fake_news/labels.csv')
    df_test['label'] = df_label['label']
    df_test = df_test.dropna(subset=['text'])
    df_test['clean_text'] = df_test['text'].progress_apply(lambda x: pre_process.process_text(x))
    df_test = df_test.dropna(subset=['text','clean_text'])

    X_test, y_test = df_test['clean_text'], df_test['label']
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error='replace', vocabulary=pickle.load(open('feature.pkl','rb')))
    X_test_vectors_tfidf = transformer.fit_transform(loaded_vec.fit_transform(X_test))
    print('Done data processing ...\n')

    # load model
    loaded_model = pickle.load(open('model.sav', 'rb'))
    print('model loaded ... \n')

    # run prediction and compute metrics
    print('Start running prediction ...')
    y_predict = loaded_model.predict(X_test_vectors_tfidf)
    print('Prediction result:\n')
    print(classification_report(y_test,y_predict))

if __name__ == '__main__':
    main()