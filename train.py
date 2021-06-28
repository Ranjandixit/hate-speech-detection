import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re
import string
from sklearn.model_selection import train_test_split


def tokenize(s):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    out = regex.sub(' ', s).split()
    return out

def fit_logistic(x, y):
    y = y.values
    model = LogisticRegression(C=4, dual=True)
    return model.fit(x, y)

def training_code(PATH = "archive/labeled_data.csv"):
    COMMENT = 'tweet'
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    print("Load data...")
    train = pd.read_csv(PATH)

    print("Fill empty with unknown...")
    train[COMMENT].fillna('unknown', inplace=True)

    print("Train TFIDF vectorizer...")
    tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                                      min_df=3, max_df=0.9, strip_accents='unicode',
                                      use_idf=1, smooth_idf=True, sublinear_tf=1)

    train_term_doc = tfidfvectorizer.fit_transform(train[COMMENT])
    x = train_term_doc

    # joblib.dump(tfidfvectorizer, 'models/tfidf_vectorizer_train.pkl')
    with open('models/tfidf_vectorizer_train.pkl', 'wb') as tfidf_file:
        pickle.dump(tfidfvectorizer, tfidf_file)

    print("Fit logistic regression for each class...")
    for i, j in enumerate(label_cols):
        print("Fitting:", j)
        model = fit_logistic(x, train[j])
    
        # joblib.dump(model, 'models/logistic_{}.pkl'.format(j))
        with open('models/logistic_{}.pkl'.format(j), 'wb') as lg_file:
            pickle.dump(model, lg_file)



if __name__ == "__main__":
    training_code()