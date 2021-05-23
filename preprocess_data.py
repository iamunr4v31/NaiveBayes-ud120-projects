import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif, SelectPercentile

def preprocess(word_file: str ='./word_data_unix.pkl', author_file: str ='./email_authors.pkl') -> tuple:
    with open(author_file, 'rb') as f:
        authors = pickle.load(f)
        f.close()
    
    with open(word_file, 'rb') as f:
        word_data = pickle.load(f)
        f.close()
    
    X_train, X_test, Y_train, Y_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)


    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X_train_transformed, Y_train)
    X_train_transformed = selector.transform(X_train_transformed).toarray()
    X_test_transformed  = selector.transform(X_test_transformed).toarray()

    print("no. of Chris's training emails: {}".format(sum(Y_train)))
    print("no. of Chris's training emails: {}".format(len(Y_train) - sum(Y_train)))

    return X_train_transformed, X_test_transformed, Y_train, Y_test

