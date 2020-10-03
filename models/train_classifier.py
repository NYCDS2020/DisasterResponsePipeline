import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData,  select, func, Integer, Table, Column

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

from sklearn.metrics import precision_recall_fscore_support

def load_data(database_filepath):
    '''
    Function: 
        Load table from database
    Args:
        database_filepath: path to database holding the data
    Output: 
        X, y: ML-ready pandas dataframes
    '''
    # load data from database
    table_name = 'DisasterResponseData' #Ready to be converted to input variable - future use
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("SELECT * FROM " + table_name, engine)
    # Read columns from dataframe into arrays for ML
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names
    


def tokenize(text):
    '''
    Function: 
        Tokenize text for use in vectorizers, remove language-specific stop words
    Args:
        text: text string to be tokenized
    Output: 
        words_lemmed: lemmatized text, cleansed of stop words
    '''

    #Convert to lower case, remove non-alphanumeric chars
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words('english')
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    #lemmatizing and eliminating stop words
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in tokens if w not in stop_words]
   
    return words_lemmed


def build_model():
    '''
    Function: 
        Create model - RandomForestClassifier in this example
    Args:
        None: could be expanded for future use to specify which kind of model to build
    Output: 
        cv: GridSearch model output
    '''

    pipeline =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), # Initial vectorization
        ('tfidf', TfidfTransformer()),                 # TF/IDF to highlight term frequency based significance
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Wrap RandomForestClassifier in MultiOutputClassifier
        ])
 
    parameters = {'vect__min_df': [1, 5],
                #'tfidf__use_idf': [True, False],
                 'clf__estimator__n_estimators': [10, 25]
                #'clf__estimator__min_samples_split': [2, 4]
                 }
                  
    cv = GridSearchCV(pipeline, parameters, verbose=2)
    return cv

def get_results(y_test, y_pred):
    '''
    Function: 
        Return aggregate result scores for model
        Credit: isakkabir https://github.com/isakkabir/Disaster-Response-ML-Pipeline/blob/master/ML%20Pipeline%20Preparation.ipynb
    Args:
        y_test, y_pred: standard dataframes to evaluate
    Output: 
        results: dataframe holding detailed results by column
    '''
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for cat in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', cat)
        results.set_value(num+1, 'f_score', f_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    return results

def evaluate_model(model, X_test, Y_test):
    '''
    Function: 
        Evaluate model performance by calling get_results function
    Args:
        model: model to be evaluated
        X_test: test dataframe training params
        Y_test: y_pred: test target variables
    Output: 
        results: detailed results by column
    '''
    y_pred = model.predict(X_test)
    results = get_results(Y_test, y_pred)
    # Optional: print detailed results by uncommenting the line below. 
    # Summary results already printed by get_results() function
    # print(results)
    return results


def save_model(model, model_filepath):
    """Save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()