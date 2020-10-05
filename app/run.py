import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df_categories = df.iloc[:,5:]
    categories_names = df_categories.columns
    categories_count = df_categories.sum()
    related_counts = df.related.value_counts().tolist()
  
    
    #col_names = [col.replace('_', ' ').title() for col in data.columns]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'height': 650,
                'margin': dict(
                    b = 200,
                    pad = 4
                    )
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_count
                )
            ],
            'layout': {
                'title': 'Message Category Counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Type"
                },
                'height': 650,
                'margin': dict(
                    b = 200,
                    pad = 4
                    )
            }
        },
        {
          'data': [
                Bar(
                    x=['Related', 'Not Related'],
                    y=related_counts
                )
            ],
            'layout': {
                'title': 'Message count - Related / Not Related to Disaster',
                 'height': 650,
                'margin': dict(
                  b = 200,
                  pad = 4
                  )      
            }
            
        }
            
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()