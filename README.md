# Disaster Response Pipeline
This project was completed as part of the Udacity Data Science Nanodegree requirement [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

## Required libraries
- nltk 
- numpy
- pandas
- scikit-learn 
- sqlalchemy 
- plotly

## Motivation
During times of natural or man-made disasters, a flood of messages inundates first responders, governmental agencies and other organizations. The abililty to quickly distinguish between types of messages and then route them to the appropriate organization for action is a key element in successful disaster response. 
NLP and ML classifiers are uniquely suited to handling large spikes in message volumes. 
This project uses a pre-labeled data set generously provided by [Figure Eight](https://www.figure-eight.com/) that contains labeled disaster messages. 

## Implementation
### Summary 
A multi-output Random Forest classifier was trained using a supervised learning pipeline containing normalization steps (vectorizer, TF/IDF) and a train/fit step. The classifier pickle file is loaded into a web app to deliver user results. 

### Detail
Initially, an ETL pipeline extracts data from CSV format, cleans and merges data from several files, and loads this data into a SQL database. From there, a machine learning pipeline picks up the data, uses a count vectorizer to vectorize textual data, a TF/IDF pass to normalize for occurrence, and a Random Forest classifier to create a multi-output classification model. This model is then used in a Bootstrap/Flask based web app.

The end result is a web app that allows users to a) observe some visual insights into the training data set and b) classify their own messages. 


## Project Screenshots
Users can interactively classify a message

!['Enter Message'](readme_imgs/001_input.jpg)

The app then displays the classification results.

!['Example Output'](readme_imgs/001a_input.jpg)

During startup of the web app, it displays summary stats of the training data set
!['Message Types'](readme_imgs/002_visual.jpg)
!['Category Counts'](readme_imgs/003_visual.jpg)


## Acknowledgements
isakkabir for result summarization code for classifier training passes [here](https://github.com/isakkabir/Disaster-Response-ML-Pipeline/blob/master/ML%20Pipeline%20Preparation.ipynb)
Figure Eight for the labelled training data set [here](https://www.figure-eight.com/)