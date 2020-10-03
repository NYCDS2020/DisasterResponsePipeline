import sys
import numpy as np
import pandas as pd
import chardet

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData,  select, func, Integer, Table, Column



def load_data(messages_filepath, categories_filepath):
    '''
    Function: 
        Load data from two CSV file paths, combine, and return as Pandas dataframe
    Args:
        messages_filepath: CSV file containing messages
        categories_filepath: CSV file containing categories
    Output: 
        Pandas dataframe
    '''
    
    # Load messages dataset with encoding-aware method
    with open(messages_filepath, 'rb') as file:
        encoding = chardet.detect(file.read())['encoding']
    messages =  pd.read_csv(messages_filepath, encoding=encoding)
    # load categories dataset - assuming same encoding for convenience
    categories = pd.read_csv(categories_filepath, encoding=encoding)
    # merge datasets
    df = messages.merge(categories, on='id')
    # return finished dataframe 
    return df


def clean_data(df):
    '''
    Function: 
        Prepare dataframe for machine learning operations. Various cleanup steps, expansion of categories, etc.
    Args:
        df: Pandas dataframe containing 'raw' information
    Output: 
        df: Pandas dataframe in ML-ready state
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # Extract column names from row entries. Using split() for simplicity
    category_colnames = list(row.apply(lambda x: x.split('-')[0]))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
    
def save_data(df, database_filename):
    '''
    Function: 
        Save data to database
    Args:
        database_filename: Name of database we want to write df to 
    Output: 
        None - other than print() statements letting user know what's happening with to_sql process
    '''
    table_name = 'DisasterResponseData' #ready made for later variablization as function argument
    engine = create_engine('sqlite:///'+ database_filename)
     # Only insert if table doesn't already exist. 
     # If table exists, log the info about the table and replace if less rows
    try:
        df.to_sql(table_name, engine, index=False)
        print("Successfully wrote table named {} to database with {} rows.".format(table_name, df.shape[0]))
    except ValueError:
        # Table exists. Let's investigate it
        existing_df = pd.read_sql('SELECT id FROM ' + table_name, engine)
        if existing_df.shape[0] < df.shape[0]:  # Is the existing data set smaller than our current dataframe?
            df.to_sql(table_name, engine, index=False, if_exists='replace') # Yes, smaller. So replace and log
            print("Found existing table named {} in database with {} rows - {} less rows than new dataframe. \
               Existing table was replaced.".format(table_name, existing_df.shape[0], df.shape[0]-existing_df.shape[0]))
        elif existing_df.shape[0]  == df.shape[0]: # Same number of rows. Let's drop table and re-insert
            df.to_sql(table_name, engine, index=False, if_exists='replace') # Replace and log
            print("Found existing table named {} in database with the same number of rows as the new dataframe. \
               Existing table was replaced.".format(table_name))
        elif existing_df.shape[0] > df.shape[0]: # More rows. Let's leave existing table alone
            print("Found existing table named {} in database with a higher number of rows ({}) than the new dataframe. \
                Existing table was left in place and database was not updated.".format(table_name, existing_df.shape[0]))

  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()