import requests
import unicodedata
from bs4 import BeautifulSoup
import pandas as pd
from time import strftime
import json
from typing import Dict, List, Optional, Union, cast
import requests
import time
import re

from env import github_token, github_username

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


###### Functions ######

# Function to get the list of repos from Microsoft's GitHub page: 
def get_repo_links():
    """
    Function to get all of the repo urls on the Microsoft GitHub page
    Creates csv file containing the list of repos to be scrapped
    """
    repo_names = []
    base_url = 'https://github.com/orgs/microsoft/repositories?page='
    for i in range(1, 51):
        response = requests.get(base_url + str(i), headers={"user-agent": "Codeup DS"})
        soup = BeautifulSoup(response.text, features="lxml")
        links = [a.attrs["href"] for a in soup.select("a") if 'data-hovercard-type' in a.attrs]
        for link in links:
            repo_names.append(link)
        time.sleep(1) 
    repo_names = pd.read_csv('microsoft_repo_list.csv')
    return repo_names


def acquire_df():
    """
    Returns the pre-existing json file of the web-scrapped repos into a pandas dataframe
    """
    return pd.read_json('data.json')


def basic_clean(string):
    """
    This function takes in a string and
    returns the string normalized.
    """
    # converts string characters to lower case
    article = string.lower()
    # removes accented characters
    article = unicodedata.normalize('NFKD', article).encode('ascii', 'ignore').decode('utf-8')
    #article = re.sub(r"[^\w]", ' ', article)
    # removes any special characters
    article = re.sub("[^a-z0-9'\s]", '', article)
    article = re.sub("\\n", ' ', article)

    return article


def basic_clean2(string):
    """
    This function takes string, normalizes the string, and removes apostrophes
    """
    article = string.lower()
    article = unicodedata.normalize('NFKD', article).encode('ascii', 'ignore').decode('utf-8')
    #article = re.sub(r"[^\w]", ' ', article)
    article = re.sub("[^a-z0-9\s]", '', article)
    article = re.sub("\\n", ' ', article)

    return article


def tokenize(string):
    """
    Function takes in a string and 
    breaks words and any punctuation into discrete units.
    """
    article = []
    # Create the tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()

    # Use the tokenizer
    article = tokenizer.tokenize(string, return_str = True)

    return article


def lemmatize(text):
    """
    Function takes in a string and returns the root word for each element
    """
    # Create the Lemmatizer.
    wnl = nltk.stem.WordNetLemmatizer()
    # use the lemmatizer
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    # Join the lemmatized words together
    article_lemmatized = ' '.join(lemmas)
    
    return article_lemmatized


def remove_stopwords(string, extra_words=[], exclude_words=[]):  
    """
    Function takes in a string and lists of additional words to include or exlclude and
    drops any insignificant words deemed insignifcant
    """
    stopword_list = stopwords.words('english')
    # Remove 'exclude_words' from stopword_list to keep these in my text.
    stopword_list = set(stopword_list) - set(exclude_words)
    # Add in 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))
    # Split words in lemmatized article.
    words = string.split()
    # Iterate through the words and only keep those not in stopword_list
    filtered_words = [word for word in words if word not in stopword_list]
    # Join words in the list back into strings; assign to a variable to keep changes.
    article_without_stopwords = ' '.join(filtered_words)

    return article_without_stopwords


def prep_df(df):
    """
    Function takes in a dataframe and:
        1. Drops rows with null values in 'language' column
        2. Resets index
        3. Creates a series consisting of normalized string values and combines series with dataframe
        4. Creates normalized, lemmatized strings with no stopwords from 'clean' column
        5. Drop rows that have non-null. empty values for the content columns and resets index
        6. Creates word count and character count columns 
        7. Create target column that shows whether repo language is TypeScript or not
        8. Drops original readme contents column
    """
    # 1
    df.dropna(subset=['language'], inplace=True)
    # 2
    df = df.reset_index(drop=True)
    # 3
    cleaned = [basic_clean2(tokenize(basic_clean(BeautifulSoup(readme_contents, 'html.parser').text))) for readme_contents in df.readme_contents]
    df['clean'] = pd.Series(cleaned)  
    # 4
    df['lemma_no_stopwords'] = df.clean.apply(lemmatize).apply(remove_stopwords, extra_words=['build', 'file', 'example', 'code', 'use', 'using'], exclude_words=[])
    # 5
    df.drop(index=df.loc[df['lemma_no_stopwords'] == df['clean']].index, inplace=True)
    df = df.reset_index(drop=True)
    # 6
    df['clean_word_count'] = df.clean.str.split().str.len()
    df['readme_char_count'] = df.clean.str.len()
    # 7
    df['is_TypeScript'] = df.language == 'TypeScript'
    # 8
    df = df[["repo", "language", "clean", "lemma_no_stopwords", "clean_word_count", "readme_char_count", "is_TypeScript"]]
    
    return df


def wrangle():
    """
    Function creates a dataframe from web-scrapped repos and calls the prep_data() function to run the acquired dataframe through the preparation steps
    """
    # acquire the function
    df = acquire_df()
    
    # prepares and cleans the dataframe
    df = prep_df(df)
    
    return df