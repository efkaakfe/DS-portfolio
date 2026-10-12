
import re
import string
import pandas as pd
import nltk
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')



def normalize_text(text:str)->str:
    '''
    Normalization of given text
    Returns cleaned text
    '''

    # Convert text to lowercase
    text = text.lower()
    # Replace special symbols between words by space
    text = re.sub(r'(\w)[.\-\/]+(\w)', r'\1 \2', text)

    # Replace HTML syntax by space
    text = re.sub(r'<br \/>', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    return text


def remove_stopwords(text:str, stops:list)->str:
    '''
    Removal of stop words
    text: text to process
    stops: stopwords list
    Returns text without stop words
    '''
    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    filtered_words = [word for word in words if word not in stops]

    return " ".join(filtered_words)


def process_words(text:str, lemmatizer:WordNetLemmatizer)->str:
    '''
    Lemmatization of given text 
    text: text to lemmatize
    lemmatizer: chosen lemmatizer
    Returns lemmatized text
    '''
    # Tokenize the text
    words = word_tokenize(text)
    # Apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(lemmatized_words)


# Defining function to calculate word frequencies

def freq(data:pd.DataFrame)->pd.DataFrame:
    '''
    Calculates frequency of each word in the dataset
    data: dataset to process
    Returns dataframe with columns representing words and their frequencies
    '''

    all_words = ' '.join(data).split()

    word_counts = Counter(all_words)

    word_freq_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])

    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

    return word_freq_df

# Normalization transformer


class NormalizationTransformer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer to normalize texts
    Returns transformed dataframe
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('Normalization')
        Xtr = X.apply(lambda x: normalize_text(x))

        return Xtr


# stopwords removal transformer

class StopremovalTransformer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer to remove stopwords
    Returns transformed dataframe
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('Stop words removal')
        stop_words = set(stopwords.words('english'))

        Xtr = X.apply(lambda x: remove_stopwords(x, stop_words))

        return Xtr

# Lemmatizer Transfomer


class LemmaTransformer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer to perform lemmatization
    Returns transformed dataframe
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('Lemmatization')
        lemmatizer = WordNetLemmatizer()

        Xtr = X.apply(lambda x: process_words(x, lemmatizer))
        print('Completed')
        return Xtr