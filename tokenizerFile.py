# Here we create a tokenizer class to make the data processable.
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import contractions

class Tokenizer:
    def tokenize_dataframe(self, data:pd.DataFrame):
        # tokenize all the titles and texts
        data['title'] = data['title'].apply(self._tokenize_datapoint)

        data['Text'] = data['Text'].apply(self._tokenize_datapoint)

        return data

    def _tokenize_datapoint(self, text: str):
        # expand contractions like "it's" to "it" and "is"
        text_expanded = contractions.fix(text)

        # tokanize the words
        tokens = word_tokenize(text_expanded)

        # remove punctuation
        tokens = [word for word in tokens if word.isalnum()]

        # make them lowercase
        tokens = [word.lower() for word in tokens]

        return tokens