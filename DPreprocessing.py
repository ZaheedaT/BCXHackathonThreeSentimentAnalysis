#Data Preprocessing
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import nltk
import re
from nltk import word_tokenize
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
import string
from sklearn.externals import joblib
from textstat.textstat import textstat
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import (TfidfVectorizer,
                                             TfidfTransformer,
                                             CountVectorizer)

# Classifiers and ML
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import (FunctionTransformer,
                                   StandardScaler,
                                   label_binarize)
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     cross_val_score)
from sklearn.ensemble import (
                              RandomForestClassifier)

from mlxtend.preprocessing import DenseTransformer

def preprocess(stringss):
    """Preprocess text for Natural Language Processing.

    This function preprocesses text data for Natural Language processing
    by removing punctuation, except emotive punctuation ('!', '?').

    Parameters:
    -----------
    stringss -- Input string which is to be processed.

    Returns:
    -----------
    String of which the punctuation has been removed.
    """

    import string
    #keep_punc = ['?', '!']
    punctuation = [str(i) for i in string.punctuation]
    #punctuation = [punc for punc in punctuation if punc not in keep_punc]
    s = ''.join([punc for punc in stringss if punc not in punctuation])
    return s




vectorizer = TfidfVectorizer(preprocessor=preprocess,
                             stop_words=None,
                             tokenizer=TweetTokenizer().tokenize
                             )
