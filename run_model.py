
# coding: utf-8

# In[ ]:

from argparse import ArgumentParser
from sklearn.externals import joblib
import pandas as pd
from DPreprocessing import *
#import english

parser = ArgumentParser()
parser.add_argument('--input', help='Path to the input CSV')
parser.add_argument('--submission', help='Path to the submission CSV')
args = parser.parse_args()

features = pd.read_csv(args.input)
fit_pipe = joblib.load('fit_pipe2.pkl')
prediction = features['text']
prediction22 = fit_pipe.transform(prediction)

model_10 = joblib.load('model_104.pkl')
preds= model_10.predict_proba(prediction22)
new = pd.DataFrame(preds)
new.rename(columns={0:'stars_1', 1:'stars_2', 2:'stars_3',3:'stars_4', 4:'stars_5' }, inplace=True)
submit = prediction.copy()
df = pd.DataFrame(features['review_id'])
df2= df.join(new)
#submit.drop('text',  inplace=True, axis=1,)
#submit.join(new)
df2.to_csv(args.submission, index=False)

