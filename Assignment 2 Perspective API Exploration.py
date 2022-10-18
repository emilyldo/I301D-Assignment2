#!/usr/bin/env python
# coding: utf-8

# ##Perspective API Exploration##
# 
# Below, I will be analyzing a dataset of Wikipedia comments made available by Jigsaw, a subsidiary of Google that created the Perspective tool. This dataset includes a unique comment id, the text of the comment, and a series of binary labels applied by human raters: "toxic," "severe_toxic," "obscene," "threat," "insult," and "identity_hate" + an appended "score" column, which represents the toxicity score assigned to the comment text by the live version of the Perspective API. The data is available under a CC0 license.
# 
# My hypothesis is that the Perspective API will make more mistakes in classifying comments as toxic if they contain more Internet slang acronyms, such as 'lol', 'lmao', 'lmfao', 'wth','wtf', 'jk', 'idk', 'smh', 'ikr', and 'tbh'.
import pandas as pd
import time
df = pd.read_csv('/Users/emilydo/Downloads/labeled_and_scored_comments.csv')
df.sort_values(['score'])
df.head()
df[(df['toxic']==1)|(df['severe_toxic']==1)].head()
from googleapiclient.discovery import build
import json
def get_toxicity_score(comment):
    
  API_KEY = 'XXXX' # Put your API key here
    
  client = build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
  )

  analyze_request = {
  'comment': { 'text': comment },
  'requestedAttributes': {'TOXICITY': {}}
  }
    
  response = client.comments().analyze(body=analyze_request).execute()
  toxicity_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
    
  return toxicity_score
get_toxicity_score("I love you")
get_toxicity_score("thank you")
get_toxicity_score("how are you")
get_toxicity_score("ugly")
get_toxicity_score("fat")
get_toxicity_score("fuck")
get_toxicity_score("hate")
threshold = 0.4
df['prediction'] = (df['score'] > threshold).astype(int)
df['prediction'].value_counts()
from sklearn.metrics import confusion_matrix
confusion_matrix(df['toxic'], df['prediction'])
slang_df = df.loc[df.comment_text.str.contains(r'\b(?:lol|lmao|lmfao|jk|wth|wtf|idk|smh|ikr|tbh)\b')]
slang_df.describe()
slang_df.sort_values(['score'])
slang_df[(slang_df['toxic']==1)|(slang_df['severe_toxic']==1)].head()
threshold = 0.4
slang_df['prediction'] = (slang_df['score'] > threshold).astype(int)
slang_df['prediction'].value_counts()
confusion_matrix(slang_df['toxic'], slang_df['prediction'])

