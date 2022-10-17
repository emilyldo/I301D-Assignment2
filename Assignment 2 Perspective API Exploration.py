#!/usr/bin/env python
# coding: utf-8

# ##Perspective API Exploration##
# 
# Below, I will be analyzing a data dataset of Wikipedia comments made available by Jigsaw, a subsidiary of Google that created the Perspective tool. This dataset includes a unique comment id, the text of the comment, and a series of binary labels applied by human raters: "toxic," "severe_toxic," "obscene," "threat," "insult," and "identity_hate" + an appended "score" column, which represents the toxicity score assigned to the comment text by the live version of the Perspective API. The data is available under a CC0 license.
# 
# My hypothesis is that the Perspective API will make more mistakes in classifying comments as toxic if they contain more Internet slang acronyms, such as 'lol', 'lmao', 'lmfao', 'wth','wtf', 'jk', 'idk', 'smh', 'ikr', and 'tbh'.

# In[74]:


import pandas as pd
import time

df = pd.read_csv('/Users/emilydo/Downloads/labeled_and_scored_comments.csv')


# Below is the dataset, organized by toxicity score.

# In[75]:


df.sort_values(['score'])


# In[76]:


df.head()


# Then, I pulled the comments labeled as toxic or severe toxic by human labelers.

# In[77]:


df[(df['toxic']==1)|(df['severe_toxic']==1)].head()


# Below includes a function to make calls to the Perspective API to get the toxicity score.

# In[79]:


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


# Here, I am gauging to see what the threshold of toxicity could be, aka at what point is a score considered toxic. I tested different phrases to do so, including positive, neutral, and explicit phrases.

# In[80]:


get_toxicity_score("I love you")


# In[81]:


get_toxicity_score("thank you")


# In[82]:


get_toxicity_score("how are you")


# In[83]:


get_toxicity_score("ugly")


# In[84]:


get_toxicity_score("fat")


# In[85]:


get_toxicity_score("fuck")


# In[86]:


get_toxicity_score("hate")


# I have decided to make the threshold as 0.4, to account for how a word like "hate" scored 0.271 while "ugly" scored 0.636. Then, I am testing to see how well the Perspective API performs in marking toxic comments in general with the complete dataset, based off this threshold.

# In[15]:


threshold = 0.4

df['prediction'] = (df['score'] > threshold).astype(int)
df['prediction'].value_counts()


# It seems that the Perspective API marks more comments as toxic than not. I am interested to see the ratio of true positivies to false positives as well as true negatives to false negatives.

# In[17]:


from sklearn.metrics import confusion_matrix


# In[18]:


confusion_matrix(df['toxic'], df['prediction'])


# It seems like in general, the Perspective API has a greater ratio of true positives to fale positives, compared to true negatives to false negatives. 

# Now, I am going to pull comments from the dataset that contain Internet slang acronyms, specifically 'lol', 'lmao', 'lmfao', 'wth','wtf', 'jk', 'idk', 'smh', 'ikr', and 'tbh'.

# In[101]:


slang_df = df.loc[df.comment_text.str.contains(r'\b(?:lol|lmao|lmfao|jk|wth|wtf|idk|smh|ikr|tbh)\b')]


# I have created a new sample from the dataset, containing comments with the above Internet sland acronyms. Notedly, it has an average API toxicity score of 0.440, with a standard deviation of 0.302.

# In[68]:


slang_df.describe()


# In[69]:


slang_df.sort_values(['score'])


# Below, I pulled specifically comments that were labeled as toxic or severe toxic by humans.

# Below, I pulled specifically comments that were labeled as toxic by humans.

# In[70]:


slang_df[(slang_df['toxic']==1)|(slang_df['severe_toxic']==1)].head()


# Using the same threshold of 0.4, I am testing to see how well the Perspective API performs in marking toxic comments with the Internet slang acronyms.

# In[99]:


threshold = 0.4

slang_df['prediction'] = (slang_df['score'] > threshold).astype(int)
slang_df['prediction'].value_counts()


# It seems that the Perspective API marks almost just as many comments toxic as not toxic. I am interested to see the ratio of true positivies to false positives as well as true negatives to false negatives.

# In[100]:


confusion_matrix(slang_df['toxic'], slang_df['prediction'])


# Here, it is shown that the Perspective API falsely marked more comments as non-toxic, something I didn't see when testing the whole dataset. However, notedly I am working with a much smaller sample, which could be influencing these numbers. From this sample though, this proves my hypothesis to be true as the Perspective API performed worse with comments that had Internet slang acronynms. 
