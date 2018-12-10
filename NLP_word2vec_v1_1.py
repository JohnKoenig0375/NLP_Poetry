#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: Deploy Word2vec on Processed Poetry Data
Project: NLP Final Project
Version: 1
Date: 10DEC2018
Author: John Koenig
Purpose: Deploy Word2vec and save word embeddings
Inputs: 
    poetry_lines_df.csv
    
Outputs:
    word2vec_input.csv
    NLP_poetry_word2vec.model
Notes:
    None

'''

#%%
#Import Necessary Libraries

import pandas as pd
from gensim.models import word2vec


#%%
#Load Project Data

project_data_dir = './data/'
output_dir = './output/'
viz_dir = output_dir + 'visualizations/'

#Load poetry_lines_df
poetry_lines_df_filename = 'poetry_lines_df.csv'
poetry_lines_df = pd.read_csv(project_data_dir + poetry_lines_df_filename)

#poetry_lines_df_columns = list(poetry_lines_df.columns)
#poetry_lines_df_head = poetry_lines_df.iloc[:100,:]

w2vec_raw = list(poetry_lines_df['poem_line_text'].str.lower().str.findall(r"[\w']+")) #Extract tokens

#Build stop words list
w2vec_raw_value_counts = pd.DataFrame(w2vec_raw).stack().value_counts()
stop_words = pd.Series(w2vec_raw_value_counts[w2vec_raw_value_counts > 200].index)

#Remove tokens that do not appear more than once
w2vec_input1 = [[token for token in poem_line if w2vec_raw_value_counts[token] > 1 and sum(stop_words.isin([token])) == 0] for poem_line in w2vec_raw]

#Remove lines with no valid tokens
poetry_word2vec_df = pd.concat([poetry_lines_df,pd.Series(w2vec_input1).rename('w2vec_input')],axis=1)
w2vec_input2 = pd.DataFrame(columns=list(poetry_word2vec_df.columns))

#Remove lines with no valid tokens
for index,row in poetry_word2vec_df.iterrows():
    if len(row['w2vec_input']) > 0:
        print('processing row = ' + str(index))
        w2vec_input2 = pd.concat([w2vec_input2,pd.DataFrame([row])], axis=0)

w2vec_input2.index = range(len(w2vec_input2))

#Save Word2vec Input File
w2vec_input_filename = 'word2vec_input.csv'
w2vec_input2.to_csv(output_dir + w2vec_input_filename,index=False)


#%%
#Train Word2vec using Gensim

w2vec_input_list = list(w2vec_input2['w2vec_input'])

#Train Word2vec model with 3 dimensions
gensim_model = word2vec.Word2Vec(w2vec_input_list, min_count=1, size=3)

#Save Word2vec Model
gensim_model.save(output_dir + 'NLP_poetry_word2vec.model')




