#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: Calculate Cosine Similarities and Visualize with Heatmaps
Project: NLP Final Project
Version: 1
Date: 10DEC2018
Author: John Koenig
Purpose: Create cosine similarity matrices and build heatmap vizualizations
Inputs: 
    word2vec_input.csv
    NLP_poetry_word2vec.model
Outputs:
    w2vec_df.csv
    poem_features_combined_df.csv
    poet_features_combined_df.csv
    poem_author_heatmap.jpg
    poem_type_heatmap.jpg
    poem_age_heatmap.jpg
    poet_type_heatmap.jpg
    poet_age_heatmap.jpg
Notes:
    None

'''

#%%
#Import Necessary Libraries

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns


#%%
#Load Project Data

project_data_dir = './data/'
output_dir = './output/'
viz_dir = output_dir + 'visualizations/'

#Load Word2Vec Input File
w2vec_input_filename = 'word2vec_input.csv'
w2vec_input_df = pd.read_csv(output_dir + w2vec_input_filename)
w2vec_input_df_columns = list(w2vec_input_df.columns)
w2vec_input_df_head = w2vec_input_df.iloc[:100,:]


#%%
#Prepare w2vec Tokens

w2vec_tokens_raw_series = w2vec_input_df['w2vec_input'].str.replace('[','')
w2vec_tokens_1_series = w2vec_tokens_raw_series.str.replace(']','')
w2vec_tokens_2_series = w2vec_tokens_1_series.str.replace('\'','')
w2vec_tokens_3_series = w2vec_tokens_2_series.str.replace('\s','')
w2vec_input_tokens_df = w2vec_tokens_3_series.str.split(',',expand=True)
w2vec_tokens_counts = w2vec_input_tokens_df.stack().value_counts().rename('count')
w2vec_input_list = list(w2vec_tokens_counts.index)

#Create Token Count Dictionary
w2vec_tokens_counts_dict = w2vec_tokens_counts.to_dict()


#%%
#Extract word embeddings

#Load gensim model
gensim_model = Word2Vec.load(output_dir + 'NLP_poetry_word2vec.model')
w2vec_labels = list(gensim_model.wv.vocab)
w2vec_vectors_df = pd.DataFrame(gensim_model[gensim_model.wv.vocab],index=w2vec_labels)


w2vec_word_counts = pd.Series(w2vec_vectors_df.index.map(w2vec_tokens_counts_dict),index=w2vec_labels)

w2vec_df = pd.concat([pd.Series(w2vec_labels,index=w2vec_labels),
                      w2vec_vectors_df,
                      w2vec_word_counts],axis=1)

w2vec_df.index = range(len(w2vec_df))

features_columns = ['feature1',
                    'feature2',
                    'feature3']

w2vec_df_columns = ['token'] + features_columns + ['count']
w2vec_df.columns = w2vec_df_columns

#Save w2vec_df
w2vec_df_filename = 'w2vec_df.csv'
w2vec_df.to_csv(output_dir + w2vec_df_filename,index=False)


#%%
#Calculate Poem Feature Vectors

unique_poems_list = list(w2vec_input_df['poem_index'].unique())

num_poems = len(unique_poems_list)

poem_features_columns = ['poem_feature1',
                         'poem_feature2',
                         'poem_feature3']

poem_features_df_columns = ['poem_feature1',
                            'poem_feature2',
                            'poem_feature3',
                            'poem_token_count']

poem_features_df = pd.DataFrame(columns=poem_features_df_columns)

poem_token_concat = pd.concat([w2vec_input_df,w2vec_input_tokens_df],axis=1)

#For each poem, stack all tokens and average word embeddings to get poem vector
for poem in range(num_poems):
    poem_tokens_stack = poem_token_concat[poem_token_concat['poem_index'] == poem].iloc[:,62:].stack()
    
    poem_lines_features_df = pd.DataFrame(gensim_model.wv[poem_tokens_stack],columns=poem_features_columns,index=poem_tokens_stack)
    poem_tokens_count = len(poem_lines_features_df)
    poem_features_avg_series = poem_lines_features_df.mean()
    poem_features_series = poem_features_avg_series.append(pd.Series(poem_tokens_count,index=['poem_token_count']))
    
    poem_features_df_tmp = pd.DataFrame([poem_features_series],columns=poem_features_df_columns)
    poem_features_df = pd.concat([poem_features_df,poem_features_df_tmp],axis=0)

poem_features_df.index = range(len(poem_features_df))

#Combine poem vectors with poetry combined
poem_features_dict = poem_features_df.to_dict(orient='index')
poem_features_df_mapped = pd.DataFrame(list(w2vec_input_df['poem_index'].map(poem_features_dict)))
poem_features_combined_df = pd.concat([w2vec_input_df,poem_features_df_mapped],axis=1)

#Save poem_features_combined_df
poem_features_combined_df_filename = 'poem_features_combined_df.csv'
poem_features_combined_df.to_csv(output_dir + poem_features_combined_df_filename,index=False)


#%%
#Calculate Poet Feature Vectors

#Load poem_author_index and reindex authors
poetry_authors_series_filename = 'poetry_authors_df.csv'
poetry_authors_df = pd.read_csv(project_data_dir + poetry_authors_series_filename)
poetry_authors_series = pd.Series(list(poetry_authors_df.iloc[:,0]),index=poetry_authors_df.iloc[:,0])
poetry_authors_dict = poetry_authors_series.to_dict()

poet_info_columns = ['poet_author_index','poet_author_name']

poet_metrics_columns = ['poet_author_age',
                        'poet_author_age_index',
                        'poet_author_type',
                        'poet_author_type_index',
                        'poet_num_poems',
                        'poet_avg_paras',
                        'poet_avg_lines',
                        'poet_avg_tokens',
                        'poet_avg_words',
                        'poet_avg_punc',
                        'poet_avg_chars']

poet_features_columns = ['poet_feature1',
                         'poet_feature2',
                         'poet_feature3']

poets_df_columns = poet_info_columns + poet_metrics_columns + poet_features_columns

poets_df = pd.DataFrame(columns=poets_df_columns)

#For poet, average poem vectors to get poet vector
for index,poet in poetry_authors_df.iterrows():
    
    #Get current poet's poem
    curr_poet_df = poem_features_combined_df[poem_features_combined_df['poem_author_index'] == poet['poem_author_index']]
    
    #Extract poem features
    poet_features_series = pd.Series(curr_poet_df[poem_features_columns].mean())
    poet_features_series.index = poet_features_columns
    
    #Filter to only unique poems by current poet
    curr_poet_poems_df = curr_poet_df.drop_duplicates('poem_index')   
    
    poet_author_index = poet['poem_author_index']
    poet_author_name = poet['author']
    poet_author_age = poet['age']
    poet_author_age_index = poet['age_index']
    poet_author_type = poet['type']
    poet_author_type_index = poet['type_index']  
    
    poet_num_poems = 0
    poet_avg_paras = 0
    poet_avg_lines = 0
    poet_avg_tokens = 0
    poet_avg_words = 0
    poet_avg_punc = 0
    poet_avg_chars = 0
    
    if len(curr_poet_poems_df) > 0:
        
        poet_num_poems = len(curr_poet_poems_df)
        poet_avg_paras = round((sum(curr_poet_poems_df['poem_num_para']) / poet_num_poems),2)
        poet_avg_lines = round((sum(curr_poet_poems_df['poem_num_lines']) / poet_num_poems),2)
        poet_avg_tokens = round((sum(curr_poet_poems_df['poem_num_tokens']) / poet_num_poems),2)
        poet_avg_words = round((sum(curr_poet_poems_df['poem_num_words']) / poet_num_poems),2)
        poet_avg_punc = round((sum(curr_poet_poems_df['poem_num_punc']) / poet_num_poems),2)
        poet_avg_chars = round((sum(curr_poet_poems_df['poem_num_chars']) / poet_num_poems),2)
        
        poet_author_age = curr_poet_poems_df.iloc[0,:]['poem_age']
        poet_author_age_index = curr_poet_poems_df.iloc[0,:]['poem_age_index']
        poet_author_type = curr_poet_poems_df.iloc[0,:]['poem_type']
        poet_author_type_index = curr_poet_poems_df.iloc[0,:]['poem_type_index']
        
    poet_df = pd.DataFrame([[poet_author_index,
                             poet_author_name,
                             poet_author_age,
                             poet_author_age_index,
                             poet_author_type,
                             poet_author_type_index,
                             poet_num_poems,
                             poet_avg_paras,
                             poet_avg_lines,
                             poet_avg_tokens,
                             poet_avg_words,
                             poet_avg_punc,
                             poet_avg_chars,
                             poet_features_series['poet_feature1'],
                             poet_features_series['poet_feature2'],
                             poet_features_series['poet_feature3']]],columns=poets_df_columns)

    poets_df = pd.concat([poets_df,poet_df],axis=0)

poets_df.index = range(len(poets_df))

#Map poet features to full dataset and concatenate
poets_dict = poets_df.to_dict(orient='index')
poet_features_df_map = pd.DataFrame(list(poem_features_combined_df['poem_author_index'].map(poets_dict)))

poet_features_combined_df = pd.concat([poem_features_combined_df,poet_features_df_map],axis=1)
poet_features_combined_df_columns = list(poet_features_combined_df.columns)

#Save poet_features_combined_df
poet_features_combined_df_filename = 'poet_features_combined_df.csv'
poet_features_combined_df.to_csv(output_dir + poet_features_combined_df_filename,index=False)


#%%
#Calculate Poem Similarities by Poet, Poem Type, Poem Age

poems_unique_df = poem_features_combined_df.drop_duplicates('poem_index')
poems_unique_df.index = poems_unique_df['poem_index']
poems_unique_df_columns = list(poems_unique_df.columns)
poems_unique_df.index = range(len(poems_unique_df))

poems_heatmap_df = pd.concat([poems_unique_df.iloc[:,8:35],
                             poems_unique_df.iloc[:,36:51],
                             poems_unique_df.iloc[:,62:65]],axis=1)

poems_heatmap_df_columns = list(poems_heatmap_df.columns)

poems_cosine_sim = pd.DataFrame(cosine_similarity(poems_heatmap_df))

#Calculate Mean Similarity

similarities = []

for index,row in poems_cosine_sim.iterrows():
    values_tmp = list(row[(index + 1):])
    similarities = similarities + values_tmp

sim_mean = pd.Series(similarities).mean()

#Replace 1's along diagonal with similarity mean
for i in range(len(poems_cosine_sim)):
    poems_cosine_sim.iat[i,i] = sim_mean

poems_cosine_df = pd.concat([poems_unique_df,poems_cosine_sim],axis=1)

#Sort by Author
poems_sim_df_author = poems_cosine_df.sort_values(['poem_author','poem_name'])
poems_sim_df_author.index = range(len(poems_sim_df_author))

#Sort by Type
poems_sim_df_type = poems_cosine_df.sort_values(['poem_type_index','poem_feature1','poem_feature2','poem_feature3'])
poems_sim_df_type.index = range(len(poems_sim_df_type))

#Sort by Age
poems_sim_df_age = poems_cosine_df.sort_values(['poem_age_index','poem_feature1','poem_feature2','poem_feature3'])
poems_sim_df_age.index = range(len(poems_sim_df_age))

# Create Poem Cosine Heatmaps - Sort by Author, Type, Age
    
fig_size = (18,18)
dpi = 300
sns.set(font_scale=1)

# Generate a mask for the upper triangle
mask = np.zeros_like(poems_cosine_sim.values, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#Author Sorted Heatmap
poem_author_heatmap = sns.heatmap(poems_sim_df_author.iloc[:,len(poems_unique_df_columns):],
                                  cmap='Reds',
                                  xticklabels=False,
                                  yticklabels=False,
                                  mask=mask)

plt.tight_layout()

poem_author_heatmap_filename = 'poem_author_heatmap.jpg'
poem_author_heatmap.get_figure().savefig(viz_dir + poem_author_heatmap_filename, dpi=dpi, figsize=fig_size)

plt.close()


#Type Sorted Heatmap
poem_type_heatmap = sns.heatmap(poems_sim_df_type.iloc[:,len(poems_unique_df_columns):],
                                  cmap='Greens',
                                  xticklabels=False,
                                  yticklabels=False,
                                  mask=mask)

plt.tight_layout()

poem_type_heatmap_filename = 'poem_type_heatmap.jpg'
poem_type_heatmap.get_figure().savefig(viz_dir + poem_type_heatmap_filename, dpi=dpi, figsize=fig_size)

plt.close()


#Age Sorted Heatmap
poem_age_heatmap = sns.heatmap(poems_sim_df_age.iloc[:,len(poems_unique_df_columns):],
                                  cmap='Blues',
                                  xticklabels=False,
                                  yticklabels=False,
                                  mask=mask)

plt.tight_layout()

poem_age_heatmap_filename = 'poem_age_heatmap.jpg'
poem_age_heatmap.get_figure().savefig(viz_dir + poem_age_heatmap_filename, dpi=dpi, figsize=fig_size)

plt.close()


#TODO - Layer these heatmaps so the colors match the categories



#%%
#Calculate Poet Similarities by Poet Type and Poet Age

poets_unique_df = poet_features_combined_df.drop_duplicates('poem_author_index')
poets_unique_df.index = poets_unique_df['poem_author_index']
poets_unique_df_columns = list(poets_unique_df.columns)
poets_unique_df.index = range(len(poets_unique_df))

poets_heatmap_df = poets_unique_df.iloc[:,72:]
poets_cosine_sim = pd.DataFrame(cosine_similarity(poets_heatmap_df))

#Calculate Mean Similarity

similarities = []

for index,row in poets_cosine_sim.iterrows():
    values_tmp = list(row[(index + 1):])
    similarities = similarities + values_tmp

sim_mean = pd.Series(similarities).mean()

#Replace 1's along diagonal with similarity mean
for i in range(len(poets_cosine_sim)):
    poets_cosine_sim.iat[i,i] = sim_mean

poets_cosine_df = pd.concat([poets_unique_df,poets_cosine_sim],axis=1)

#Sort by Type
poets_sim_df_type = poets_cosine_df.sort_values(['poet_author_type_index','poet_feature1','poet_feature2','poet_feature3'])
poets_sim_df_type.index = range(len(poets_sim_df_type))

#Sort by Age
poets_sim_df_age = poets_cosine_df.sort_values(['poet_author_age_index','poet_feature1','poet_feature2','poet_feature3'])
poets_sim_df_age.index = range(len(poets_sim_df_age))

# Create Poet Cosine Heatmaps - Sort by Type and Age
    
fig_size = (36,18)
dpi = 300
sns.set(font_scale=0.5)

# Generate a mask for the upper triangle
mask = np.zeros_like(poets_cosine_sim.values, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


#Type Sorted Heatmap
poem_type_heatmap = sns.heatmap(poets_sim_df_type.iloc[:,len(poets_unique_df_columns):],
                                  cmap='Greens',
                                  xticklabels=False,
                                  yticklabels=poets_sim_df_type['poet_author_name'],
                                  mask=mask)

plt.tight_layout()

poem_type_heatmap_filename = 'poet_type_heatmap.jpg'
poem_type_heatmap.get_figure().savefig(viz_dir + poem_type_heatmap_filename, dpi=dpi, figsize=fig_size)

plt.close()

#Age Sorted Heatmap
poem_age_heatmap = sns.heatmap(poets_sim_df_age.iloc[:,len(poets_unique_df_columns):],
                                  cmap='Blues',
                                  xticklabels=False,
                                  yticklabels=poets_sim_df_age['poet_author_name'],
                                  mask=mask)

plt.tight_layout()

poem_age_heatmap_filename = 'poet_age_heatmap.jpg'
poem_age_heatmap.get_figure().savefig(viz_dir + poem_age_heatmap_filename, dpi=dpi, figsize=fig_size)

plt.close()



