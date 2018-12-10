#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: Process English Language Poetry with NLP
Project: NLP Final Project
Version: 1
Date: 10DEC2018
Author: John Koenig
Purpose: Clean and process raw poetry data, extract metrics by poem line
Inputs: 
    poetry.csv
    
Outputs:
    poetry_lines_df.csv
    poetry_descriptives_df.csv
Notes:
    None

'''


#%%
#Import Necessary Libraries

import pandas as pd

import spacy
nlp = spacy.load('en_core_web_lg')
from spacy.attrs import IS_PUNCT


#%%
#Load Project Data

project_data_dir = './data/'
output_dir = './output/'
viz_dir = output_dir + 'visualizations/'

#Raw Poetry File
poetry_raw_df = pd.read_csv(project_data_dir + 'poetry.csv')

#Clean Poetry Data
poetry_raw_df = poetry_raw_df.rename(index=str, columns={"poem name": "poem_name"})
poetry_raw_df.drop_duplicates(['author','poem_name'],inplace=True)
poetry_raw_df['content'] = poetry_raw_df['content'].str.replace('[0-9\'\"\)\(]','')
poetry_raw_df['content'] = poetry_raw_df['content'].str.split('\n')
poetry_raw_df['age_index'] = poetry_raw_df['age'].astype('category').cat.codes
poetry_raw_df['type_index'] = poetry_raw_df['type'].astype('category').cat.codes
poetry_raw_df.index = range(len(poetry_raw_df))

#poetry_raw_columns = list(poetry_raw_df.columns)
#poetry_raw_df_head = poetry_raw_df.iloc[:100,:]

#Create Poem Authors Index and Dictionary
poetry_authors_df = poetry_raw_df.drop_duplicates('author')
poetry_authors_df.index = range(len(poetry_authors_df))
poetry_authors_df.insert(0,'poem_author_index',list(poetry_authors_df.index))
poetry_authors_df.index = poetry_authors_df['author']
poetry_authors_df.drop(['content','poem_name'],axis=1,inplace=True)
poetry_authors_dict = poetry_authors_df.iloc[:,0].to_dict()
poetry_raw_df['poem_author_index'] = poetry_raw_df['author'].map(poetry_authors_dict)

#Save poem_author_index
poetry_authors_df_filename = 'poetry_authors_df.csv'
poetry_authors_df.to_csv(project_data_dir + poetry_authors_df_filename, index=False)


#%%
#Set Project Columns

poem_summary_columns = ['poem_name',
                        'poem_index',
                        'poem_author',
                        'poem_author_index',
                        'poem_age',
                        'poem_age_index',
                        'poem_type',
                        'poem_type_index',
                        'poem_num_para',
                        'poem_num_lines',
                        'poem_num_tokens',
                        'poem_num_words',
                        'poem_num_punc',
                        'poem_num_chars',
                        'poem_longest_para_lines',
                        'poem_shortest_para_lines',
                        'poem_longest_para_words',
                        'poem_shortest_para_words',
                        'poem_longest_para_chars',
                        'poem_shortest_para_chars',
                        'poem_longest_line_words',
                        'poem_shortest_line_words',
                        'poem_longest_line_chars',
                        'poem_shortest_line_chars',
                        'poem_longest_word',
                        'poem_shortest_word',
                        'poem_avg_lines_per_para',
                        'poem_avg_tokens_per_para',
                        'poem_avg_words_per_para',
                        'poem_avg_punc_per_para',
                        'poem_avg_chars_per_para',
                        'poem_avg_tokens_per_line',
                        'poem_avg_words_per_line',
                        'poem_avg_punc_per_line',
                        'poem_avg_chars_per_line']

poem_para_columns = ['poem_para_index',
                     'poem_para_lines',
                     'poem_para_tokens',
                     'poem_para_words',
                     'poem_para_punc',
                     'poem_para_chars',
                     'poem_para_longest_line_words',
                     'poem_para_shortest_line_words',
                     'poem_para_longest_line_chars',
                     'poem_para_shortest_line_chars',
                     'poem_para_longest_word',
                     'poem_para_shortest_word',
                     'poem_para_avg_tokens_per_line',
                     'poem_para_avg_words_per_line',
                     'poem_para_avg_punc_per_line',
                     'poem_para_avg_chars_per_line']

poem_line_columns = ['poem_line_index',
                     'poem_para_line_index',
                     'poem_line_text',
                     'poem_line_tokens',
                     'poem_line_words',
                     'poem_line_punc',
                     'poem_line_chars',
                     'poem_line_avg_chars_per_word',
                     'poem_line_longest_word',
                     'poem_line_shortest_word']

poem_df_columns = poem_para_columns + poem_line_columns
all_poems_columns = poem_summary_columns + poem_para_columns + poem_line_columns


#%%
#Process Poetry by Line

poetry_lines_df = pd.DataFrame(columns=all_poems_columns)

poem_index = 0

# For poem in corpus
for index,p in poetry_raw_df.iterrows():
    poem_content = pd.Series(p['content']).replace(to_replace='*\r*',value='').str.strip()
    poem_author = p['author']
    poem_author_index = p['poem_author_index']
    poem_name = p['poem_name']
    poem_age = p['age']
    poem_age_index = p['age_index']
    poem_type = p['type']
    poem_type_index = p['type_index']
    
    if len(poem_content) == 1:   # Only one line (not a full poem)
        print('poem_index = %s --- poem_name = %s --- SKIPPED - NO POEM CONTENT FOUND' % (str(index),poem_name))
        continue   # Skip to full poem 
        
    if pd.isna(poem_name) == True:   # If poem title is missing
        poem_name = 'Index = ' + str(index)    

    poem_para_index = 0
    poem_line_index = 0
    poem_para_line_index = 0
    
    poem_num_para = 0
    poem_num_lines = 0
    poem_num_tokens = 0
    poem_num_words = 0
    poem_num_punc = 0
    poem_num_chars = 0
    
    poem_para_lines = 0
    poem_para_tokens = 0
    poem_para_words = 0
    poem_para_punc = 0
    poem_para_chars = 0
    
    poem_longest_para_lines = 0
    poem_shortest_para_lines = 1000
    poem_longest_para_words = 0
    poem_shortest_para_words = 1000
    poem_longest_para_chars = 0
    poem_shortest_para_chars = 10000
    poem_longest_line_words = 0
    poem_shortest_line_words = 1000
    poem_longest_line_chars = 0
    poem_shortest_line_chars = 10000
    poem_longest_word = 0
    poem_shortest_word = 100  # always ends up being 2
    
    print('poem_index = %s --- poem_name = %s --- poem raw lines = %s' % (str(index),poem_name,str(len(poem_content))))

    #Create poem_df_tmp
    poem_df_tmp = pd.DataFrame(columns=poem_df_columns)   # Stores all lines of current poem
    
    #Create poem_para_lines_tmp
    poem_para_lines_tmp = pd.DataFrame(columns=poem_line_columns) # Stores all lines in current poem paragraph
    
    poem_para_longest_line_words = 0
    poem_para_shortest_line_words = 100
    poem_para_longest_line_chars = 0
    poem_para_shortest_line_chars = 1000
    poem_para_longest_word = 0
    poem_para_shortest_word = 100
    
    for l in poem_content.iteritems():   # For each line in poem
        
        poem_line_text_tmp = l[1]   # Poem line text
        curr_line_index = l[0]   # Poem line index
        
        poem_line_doc_tmp = nlp(poem_line_text_tmp)   # Tokenize using spacy
        poem_line_tokens = len(poem_line_doc_tmp)   # Store number of tokens in poem line
        
        last_line = False   # Boolean indicating that the for loop is on the last line
        
        #Check if last line
        if curr_line_index == (len(poem_content) - 1):
            last_line = True
        
        #If not last line
        if last_line == False:
            poem_next_line_doc_tmp = nlp(poem_content[(curr_line_index + 1)])   # Tokenize next poem line using spacy
            poem_next_line_tokens = len(poem_next_line_doc_tmp)   # Store number of tokens on next line (to see if it is blank)
        else:
            poem_next_line_tokens = 2   # If this is the last line, set an arbitrary value that will let the script continue

        #If line is blank (or one token)
        if poem_line_tokens <= 1:   # If current poem line is empty
            
            #Check if first line in poem
            if curr_line_index == 0:   #If first line in poem
                continue   # Skip to first non-blank line in poem
            
            #Check if next line is blank (or one token)
            if poem_next_line_tokens <= 1:  # If next line is blank (or one token)
                continue   # Skip this blank line
                
            #Check if this is the first paragraph in the poem
            if poem_para_index == 0 and last_line == False and len(poem_para_lines_tmp) == 0:  # If this is the first paragraph in the poem and it is not the last line of the poem and there are no processed line queued in poem_para_lines_tmp
                continue   # Skip this blank line
            
            pass   # Skip to append paragraph
            
        else:   # If line is NOT blank (or one token)
            
            #Calculate metrics for current line of poem
            poem_line_punc = sum(poem_line_doc_tmp.to_array([IS_PUNCT]))
            poem_line_words = poem_line_tokens - poem_line_punc
            poem_line_token_lens = pd.Series(poem_line_doc_tmp).str.len()
            poem_line_longest_word = poem_line_token_lens.max()
            poem_line_shortest_word = poem_line_token_lens[poem_line_token_lens > 1].min()
            poem_line_chars = sum(poem_line_token_lens) - poem_line_punc
        
            if poem_line_words > 0:
                poem_line_avg_chars_per_word = round((poem_line_chars / poem_line_words),2)
            else:
                poem_line_avg_chars_per_word = 0
                
            #Create poem line dataframe
            poem_line_df_tmp = pd.DataFrame([[poem_line_index,
                                              poem_para_line_index,
                                              poem_line_text_tmp,
                                              poem_line_tokens,
                                              poem_line_words,
                                              poem_line_punc,
                                              poem_line_chars,
                                              poem_line_avg_chars_per_word,
                                              poem_line_longest_word,
                                              poem_line_shortest_word]],columns=poem_line_columns)
            
            #Append poem_line_df_tmp to poem_para_lines_tmp
            poem_para_lines_tmp = pd.concat([poem_para_lines_tmp,poem_line_df_tmp],axis=0)
            
            #Increment poem line indices
            poem_line_index += 1
            poem_para_line_index += 1
        
            #Increment paragraph numbers
            poem_para_lines += 1
            poem_para_tokens += poem_line_tokens
            poem_para_words += poem_line_words
            poem_para_punc += poem_line_punc
            poem_para_chars += poem_line_chars
            
            #Increment poem numbers
            poem_num_lines += 1
            poem_num_tokens += poem_line_tokens
            poem_num_words += poem_line_words
            poem_num_punc += poem_line_punc
            poem_num_chars += poem_line_chars
            
            #Update para longest/shortest word
            if poem_line_longest_word > poem_para_longest_word:
                poem_para_longest_word = poem_line_longest_word
            if poem_line_shortest_word < poem_para_shortest_word:
                poem_para_shortest_word = poem_line_shortest_word
            
            #Update para longest/shortest lines
            if poem_line_words > poem_para_longest_line_words:
                poem_para_longest_line_words = poem_line_words
            if poem_line_words < poem_para_shortest_line_words:
                poem_para_shortest_line_words = poem_line_words
            if poem_line_chars > poem_para_longest_line_chars:
                poem_para_longest_line_chars = poem_line_chars
            if poem_line_chars < poem_para_shortest_line_chars:
                poem_para_shortest_line_chars = poem_line_chars  
            
            #Update poem longest/shortest word
            if poem_line_longest_word > poem_longest_word:
                poem_longest_word = poem_line_longest_word
            if poem_line_shortest_word < poem_shortest_word:
                poem_shortest_word = poem_line_shortest_word
            
            #Update poem longest/shortest lines
            if poem_line_words > poem_longest_line_words:
                poem_longest_line_words = poem_line_words
            if poem_line_words < poem_shortest_line_words:
                poem_shortest_line_words = poem_line_words
            if poem_line_chars > poem_longest_line_chars:
                poem_longest_line_chars = poem_line_chars
            if poem_line_chars < poem_shortest_line_chars:
                poem_shortest_line_chars = poem_line_chars               
        
        #Create and append poem_para_metrics_tmp
        if poem_line_tokens <= 1 or last_line == True:  # If  line is blank OR if last line of poem

            if poem_para_index == 0:   # Handle 1 paragraph poems
                #Update poem totals
                poem_para_lines = len(poem_para_lines_tmp)
                poem_para_tokens = sum(poem_para_lines_tmp['poem_line_tokens'])
                poem_para_words = sum(poem_para_lines_tmp['poem_line_words'])
                poem_para_punc = sum(poem_para_lines_tmp['poem_line_punc'])
                poem_para_chars = sum(poem_para_lines_tmp['poem_line_chars'])
                
                #Update longest/shortest paragraphs
                poem_longest_para_lines = poem_num_lines
                poem_shortest_para_lines = poem_num_lines
                poem_longest_para_words = poem_para_words
                poem_shortest_para_words = poem_para_words
                poem_longest_para_chars = poem_num_chars
                poem_shortest_para_chars = poem_num_chars
            
            #Update poem longest/shortest lines counts
            if poem_para_lines > poem_longest_para_lines:
                poem_longest_para_lines = poem_num_lines
            if poem_para_lines < poem_shortest_para_lines:
                poem_shortest_para_lines = poem_num_lines
            
            #Update poem longest/shortest words counts
            if poem_para_words > poem_longest_para_words:
                poem_longest_para_words = poem_para_words
            if poem_para_words < poem_shortest_para_words:
                poem_shortest_para_words = poem_para_words
                
            #Update poem longest/shortest character counts
            if poem_para_chars > poem_longest_para_chars:
                poem_longest_para_chars = poem_num_chars
            if poem_para_chars < poem_shortest_para_chars:
                poem_shortest_para_chars = poem_num_chars

            #Calculate Paragraph Averages
            poem_para_avg_tokens_per_line = round((poem_para_tokens / poem_para_lines),2)
            poem_para_avg_words_per_line = round((poem_para_words / poem_para_lines),2)
            poem_para_avg_punc_per_line = round((poem_para_punc / poem_para_lines),2)
            poem_para_avg_chars_per_line = round((poem_para_chars / poem_para_lines),2)
            
            #Create poem_para_metrics_tmp
            poem_para_metrics_tmp = pd.DataFrame([[poem_para_index,
                                                   poem_para_lines,
                                                   poem_para_tokens,
                                                   poem_para_words,
                                                   poem_para_punc,
                                                   poem_para_chars,
                                                   poem_para_longest_line_words,
                                                   poem_para_shortest_line_words,
                                                   poem_para_longest_line_chars,
                                                   poem_para_shortest_line_chars,
                                                   poem_para_longest_word,
                                                   poem_para_shortest_word,
                                                   poem_para_avg_tokens_per_line,
                                                   poem_para_avg_words_per_line,
                                                   poem_para_avg_punc_per_line,
                                                   poem_para_avg_chars_per_line]],columns=poem_para_columns)
            
            #Copy poem_para_metrics_tmp to length of poem_para_lines_tmp by rows
            poem_para_metrics_tmp = pd.concat([poem_para_metrics_tmp]*len(poem_para_lines_tmp),axis=0)
            
            #Concatenate poem_para_metrics_tmp to poem_para_lines_tmp by columns
            poem_para_combined_df = pd.concat([poem_para_metrics_tmp,poem_para_lines_tmp],axis=1)
               
            #Concatenate poem_para_combined_df to poem_df_tmp by rows
            poem_df_tmp = pd.concat([poem_df_tmp,poem_para_combined_df],axis=0)
            
            #Iterate to next paragraph
            poem_num_para += 1
            poem_para_index += 1
            poem_para_line_index = 0
            
            #Reset paragraph metrics for new paragraph
            poem_para_lines = 0
            poem_para_tokens = 0
            poem_para_words = 0
            poem_para_punc = 0
            poem_para_chars = 0
            
            poem_para_longest_line_words = 0
            poem_para_shortest_line_words = 100
            poem_para_longest_line_chars = 0
            poem_para_shortest_line_chars = 1000
            poem_para_longest_word = 0
            poem_para_shortest_word = 100
        
            #Create blank poem_para_lines_tmp dataframe
            poem_para_lines_tmp = pd.DataFrame(columns=poem_line_columns)
    
    poem_df_tmp.index = range(len(poem_df_tmp))   # Reset poem_df_tmp index
    
    #Calculate poem averages
    poem_avg_lines_per_para = round((poem_num_lines / poem_num_para),2)
    poem_avg_tokens_per_para = round((poem_num_tokens / poem_num_para),2)
    poem_avg_words_per_para = round((poem_num_words / poem_num_para),2)
    poem_avg_punc_per_para = round((poem_num_punc / poem_num_para),2)
    poem_avg_chars_per_para = round((poem_num_chars / poem_num_para),2)
    poem_avg_tokens_per_line = round((poem_num_tokens / poem_num_lines),2)
    poem_avg_words_per_line = round((poem_num_words / poem_num_lines),2)
    poem_avg_punc_per_line = round((poem_num_punc / poem_num_lines),2)
    poem_avg_chars_per_line = round((poem_num_chars / poem_num_lines),2)
    
    #Create poem_metrics_df_tmp
    poem_metrics_df_tmp = pd.DataFrame([[poem_name,
                                         poem_index,
                                         poem_author,
                                         poem_author_index,
                                         poem_age,
                                         poem_age_index,
                                         poem_type,
                                         poem_type_index,
                                         poem_num_para,
                                         poem_num_lines,
                                         poem_num_tokens,
                                         poem_num_words,
                                         poem_num_punc,
                                         poem_num_chars,
                                         poem_longest_para_lines,
                                         poem_shortest_para_lines,
                                         poem_longest_para_words,
                                         poem_shortest_para_words,
                                         poem_longest_para_chars,
                                         poem_shortest_para_chars,
                                         poem_longest_line_words,
                                         poem_shortest_line_words,
                                         poem_longest_line_chars,
                                         poem_shortest_line_chars,
                                         poem_longest_word,
                                         poem_shortest_word,
                                         poem_avg_lines_per_para,
                                         poem_avg_tokens_per_para,
                                         poem_avg_words_per_para,
                                         poem_avg_punc_per_para,
                                         poem_avg_chars_per_para,
                                         poem_avg_tokens_per_line,
                                         poem_avg_words_per_line,
                                         poem_avg_punc_per_line,
                                         poem_avg_chars_per_line]],columns=poem_summary_columns)
    
    #Copy poem_metrics_df_tmp to length of poem_df_tmp
    poem_metrics_df_tmp = pd.concat([poem_metrics_df_tmp]*len(poem_df_tmp),axis=0)
    poem_metrics_df_tmp.index = range(len(poem_metrics_df_tmp))
    
    #Concatenate final_poem_df together
    final_poem_df = pd.concat([poem_metrics_df_tmp,poem_df_tmp],axis=1)
    
    #Append final_poem_df to poetry_lines_df
    poetry_lines_df = pd.concat([poetry_lines_df,final_poem_df], axis=0)
    
    poem_index += 1
    
    
poetry_lines_df.index = range(len(poetry_lines_df))   # Reset poetry_lines_df_tmp index

#Save poetry_lines_df
poetry_lines_df_filename = 'poetry_lines_df.csv'
poetry_lines_df.to_csv(project_data_dir + poetry_lines_df_filename,index=False)


#%%
#Get Poetry Descriptive Statistics
poetry_poems_df = poetry_lines_df.drop_duplicates('poem_index')

num_poems_sum = len(poetry_poems_df)
num_lines_sum = len(poetry_lines_df)
unique_authors = pd.Series(poetry_lines_df['poem_author'].unique())
num_authors_sum = len(unique_authors)
num_lines_author_sum = poetry_lines_df['poem_author'].value_counts()
num_lines_age_sum = poetry_lines_df['poem_age'].value_counts()
num_lines_type_sum = poetry_lines_df['poem_type'].value_counts()
num_poems_author_sum = poetry_poems_df['poem_author'].value_counts()
num_poems_age_sum = poetry_poems_df['poem_age'].value_counts()
num_poems_type_sum = poetry_poems_df['poem_type'].value_counts()

poetry_descriptives_df = poetry_lines_df.iloc[:,5:].drop('poem_line_text',axis=1).describe().T

#Save poetry_descriptives_df
poetry_descriptives_df_filename = 'poetry_descriptives_df.csv'
poetry_descriptives_df.to_csv(project_data_dir + poetry_descriptives_df_filename,index=False)




