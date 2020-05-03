import pandas as pd
import numpy as np

from os import path

# Load data

def load_data(file_path):
    assert isinstance(file_path, str), 'Argument {} is not a string'
    assert path.exists(file_path), '{} does not exist'.format(path)

    df = pd.read_csv(file_path)

    return df

def drop_correlated_features(df, threshold=0.99):
    # calculate correlation matrix between columns
    corr_matrix = df.corr()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # drop correlated columns
    filtered_df = df.drop(df[to_drop], axis=1)

    return filtered_df

def calculate_correlation_matrix():
    # select only tags used more than 5k times
    tag_counts = pd.merge(book_tags, tags_df, on='tag_id').groupby('tag_id')['count'].sum().sort_values(ascending=False)
    top_tags = tag_counts[tag_counts>5000]

    # apply filter
    filtered_book_tags = book_tags[book_tags['tag_id'].isin(top_tags.index)]

    # aggregate to remove duplicate tags
    grouped_book_tags = filtered_book_tags.groupby(['goodreads_book_id','tag_id'])['count'].sum().reset_index()

    # pivot tables (books x tags)
    pivoted_book_table = grouped_book_tags.pivot('goodreads_book_id','tag_id','count')

    filtered_pivoted_book_table = drop_correlated_features(pivoted_book_table)

    # change table from count to binary (1 or 0)
    book_tag_table = (filtered_pivoted_book_table>0).astype(int)

    # calculate correlation matrix between books
    correlation_matrix = book_tag_table.T.corr()

    return correlation_matrix

def save_correlation_matrix(correlation_matrix):
    correlation_matrix.to_csv('data/correlation_matrix.csv')
    return True

# Function to be called outside
def find_k_similar_books(goodreads_book_id, n=20):
    similar_books = correlation_matrix[int(goodreads_book_id)].sort_values(ascending=False).iloc[:n]
    similar_books.name = 'correlation'

    similar_books_merged = pd.merge(similar_books,books_df,left_index=True,right_on='goodreads_book_id')
    
    return similar_books_merged.to_dict(orient='records')

# seach title or author name
def book_lookup(search_string, n=20):
    cond1 = books_df['title'].str.lower().str.contains(search_string.lower().replace('+',' '))
    cond2 = books_df['authors'].str.lower().str.contains(search_string.lower())
    return books_df[cond1|cond2].iloc[:n].to_dict(orient='records')

# get top books
def top_books(n=20):
    return books_df.sort_values('average_rating',ascending=False).iloc[:n].to_dict(orient='records')


# Load data

# ratings_df = load_data('data/ratings.csv')
books_df = load_data('data/books.csv')
tags_df = load_data('data/tags.csv')
book_tags = load_data('data/book_tags.csv')

correlation_matrix = calculate_correlation_matrix()

# n = 20
# goodreads_book_id = 12609433
# book_recommendations = find_k_similar_books(correlation_matrix, books_df, goodreads_book_id, n)