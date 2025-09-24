# Standard library imports
import math

# Third-party imports
import numpy as np
import pandas as pd
import transformers
import torch
from tqdm.auto import tqdm

# Local application imports
def in_colab():
    """
    Checks to see if script is being run in a google colab environment.
    If it is the function returns True. If it isn't the function returns False
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False
if in_colab():
    from model_utils import BERT_text_to_embeddings # import statement for colab environment
else:
    from src.model_utils import BERT_text_to_embeddings # import statement if run from project directory



# Data loading
if in_colab():
    from google.colab import files
    print("Upload imdb_reviews.tsv")
    uploaded = files.upload() # Upload imdb_reviews.tsv

# Load data
df_reviews = pd.read_csv('imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})

# Clean features
df_reviews['review_norm'] = (
    df_reviews['review']
    .str.lower()
    .str.replace(r'[^a-z\s]', '', regex=True)
    .str.replace(r'[\s+]', ' ', regex=True)
    .str.strip()
)

# Create training and test sets
df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

# Define target for training and test set
train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

# Define tokenizing function
def tok_conf_mod():
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    config = transformers.BertConfig.from_pretrained('bert-base-uncased')
    model = transformers.BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, config, model


if __name__ == '__main__':

    # Assign values to tokenizer, config, and model variables
    tokenizer, config, model = tok_conf_mod()

    # Import BERT_text_to_embeddings by calling import_embedding_function
    BERT_text_to_embeddings = import_embedding_function() 

    # Attention! Running BERT for thousands of texts may take long run on CPU, at least several hours
    train_features = BERT_text_to_embeddings(df_reviews_train['review_norm'], tokenizer, model, force_device='cuda') # train
    test_features = BERT_text_to_embeddings(df_reviews_test['review_norm'], tokenizer, model, force_device='cuda') # test

    # Save embedded features
    np.savez_compressed('embedded_features.npz', X_train=train_features, X_test=test_features) 
    with np.load('embedded_features.npz') as data:
        train_features = data['X_train']
        test_features = data['X_test']
    
    # Download embedded features from GPU environment
    if in_colab():
        files.download('embedded_features.npz')