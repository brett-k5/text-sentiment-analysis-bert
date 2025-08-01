# Standard library imports
import math

# Third-party imports
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm

# Local imports
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False
# We don't want to have to import model_utils every time we import variables from this script
# So we are defining a function to import the BERT_text_to_embeddings function from model_utils.py.
# We will call this function inside the if __name__ == '__main__': block at the bottom.
def import_embedding_function():   
    if in_colab():
        from model_utils import BERT_text_to_embeddings
    else:
        from src.model_utils import BERT_text_to_embeddings
    return BERT_text_to_embeddings


# Data loading
if in_colab():
    from google.colab import files
    print("Upload imdb_reviews.tsv")
    uploaded = files.upload() # Upload imdb_reviews.tsv

df_reviews = pd.read_csv('imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})

df_reviews['review_norm'] = (
    df_reviews['review']
    .str.lower()
    .str.replace(r'[^a-z\s]', '', regex=True)
    .str.replace(r'[\s+]', ' ', regex=True)
    .str.strip()
)

df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

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
    train_features = BERT_text_to_embeddings(df_reviews_train['review_norm'], tokenizer, model, force_device='cuda')

    test_features = BERT_text_to_embeddings(df_reviews_test['review_norm'], tokenizer, model, force_device='cuda')

    np.savez_compressed('embedded_features.npz', X_train=train_features, X_test=test_features)

    with np.load('embedded_features.npz') as data:
        train_features = data['X_train']
        test_features = data['X_test']

    if in_colab():
        files.download('embedded_features.npz')