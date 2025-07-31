# Third-party imports
import numpy as np              # Used for np.savez_compressed and np.load
import pandas as pd            # Used for DataFrame string operations, query, copy
import transformers            # Used for loading BERT tokenizer, config, model

# Local imports
# Local imports
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False
    
if in_colab():
    from google.colab import files
    from model_utils import BERT_text_to_embeddings
    from data_pre_processing import tokenizer, model 

else:
    from src.model_utils import BERT_text_to_embeddings  # Used to embed text
    from src.data_pre_processing import tokenizer, model

custom_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and fell asleep in the middle of the movie.',
    'I was really fascinated with the movie',    
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])

custom_reviews['review_norm'] = (
    custom_reviews['review']
    .str.lower()
    .str.replace(r'[^a-z\s]', '', regex=True)
    .str.replace(r'[\s+]', ' ', regex=True)
    .str.strip()
)

embedded_custom_reviews = BERT_text_to_embeddings(custom_reviews['review_norm'], tokenizer, model, force_device='cuda')

np.savez_compressed('embedded_custom_reviews.npz', cust_test=embedded_custom_reviews)

with np.load('embedded_custom_reviews.npz') as data:
     embedded_custom_reviews = data['cust_test']

if in_colab():
    files.download('embedded_custom_reviews.npz')