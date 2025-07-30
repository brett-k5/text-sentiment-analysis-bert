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


tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
model = transformers.BertModel.from_pretrained('bert-base-uncased')

def BERT_text_to_embeddings(texts, max_length=512, batch_size=100, force_device=None, disable_progress_bar=False):

    ids_list = []
    attention_mask_list = []

    for text in texts:
        ids = tokenizer.encode(
            text.lower(), add_special_tokens=True, truncation=True, max_length=max_length)
        padded = np.array(ids + [0]*(max_length - len(ids)))
        attention_mask = np.where(padded != 0, 1, 0)
        ids_list.append(padded)
        attention_mask_list.append(attention_mask)

    if force_device is not None:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    if not disable_progress_bar:
        print(f'Using the {device} device.')

    # getting embeddings in batches

    embeddings = []

    for i in tqdm(range(math.ceil(len(ids_list)/batch_size)), disable=False):

        ids_batch = torch.LongTensor(np.array(ids_list[batch_size*i:batch_size*(i+1)])).to(device)
        attention_mask_batch = torch.LongTensor(np.array(attention_mask_list[batch_size*i:batch_size*(i+1)])).to(device)

        with torch.no_grad():
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)
        embeddings.append(batch_embeddings[0][:,0,:].detach().cpu().numpy())

    return np.concatenate(embeddings)

# Attention! Running BERT for thousands of texts may take long run on CPU, at least several hours
train_features = BERT_text_to_embeddings(df_reviews_train['review_norm'], force_device='cuda')

test_features = BERT_text_to_embeddings(df_reviews_test['review_norm'], force_device='cuda')

np.savez_compressed('embedded_features.npz', X_train=train_features, X_test=test_features)

with np.load('embedded_features.npz') as data:
     train_features = data['X_train']
     test_features = data['X_test']