#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
from torch.optim import Adam
from utils import load_data, get_tokenizer, fetch_data_loader
from models import EncoderCNN, DecoderLSTM
from hparams import *


# In[ ]:


import torch
from torch import nn
from torchvision.models import resnet50


# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from nltk import WhitespaceTokenizer
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# In[ ]:


def load_data(data_path, test_size):
    df = pd.read_csv(data_path)
    image_paths = df['image'].values
    captions = df['caption'].values
    list_unique_words = list(set(' '.join(captions).split(' ')))
    train_image_paths, test_image_paths, train_captions, test_captions = train_test_split(
        image_paths, captions, test_size=test_size, shuffle=True, random_state=1104)
    return list_unique_words, train_image_paths, test_image_paths, train_captions, test_captions


# In[ ]:


def get_tokenizer(list_unique_words):
    whitespace_tokenizer = WhitespaceTokenizer()
    special_tokens = ['SOS', 'EOS', 'UNK']
    list_unique_words = special_tokens + list_unique_words
    dict_vocab_to_idx, dict_idx_to_vocab = dict(), dict()
    for idx, word in enumerate(list_unique_words):
        dict_vocab_to_idx[word] = idx
        dict_idx_to_vocab[idx] = word
    return whitespace_tokenizer, list_unique_words, dict_vocab_to_idx, dict_idx_to_vocab


# In[ ]:


def tokens_to_indices(tokens, dict_vocab_to_idx):
    encoded_text = list()
    encoded_text.append(dict_vocab_to_idx['SOS'])
    for token in tokens:
        try:
            encoded_text.append(dict_vocab_to_idx[token])
        except:
            encoded_text.append(dict_vocab_to_idx['UNK'])
    encoded_text.append(dict_vocab_to_idx['EOS'])
    return torch.tensor(encoded_text).flatten()


# In[ ]:


def fetch_data_loader(image_paths, captions, tokenizer, dict_vocab_to_idx, is_shuffle):
    this_dataset = CaptionDataset(image_paths, captions, tokenizer=tokenizer, dict_vocab_to_idx=dict_vocab_to_idx)
    return DataLoader(this_dataset, batch_size=1, shuffle=is_shuffle)


# In[ ]:


class CaptionDataset(Dataset):
    def __init__(self, image_paths, captions, tokenizer, dict_vocab_to_idx):
        super(CaptionDataset, self).__init__()
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.dict_vocab_to_idx = dict_vocab_to_idx
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, this_index):
        this_caption = self.captions[this_index]
        this_image_path = os.path.join('data/Images/', self.image_paths[this_index])
        this_image = mpimg.imread(this_image_path)
        transformed_image = self.transform(this_image)
        tokenized_caption = tokens_to_indices(self.tokenizer.tokenize(this_caption), self.dict_vocab_to_idx)
        return transformed_image, this_caption, tokenized_caption, this_image_path


# In[ ]:


THIS_DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

N_EPOCHS = 5
BATCH_SIZE = 1
EMBED_DIM = 512

MODEL_PATH = None


# In[ ]:


class EncoderCNN(nn.Module):
    def __init__(self, embed_dim):
        super(EncoderCNN, self).__init__()
        self.embed_dim = embed_dim
        resnet_model = resnet50(pretrained=True)
        for param in resnet_model.parameters():
            param.requires_grad_(False)
        feat_extractor_modules = list(resnet_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*feat_extractor_modules)
        self.embed_layer = nn.Sequential(
            nn.Linear(resnet_model.fc.in_features, self.embed_dim)
        )

    def forward(self, input_images):
        n_images = input_images.shape[0]
        features = self.feature_extractor(input_images).view(n_images, 1, -1)
        embedded_features = self.embed_layer(features)
        return embedded_features


# In[ ]:


class DecoderLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, n_layers=1):
        super(DecoderLSTM, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_layer = nn.Embedding(self.vocab_size, self.embed_dim)
        self.sequence_generator = nn.LSTM(input_size=2*embed_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.init_hidden = nn.Parameter(torch.zeros(n_layers, BATCH_SIZE, hidden_dim))
        self.init_cell = nn.Parameter(torch.zeros(n_layers, BATCH_SIZE, hidden_dim))
        self.token_classifier = nn.Sequential(
            nn.Linear(hidden_dim, self.vocab_size),
            nn.Softmax(dim=1)
        )

    def forward(self, image_features, this_token_id, hidden_states, cell_states):
        token_vector = self.embed_layer(this_token_id)
        combined_features = torch.cat((image_features.view(-1, 1, self.embed_dim), token_vector.view(-1, 1, self.embed_dim)), dim=2)
        lstm_output, (hidden_states, cell_states) = self.sequence_generator(combined_features, (hidden_states, cell_states))
        predicted_token = self.token_classifier(lstm_output.view(-1, self.hidden_dim))
        return predicted_token, hidden_states, cell_states


# In[ ]:


# Load data
data_path = 'data/captions/captions.txt'
list_unique_words, train_image_paths, test_image_paths, train_captions, test_captions = load_data(data_path=data_path, test_size=0.2)


# In[ ]:


# Load tokenizer
whitespace_tokenizer, list_unique_words, dict_vocab_to_idx, dict_idx_to_vocab = get_tokenizer(list_unique_words)


# In[ ]:


# Instantiate models
image_encoder = EncoderCNN(embed_dim=EMBED_DIM).to(THIS_DEVICE)
caption_generator = DecoderLSTM(embed_dim=EMBED_DIM, hidden_dim=EMBED_DIM, vocab_size=len(list_unique_words), n_layers=2).to(THIS_DEVICE)


# In[ ]:


# Instantiate loss function
loss_function = torch.nn.BCELoss().to(THIS_DEVICE)


# In[ ]:


# Instantiate optimizer
params = list(image_encoder.embed_layer.parameters()) + list(caption_generator.parameters())
optimizer = Adam(params=params, lr=3e-4)


# In[ ]:


# Training
list_training_loss = list()
for this_epoch in range(N_EPOCHS):
    this_data_loader = fetch_data_loader(image_paths=train_image_paths, captions=train_captions, tokenizer=whitespace_tokenizer,
                                         dict_vocab_to_idx=dict_vocab_to_idx, is_shuffle=True)
    n_images = 0
    for this_image, this_caption, tokenized_caption, _ in this_data_loader:
        n_images += 1
        image_features = image_encoder(this_image.to(THIS_DEVICE))
        hidden_states, cell_states = caption_generator.init_hidden, caption_generator.init_cell
        this_loss = torch.tensor([0], dtype=torch.float32, device=THIS_DEVICE)
        predicted_caption = ''
        tokenized_caption = tokenized_caption[0]
        for this_idx in range(len(tokenized_caption)-1):
            this_token_id = torch.tensor(tokenized_caption[this_idx], device=THIS_DEVICE)
            next_token_id = torch.tensor(tokenized_caption[this_idx+1], device=THIS_DEVICE)
            target_token = torch.zeros((1, len(list_unique_words)), device=THIS_DEVICE)
            target_token[:, next_token_id] = 1
            predicted_token, hidden_states, cell_states = caption_generator(image_features, this_token_id, hidden_states, cell_states)
            predicted_token_id = torch.argmax(predicted_token)
            predicted_caption += dict_idx_to_vocab[predicted_token_id.item()] + ' '
            this_loss += loss_function(predicted_token, target_token)
        this_loss /= len(tokenized_caption) - 1
        print(f'Epoch: {this_epoch}, Image No: {n_images}, Target caption is: {this_caption[0]}')
        print(f'Predicted caption is: {predicted_caption[:-1]}\n')

        list_training_loss.append(this_loss.item())
        this_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# In[ ]:


# Display train loss history
plt.plot(list_training_loss)
plt.xlabel('Iterations')
plt.ylabel('BCE Loss')
plt.title('Training Loss')
plt.grid()
plt.show()


# In[ ]:


# Evaluation
this_data_loader = fetch_data_loader(image_paths=test_image_paths, captions=test_captions, tokenizer=whitespace_tokenizer,
                                     dict_vocab_to_idx=dict_vocab_to_idx, is_shuffle=False)
image_encoder.eval()
caption_generator.eval()
n_images = 0
for this_image, this_caption, _, this_image_path in this_data_loader:
    n_images += 1
    image_features = image_encoder(this_image.to(THIS_DEVICE))
    hidden_states, cell_states = caption_generator.init_hidden, caption_generator.init_cell
    predicted_caption = ''
    this_token_id = torch.tensor(dict_vocab_to_idx['SOS'], device=THIS_DEVICE)
    n_steps = 0
    while True:
        n_steps += 1
        predicted_token, hidden_states, cell_states = caption_generator(image_features, this_token_id, hidden_states, cell_states)
        predicted_token_id = torch.argmax(predicted_token)
        predicted_caption += dict_idx_to_vocab[predicted_token_id.item()] + ' '
        this_token_id = predicted_token_id
        if predicted_token_id.item() == dict_vocab_to_idx['EOS'] or n_steps > 50:
            break
    plt.imshow(plt.imread(this_image_path[0]))
    plt.title(this_caption[0] + '\n=>\n' + predicted_caption)
    plt.show()

