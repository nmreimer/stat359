import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm


os.chdir(os.path.dirname(os.path.abspath(__file__))) # temporary

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        self.data = torch.tensor(skipgram_df.values, dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx): # returns center, context
        return self.data[idx, 0], self.data[idx, 1]

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
    def forward(self, center, context):
        center_embeddings = self.embeddings(center)
        context_embeddings = self.output_embeddings(context)
        return torch.sum(center_embeddings * context_embeddings, dim=1)

# Load processed data


with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

counter = data['counter']
word2idx = data['word2idx']
vocab_size = len(word2idx)

# Precompute negative sampling distribution below
def get_negative_sampling_distribution(counter, vocab_size, word2idx):
    negative_sampling_distribution = torch.zeros(vocab_size)
    for word, count in counter.items():
        if word in word2idx:
            negative_sampling_distribution[word2idx[word]] = count
    negative_sampling_distribution = negative_sampling_distribution ** (3/4)
    negative_sampling_distribution = negative_sampling_distribution / negative_sampling_distribution.sum()
    return negative_sampling_distribution

negative_sampling_distribution = get_negative_sampling_distribution(counter, vocab_size, word2idx)


# Device selection: CUDA > MPS > CPU

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# negative_sampling_distribution = negative_sampling_distribution.to(device)

# Dataset and DataLoader
skipgram_dataset = SkipGramDataset(data['skipgram_df'])
skipgram_dataloader = DataLoader(skipgram_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

def make_targets(batch_size, center, context, negative_sampling_distribution):
    num_neg_samples = NEGATIVE_SAMPLES * batch_size
    neg_samples = torch.multinomial(negative_sampling_distribution, num_neg_samples, replacement=True)
    neg_samples = neg_samples.view(batch_size, NEGATIVE_SAMPLES)
    device = context.device
    neg_samples = neg_samples.to(device) 
    context_unsqueezed = context.unsqueeze(1)
    collision_mask = (neg_samples == context_unsqueezed)
    if collision_mask.any():
        num_collisions = collision_mask.sum().item()
        replacements = torch.multinomial(negative_sampling_distribution, num_collisions, replacement=True)
        neg_samples[collision_mask] = replacements.to(device)

    neg_samples = neg_samples.view(-1)
    
    pos_targets = torch.ones(batch_size, device=device)
    neg_targets = torch.zeros(num_neg_samples, device=device)
    
    return pos_targets, neg_targets, neg_samples

# Training loop

for epoch in range(EPOCHS):
    total_loss = 0
    for center, context in tqdm(skipgram_dataloader):
        center = center.to(device)
        context = context.to(device)
        optimizer.zero_grad()
        current_batch_size = center.size(0)

        pos_targets, neg_targets, neg_contexts = make_targets(
            batch_size = current_batch_size,
            center = center,
            context = context,
            negative_sampling_distribution = negative_sampling_distribution
        )

        pos_scores = model(center, context)
        pos_loss = criterion(pos_scores, pos_targets)

        center_repeated = center.repeat_interleave(NEGATIVE_SAMPLES)
        neg_scores = model(center_repeated, neg_contexts)
        neg_loss = criterion(neg_scores, neg_targets)   

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()     

    avg_loss = total_loss / len(skipgram_dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

embeddings = model.embeddings.weight.data.cpu().numpy()

# Save embeddings and mappings
# embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
