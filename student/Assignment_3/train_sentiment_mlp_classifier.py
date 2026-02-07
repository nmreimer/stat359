import numpy as np
import pandas as pd
import datasets
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset
import copy

num_epochs = 30
lr = 0.001
batch_size = 32


dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")

model_path = "student/Assignment_2/fasttext-wiki-news-subwords-300.model"
embedding_model = KeyedVectors.load(model_path, mmap='r')

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def get_mean_sentence_embeddings(sentences, model):
    embeddings = []
    for sentence in sentences:
        tokens = simple_preprocess(sentence)
        vectors = [model[token] for token in tokens if token in model.key_to_index]
        if len(vectors) > 0:
            mean_vec = np.mean(vectors, axis=0)
            embeddings.append(mean_vec)
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.vstack(embeddings)


# print("\n========== Encoding Sentences as Sequences ==========")
max_seq_len = 32
X_seq = get_mean_sentence_embeddings(data['sentence'], embedding_model)
y = data['label'].values


# # ========== Train/Test Split ==========
print("\n========== Splitting Data ==========")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_seq, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
)

class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)

# convert to tensor
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Train MLP Classifier

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.dropout(x)
        x = self.activation2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

device = get_device()
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)


model = MLPClassifier(input_dim=300, hidden_dim=128, num_classes=3)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []
best_val_f1 = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):

    # training

    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_f1 = f1_score(all_labels, all_preds, average='macro')

    # validation
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_val_loss = running_loss / len(val_dataset)
    epoch_val_f1 = f1_score(all_labels, all_preds, average='macro')

    train_loss_history.append(epoch_train_loss)
    val_loss_history.append(epoch_val_loss)
    train_f1_history.append(epoch_train_f1)
    val_f1_history.append(epoch_val_f1)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {epoch_train_loss:.4f} | Train Macro F1: {epoch_train_f1:.4f}")
    print(f"Val   Loss: {epoch_val_loss:.4f} | Val   Macro F1: {epoch_val_f1:.4f}")

    if epoch_val_f1 > best_val_f1:
        best_val_f1 = epoch_val_f1
        best_model_wts = copy.deepcopy(model.state_dict())


# evaluation
model.load_state_dict(best_model_wts)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_macro_f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Final Test Macro F1 Score: {test_macro_f1:.4f}")
print(classification_report(all_labels, all_preds, digits=4))