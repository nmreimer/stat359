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

num_epochs = 40
lr = 0.0005
batch_size = 64

dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")

model_path = "student/Assignment_2/fasttext-wiki-news-subwords-300.model"
embedding_model = KeyedVectors.load(model_path, mmap='r')

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def get_sentence_embeddings(sentences, model):
    embeddings = []
    emb_dim = 300
    for sentence in sentences:
        tokens = simple_preprocess(sentence)
        sentence_embedding = np.zeros((32, emb_dim))
        if len(tokens)>32:
            tokens = tokens[:32]
        for i, token in enumerate(tokens):
            if token in model.key_to_index:
                sentence_embedding[i] = model[token]
        embeddings.append(sentence_embedding)
    return np.array(embeddings)

X_seq = get_sentence_embeddings(data['sentence'], embedding_model)
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

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):   
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out.mean(dim=1) # mean pooling
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
model = LSTMClassifier(input_size=300, hidden_size=128, num_classes=3, num_layers=2)
device = get_device()
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []
train_acc_history = []
val_acc_history = []
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
    epoch_train_acc = (np.array(all_preds) == np.array(all_labels)).mean()

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
    epoch_val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    
    train_loss_history.append(epoch_train_loss)
    val_loss_history.append(epoch_val_loss)
    train_f1_history.append(epoch_train_f1)
    val_f1_history.append(epoch_val_f1)
    train_acc_history.append(epoch_train_acc)
    val_acc_history.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {epoch_train_loss:.4f} | Train Macro F1: {epoch_train_f1:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"Val   Loss: {epoch_val_loss:.4f} | Val   Macro F1: {epoch_val_f1:.4f} | Val   Acc: {epoch_val_acc:.4f}")

    if epoch_val_f1 > best_val_f1 and epoch > 29:
        best_val_f1 = epoch_val_f1
        best_model_wts = copy.deepcopy(model.state_dict())


# ========== Plot Learning Curves ==========
print("\n========== Plotting Learning Curves ==========")
plt.figure(figsize=(12, 15))

plt.subplot(3, 1, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(train_f1_history, label='Train F1')
plt.plot(val_f1_history, label='Val F1')
plt.title('F1 Macro Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('student/Assignment_3/outputs/lstm_f1_learning_curves.png')
# plt.show()  # Commented out to prevent display, plots are saved instead


# Save accuracy plot separately
plt.figure(figsize=(8, 6))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('student/Assignment_3/outputs/lstm_accuracy_learning_curve.png')
# plt.show()  # Commented out to prevent display, plots are saved instead


# ========== Test Evaluation ==========
print("\n========== Evaluating on Test Set ==========")
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

test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
test_f1_macro = f1_score(all_labels, all_preds, average='macro')
test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

print('\n' + '='*50)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Macro: {test_f1_macro:.4f}")
print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
print('='*50 + '\n')

class_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('student/Assignment_3/outputs/lstm_confusion_matrix.png')
# plt.show()  # Commented out to prevent display, plots are saved instead


print("\nPer-class F1 Scores:")
per_class_f1 = f1_score(all_labels, all_preds, average=None)
for i, name in enumerate(class_names):
    class_f1 = per_class_f1[i]
    print(f"{name}: {class_f1:.4f}")

# Save results to CSV for model comparison
print("\n========== Saving Results to CSV ==========")
results_df = pd.DataFrame({
    'model': ['LSTM'],
    'macro_f1': [test_f1_macro],
    'f1_negative': [per_class_f1[0]],
    'f1_neutral': [per_class_f1[1]],
    'f1_positive': [per_class_f1[2]]
})

csv_path = 'student/Assignment_3/outputs/model_performance.csv'
# Check if CSV exists and append or create new
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    # Remove existing row for this model if it exists
    existing_df = existing_df[existing_df['model'] != 'LSTM']
    results_df = pd.concat([existing_df, results_df], ignore_index=True)
else:
    os.makedirs('student/Assignment_3/outputs', exist_ok=True)

results_df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")