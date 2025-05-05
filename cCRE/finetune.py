import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import accuracy_score, roc_auc_score

# Function to convert DNA sequences to one-hot encodings
def one_hot_encode(sequences):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((len(sequences), len(sequences[0]), 4), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for j, nucleotide in enumerate(sequence):
            if nucleotide in mapping:
                one_hot[i, j, mapping[nucleotide]] = 1
    return one_hot

# Custom dataset class for one-hot encoded sequences
class OneHotDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'encodings': self.encodings[idx], 'labels': self.labels[idx]}

# Custom dataset class
class EncodedData(Dataset):
    def __init__(self, encodings, attention_mask, labels):
        self.encodings = encodings
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        result = {}
        result['encodings'] = {key: tensor[idx] for key, tensor in self.encodings.items()}
        result['attention_mask'] = self.attention_mask[idx]
        result['labels'] = self.labels[idx]
        return result
    
# Define model structure for fine-tuning
class cCRE_BERT_MLP(nn.Module):
    def __init__(self, pretrained_model_path, cache_dir=None, is_finetune=False):

        super(cCRE_BERT_MLP, self).__init__()        
        config = BertConfig.from_pretrained(pretrained_model_path, cache_dir=cache_dir)
        self.bert = AutoModel.from_pretrained(pretrained_model_path, trust_remote_code=True, config=config, cache_dir=cache_dir).requires_grad_(is_finetune)
        self.dim = config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):   

        if token_type_ids is None:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # Use CLS token embeddings for classification
        # x = outputs[1]

        # Use meaning pooling for classification
        x = torch.mean(outputs[0], dim=1)
        logits = self.mlp(x)
        return logits

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = x.view(x.size(0), -1)  # Flatten the input
        return self.mlp(x)

# Function to load data
def load_data(dataset = 'all'):
    sequences = []
    labels = []
    dir = os.path.dirname(os.path.realpath(__file__))
    with open(f'{dir}/cCRE-data/fasta/{dataset}_cCREs.fasta','r') as f:
        cur_seq = ''
        for line in f:
            if '>' in line:
                sequences.append(cur_seq)
                cur_seq = ''
            else:
                cur_seq += line.strip(" \n")
    sequences.append(cur_seq)
    sequences.pop(0) #the first element in the list is just empty
    if 'non' in dataset: # negative samples
        labels = [0 for _ in range(len(sequences))]
    else: # positive samples
        labels = [1 for _ in range(len(sequences))]

    print(f'#sequences of {dataset}_cCREs dataset: {len(sequences)}')
    #print('length of the first sequence: %s' %len(sequences[0]))

    return sequences, labels

# Merged encode_data function
def encode_data(data_split, encoding_type, tokenizer=None):
    # Load the data for the positive and negative samples
    sequences_pos, labels_pos = load_data(data_split)
    sequences_neg, labels_neg = load_data(data_split + '_non')
    sequences = sequences_pos + sequences_neg
    labels = labels_pos + labels_neg
    sequences_pos = sequences_neg = None
    gc.collect()

    if encoding_type == "onehot":
        one_hot_encodings = one_hot_encode(sequences)
        dataset = OneHotDataset(one_hot_encodings, labels)
    elif encoding_type == "embedding":
        max_length = tokenizer.model_max_length
        encodings = tokenizer.batch_encode_plus(sequences, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        attention_mask = encodings['input_ids'] != tokenizer.pad_token_id
        print(f"#sequences of {data_split} dataset: {len(sequences)}")
        print(attention_mask)
        sequences = None
        gc.collect()
        dataset = EncodedData(encodings, attention_mask, labels)
    else:
        raise ValueError("Invalid encoding_type. Choose either 'onehot' or 'embedding'.")

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return data_loader

# Modified train function
def train(model, embed_mode, train_loader, validation_loader, optimizer, criterion, device, num_epochs=3):

    print("Training the model...")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
    
        for step, batch in enumerate(train_loader, 1):  # Start enumeration at 1
            if embed_mode == "embedding":
                input_ids = batch['encodings']['input_ids'].to(device)
                #token_type_ids = batch['encodings']['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device).float()
                #outputs = model(input_ids, token_type_ids, attention_mask)
                outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            elif embed_mode == "onehot":
                encodings = batch['encodings'].to(device)
                labels = batch['labels'].to(device).float()
                outputs = model(encodings)
        
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            total_loss += loss.item()
            
            if step % 1000 == 0:
                avg_train_loss = total_loss / step
                step_time = time.time() - start_time
                start_time = time.time()  # Reset start time for the next 500 steps
                
                # Validation
                model.eval()
                val_labels = []
                val_probs = []
                val_preds = []
                val_loss = 0
                with torch.no_grad():
                    for batch in validation_loader:
                        if embed_mode == "embedding":
                            input_ids = batch['encodings']['input_ids'].to(device)
                            #token_type_ids = batch['encodings']['token_type_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['labels'].to(device).float()
                            #outputs = model(input_ids, token_type_ids, attention_mask)
                            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
                        elif embed_mode == "onehot":
                            encodings = batch['encodings'].to(device)
                            labels = batch['labels'].to(device).float()
                            outputs = model(encodings)
                        
                        loss = criterion(outputs.squeeze(), labels)
                        val_loss += loss.item()
                        probs = outputs.squeeze()
                        preds = probs.round()
                    
                        val_labels.extend(labels.cpu().numpy())
                        val_probs.extend(probs.cpu().numpy())
                        val_preds.extend(preds.cpu().numpy())
                
                avg_val_loss = val_loss / len(validation_loader)
                val_accuracy = accuracy_score(val_labels, val_preds)
                val_auc = roc_auc_score(val_labels, val_probs)
                
                print(f"Epoch {epoch+1}/ Step {step} - Train Loss: {avg_train_loss} - Time: {step_time:.2f}s - "
                    f"Val Loss: {avg_val_loss} - Val Acc: {val_accuracy} - Val AUC: {val_auc}")
                
                return None

                model.train()

# Modified sancheck function
def sancheck(embed_mode, train_mode="probe", pretrained_model_path=None, cache_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device == "cuda":
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')
    
    config = BertConfig.from_pretrained(pretrained_model_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True, config=config, cache_dir=cache_dir)
    train_loader = encode_data('train', embed_mode, tokenizer)
    validation_loader = encode_data('validation', embed_mode, tokenizer)
    test_loader = encode_data('test', embed_mode, tokenizer)
    print("Data loaded.")

    if embed_mode == "onehot":
        input_size = train_loader.dataset[0]['encodings'].size
        model = SimpleMLP(input_size)
    elif embed_mode == "embedding":
        print(f"Hidden Layer sizes is {config.hidden_size}")
        is_finetune = False if train_mode == "probe" else True
        print(f"Finetuning is {is_finetune}")
        model = cCRE_BERT_MLP(pretrained_model_path, cache_dir=cache_dir, is_finetune=is_finetune)
    else:
        raise ValueError("Invalid train_mode. Choose either 'onehot' or 'embedding'.")

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.BCELoss()
    num_epochs = 5

    # Check if the weights of model.bert are frozen
    for name, param in model.bert.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} is trainable.")
        else:
            print(f"Parameter {name} is frozen.")

    # Training
    train(model, embed_mode, train_loader, validation_loader, optimizer, criterion, device, num_epochs)
    print("Finish test training ")
    return None
    
    # Testing
    print("Testing the model...")
    model.eval()
    test_labels = []
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            if embed_mode == "onehot":
                encodings = batch['encodings'].to(device)
                labels = batch['labels'].to(device).float()
                outputs = model(encodings)
            elif embed_mode == "embedding":
                input_ids = batch['encodings']['input_ids'].to(device)
                #token_type_ids = batch['encodings']['token_type_ids'].to(device)
                #attention_mask = batch['encodings']['attention_mask'].to(device)
                labels = batch['labels'].to(device).float()
                #outputs = model(input_ids, token_type_ids, attention_mask)
                outputs = model(input_ids)
            
            preds = outputs.squeeze().round()
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
    
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_preds)
    print(f"Test Accuracy: {test_accuracy}, Test AUC: {test_auc}")

# Example usage
if __name__ == "__main__":
    # embed_mode = "onehot" or "embedding"
    # train_mode = "probe" or "finetune", means whether to train the model with or finetune the pretrained model

    #llm_path = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    llm_path = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    cache_dir = "/hpc/group/pagelab/bl222/huggingface/."
    sancheck(embed_mode="embedding", train_mode="probe", pretrained_model_path=llm_path, cache_dir=cache_dir)