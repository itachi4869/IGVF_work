import gc
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW 
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

# Custom dataset class
class EncodedDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].size(0)

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}

# get dataloaders for the training sets
def load_data(dataset='all'):
    sequences = []
    dir = os.path.dirname(os.path.realpath(__file__))
    with open('%s/cCRE-data/fasta/%s_cCREs.fasta'%(dir, dataset),'r') as f:
        cur_seq = ''
        for line in f:
            if '>' in line:
                sequences.append(cur_seq)
                cur_seq = ''
            else:
                cur_seq += line.strip()
    sequences.append(cur_seq)
    sequences.pop(0) #the first element in the list is just empty

    print('#sequences of %s dataset: %s' %(dataset, len(sequences)))
    print('length of the first sequence: %s' %len(sequences[0]))

    return sequences

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "zhihan1996/DNABERT-2-117M"
    config = BertConfig.from_pretrained(model_name)
    #model = BertForMaskedLM.from_pretrained(model_name, trust_remote_code=True, config=config) # if use pre-trained model
    model = BertForMaskedLM(config=config) # if train a new model
    model.init_weights()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, config=config)
    
    sequences = load_data('all')
    encodings = tokenizer(sequences, add_special_tokens=False, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    sequences = None
    gc.collect()

    # Create the dataset
    dataset = EncodedDataset(encodings)
    encodings = None
    gc.collect()

    # mask input tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15  # 15% of tokens will be masked
    )

    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=data_collator)
    N_batches = len(train_dataloader)
    print("Number of batches in the training set:", N_batches)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    '''
    N_steps = 100000
    steps_per_epoch = len(train_dataloader)
    num_epochs = N_steps // steps_per_epoch
    '''

    num_epochs = 10
    model.to(device)
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        for batch in train_dataloader:
            # Move batch to device
            batch = {k: v.squeeze().to(device) for k, v in batch.items()}
        
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
        
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            total_loss += loss.item()
    
        avg_loss = total_loss / N_batches
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss} - Time: {epoch_duration:.2f}s")

    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f}s")

    model.save_pretrained('./models/db2_pretrained_model_10epochs_no_special_tokens')
    tokenizer.save_pretrained('./models/db2_pretrained_model_10epochs_no_special_tokens')

if __name__ == '__main__':
    train()