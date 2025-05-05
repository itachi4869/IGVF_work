from transformers.models.bert.modeling_bert import BertConfig
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data_utils
import torch.optim as optim
import tensorflow as tf
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
import gzip
import time
import gc
import os
import sys

def load_data(faFile, cntFile):
    
    sequences = []
    with gzip.open(faFile,'rb') as f:
        cur_seq = ''
        for line in f:
            line = line.decode('utf-8')
            if '>' in line:
                sequences.append(cur_seq)
                cur_seq = ''
            else:
                cur_seq += line.strip()
    sequences.append(cur_seq)
    sequences.pop(0) #the first element in the list is just empty

    N, len_seq = len(sequences), len(sequences[0])
    print(f'#sequences of prediction: {N}')
    print(f'length of the first sequence: {len_seq}')

    counts = []
    cur_line = 0
    with gzip.open(cntFile,'rb') as f:
        cur_seq = ''
        for line in f:
            if cur_line == 0: 
                cur_line += 1
                continue
            line = line.decode('utf-8')
            counts.append(list(map(float, line.split())))
            cur_line += 1
    
    # Create pairs of sequences and counts
    seq_count_pairs = list(zip(sequences, counts))
    
    # Filter pairs where sequence has N/n or incorrect length
    filtered_pairs = [(seq, count) for seq, count in seq_count_pairs if 'N' not in seq and 'n' not in seq and (len(seq) == len_seq)]
    
    # Unzip the filtered pairs
    filtered_sequences, filtered_counts = zip(*filtered_pairs) if filtered_pairs else ([], [])
    
    print(f'Number of sequences without N or n and length {len_seq}: {len(filtered_sequences)}')

    return list(filtered_sequences), list(filtered_counts)

def oneHot(seqs):
    ALPHA = {'A':0, 'C':1, 'G':2, 'T':3}
    numSeqs = len(seqs)
    L = len(seqs[0])
    X = np.zeros((numSeqs, L, 4))
    for j, seq in enumerate(seqs):
        for i in range(L):
            c = seq[i].upper() # Convert a,t,c,g to A,T,C,G
            cat = ALPHA.get(c, -1)
            if(cat >= 0): X[j, i, cat] = 1
    return X

# Create a simple dataset
class SeqDataset(Dataset):
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        x = self.X[idx]
        x = torch.tensor(x, dtype=torch.float32)
        # Fix tensor shape: [L, 4] -> [4, 1, L]
        # Model expects [batch, channels, height, width]
        x = x.permute(1, 0).unsqueeze(1)
        return x

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Beluga(nn.Module): # copied from https://github.com/kipoi/models/blob/master/DeepSEA/beluga/model.py
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4, 320, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(320, 320, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(320, 480, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(480, 480, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(480, 640, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(640, 640, (1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0), -1)),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(67840, 2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2003, 2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

def main(modelFile, faFile, cntFile, batch_size=512, num_chunks=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load ALL sequences and counts at once
    print("Loading all sequences and counts...")
    sequences, counts = load_data(faFile, cntFile)
    total_sequences = len(sequences)
    print(f"Total loaded sequences: {total_sequences}")
    
    # Load the model weights
    model_weights = torch.load(modelFile)
    model = Beluga()
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    print(model)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a base name for output files
    dataset_name = os.path.basename(faFile).split('.')[0]
    base_name = f"deepsea_{dataset_name}"
    manifest_file = os.path.join(output_dir, f"{base_name}_manifest.txt")
    
    # Initialize manifest file
    with open(manifest_file, 'w') as mf:
        mf.write(f"Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        mf.write(f"Model: {os.path.basename(modelFile)}\n")
        mf.write(f"Input fasta: {faFile}\n")
        mf.write(f"Input counts: {cntFile}\n")
        mf.write(f"Number of chunks: {num_chunks}\n")
        mf.write(f"Total sequences: {total_sequences}\n")
        mf.write("Chunk files:\n")
    
    # Define chunk size
    chunk_size = (total_sequences + num_chunks - 1) // num_chunks  # Ceiling division
    
    # Process sequences in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_sequences)
        
        if start_idx >= total_sequences:
            break  # No more sequences to process
        
        print(f"Processing chunk {chunk_idx+1}/{num_chunks}, sequences {start_idx} to {end_idx-1}")
        
        # Get subset of sequences for this chunk
        chunk_sequences = sequences[start_idx:end_idx]
        chunk_counts = counts[start_idx:end_idx]
        
        # One-hot encode only this chunk
        print(f"One-hot encoding chunk {chunk_idx+1} ({len(chunk_sequences)} sequences)...")
        X = oneHot(chunk_sequences)
        print(f"One-hot encoding shape: {X.shape}")
        
        # Create dataset and dataloader for this chunk
        dataset = SeqDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Process in batches and collect predictions
        predictions = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # Move batch to device, make predictions
                batch = batch.to(device)
                batch_preds = model(batch)
                batch_preds_cpu = batch_preds.cpu().numpy()
                predictions.append(batch_preds_cpu)
                current_batch_size = batch_preds_cpu.shape[0]
                
                print(f"Chunk {chunk_idx+1}: Processed batch {i+1}/{len(dataloader)} ({current_batch_size} samples)")
        
        # Stack all predictions for this chunk
        chunk_preds = np.vstack(predictions)
        
        # Create output filename for this chunk
        chunk_file = os.path.join(output_dir, f"{base_name}_chunk{chunk_idx:03d}.npz")
        print(f"Saving chunk {chunk_idx+1} with {len(chunk_sequences)} samples (indices {start_idx}-{end_idx-1}) to {chunk_file}")
        
        try:
            # Save this chunk's data
            np.savez_compressed(
                chunk_file,
                counts=chunk_counts,
                embeddings=chunk_preds,
                indices=np.array([start_idx, end_idx])
            )
            
            # Update the manifest
            with open(manifest_file, 'a') as mf:
                mf.write(f"{os.path.basename(chunk_file)}: indices {start_idx}-{end_idx-1}, samples: {len(chunk_sequences)}\n")
                mf.flush()
                
        except Exception as e:
            print(f"Error saving chunk {chunk_idx}: {str(e)}")
        
        # Clear memory
        del chunk_sequences
        del chunk_counts
        del X
        del dataset
        del dataloader
        del predictions
        del chunk_preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"Finished processing chunk {chunk_idx+1}/{num_chunks}")
    
    # Cleanup full sequences and counts at the end
    del sequences
    del counts
    gc.collect()
    
    print(f"Completed all data chunks. Results saved to {output_dir}")
    print(f"Manifest file: {manifest_file}")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python DeepSEA.py <modelFile> <dataPath> <dataset> [batch_size] [num_chunks]")
        sys.exit(1)
    
    (modelFile, dataPath, dataset) = sys.argv[1:4]
    
    batch_size = 1024  # Smaller batch size to avoid OOM
    if len(sys.argv) > 4:
        batch_size = int(sys.argv[4])
    
    num_chunks = 6  # Default number of chunks to split data into
    if len(sys.argv) > 5:
        num_chunks = int(sys.argv[5])
    
    faFile = dataPath + '/%s.fasta.gz' % dataset
    cntFile = dataPath + '/%s-counts.txt.gz' % dataset
    
    main(modelFile, faFile, cntFile, batch_size, num_chunks)