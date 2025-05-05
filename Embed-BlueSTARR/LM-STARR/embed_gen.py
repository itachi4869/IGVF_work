#!/usr/bin/env python
#==============================================================================
# Generate embeddings for all the input sequences with NT-STARR model 
# Read all sequences at once and output embeddings in chunks
#==============================================================================

import os
import sys
import gc
import gzip
import time
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

# Set the task ID for SLURM array jobs
taskID=int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else 0

def load_data(data_path, dataset):
    sequences = []
    with gzip.open(f'{data_path}/{dataset}/{dataset}_part_{taskID:02d}.fasta.gz','rb') as f:
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

    counts = []
    cur_line = 0
    with gzip.open(f'{data_path}/{dataset}/{dataset}-counts_part_{taskID:02d}.txt.gz','rb') as f:
        cur_seq = ''
        for line in f:
            if cur_line == 0: 
                cur_line += 1
                continue
            line = line.decode('utf-8')
            counts.append(list(map(int, line.split())))
            cur_line += 1

    print('%s length: %s'%(dataset, len(sequences)))
    return sequences, counts

def get_embeddings(data_path, dataset, batch_size, model_name, cache_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    if cache_dir is not None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, config=config, cache_dir=cache_dir)
        seq_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config, cache_dir=cache_dir)
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, config=config)
        seq_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)

    print(seq_encoder)
    seq_encoder.to(device)
    seq_encoder.eval()

    sequences, counts = load_data(data_path, dataset)
    print('Data loaded.')

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        inputs = tokenizer(batch_sequences, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = seq_encoder(**inputs)
        embeddings.append(outputs.last_hidden_state.cpu().numpy())
        torch.cuda.empty_cache()
 
    # Determine the most common shape on axis 1
    shapes = [emb.shape[1] for emb in embeddings]
    most_common_shape = max(set(shapes), key=shapes.count)
        
    # Modify embeddings to have the most common shape on axis 1
    for j in range(len(embeddings)):
        if embeddings[j].shape[1] != most_common_shape:
            new_embedding = np.zeros((embeddings[j].shape[0], most_common_shape, embeddings[j].shape[2]))
            slices = (slice(0, embeddings[j].shape[0]), slice(0, most_common_shape), slice(0, embeddings[j].shape[2]))
            new_embedding[slices] = embeddings[j][:, :most_common_shape, :]
            embeddings[j] = new_embedding

    return np.concatenate(embeddings, axis=0), counts

def save_embeddings_counts(embeddings, counts, filename):
    np.savez_compressed(filename, embeddings=embeddings, counts=counts)
    print(f"Saved embeddings and counts to {filename}")

def main():
    
    batch_size = 512
    data_path, dataset, save_path, model_name = sys.argv[1:]
    
    cache_dir = 'path_to_huggingface_model_cache' # Set your cache directory here. If None, it will use the default cache directory.
    embeddings, counts = get_embeddings(data_path, dataset, batch_size, model_name, cache_dir)

    filename = f'{save_path}/{dataset}/{dataset}_{taskID:02d}.npz'
    save_embeddings_counts(embeddings, counts, filename)

if __name__ == "__main__":
    main()   