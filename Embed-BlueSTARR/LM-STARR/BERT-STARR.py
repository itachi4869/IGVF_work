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

def log(x):
    return torch.log(x)

def logGam(x):
    return torch.lgamma(x)

def logLik(sumX, numX, Yj, logTheta, alpha, beta, NUM_RNA):
    #n = tf.shape(sumX)[0]
    n = sumX.shape[0]
    #sumX = tf.tile(tf.reshape(sumX,[n,1]),[1,NUM_RNA])
    sumX = torch.tile(torch.reshape(sumX, [n, 1]), [1, NUM_RNA])
    #theta = tf.math.exp(logTheta) # assume the model is predicting log(theta)
    theta = torch.exp(logTheta) # assume the model is predicting log(theta)
    LL = (sumX + alpha) * log(beta + numX) + logGam(Yj + sumX + alpha) + Yj * log(theta) \
        -logGam(sumX + alpha) - logGam(Yj + 1) - (Yj + sumX + alpha) * log(theta + beta + numX)
    return LL

def customLoss(y_true, y_pred):
    EPSILON = torch.tensor(1e-10)
    
    #y_true = torch.tensor(y_true)
    #y_pred = torch.tensor(y_pred)

    NUM_DNA = 3
    NUM_RNA = 3
    #NUM_DNA=(int(y_true.shape[1]/2))
    #NUM_RNA=(int(y_true.shape[1]/2))
    DNA = y_true[:, 0:NUM_DNA]
    RNA = y_true[:, NUM_DNA:]
    #DNA=y_true[:,0:(int(y_true.shape[1]/2))]
    #RNA=y_true[:,(int(y_true.shape[1]/2)):]
    sumX = torch.sum(DNA, dim=1)
    #sumX=tf.reduce_sum(DNA,axis=1)
    
    LL = -logLik(sumX, NUM_DNA, RNA, y_pred, EPSILON, EPSILON, NUM_RNA)
    return torch.sum(LL, dim=1)#tf.reduce_sum(LL,axis=1)

class BlueBERT(nn.Module):
    def __init__(self, NumKernels, KernelSizes, cache_dir=None):
        super(BlueBERT, self).__init__()

        # DNABERT model/embeddings
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M",
                                            cache_dir=cache_dir)
        pretrained_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", 
                                                     trust_remote_code=True, config=config,
                                                     cache_dir=cache_dir)
        
        '''
        # NT model/embeddings
        config = BertConfig.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                                            cache_dir=cache_dir)
        pretrained_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", 
                                                     trust_remote_code=True, config=config,
                                                     cache_dir=cache_dir)
        '''

        self.dim = config.hidden_size
        self.model = pretrained_model.requires_grad_(False) # Probe, not finetune

        # CNN layers, same settings as in the original BlueSTARR
        self.cnns = nn.ModuleList()
        for i, (NumKer, KerSize) in enumerate(zip(NumKernels, KernelSizes)):
            if i > 0: 
                self.cnns.append(nn.Dropout(0.5))
                self.cnns.append(nn.Conv1d(
                    in_channels=NumKernels[i-1],
                    out_channels=NumKer,
                    kernel_size=KerSize,
                    stride=1,
                    padding='same'
                ))
            elif i == 0:
                self.cnns.append(nn.Conv1d(
                    in_channels=1,
                    out_channels=NumKer,
                    kernel_size=KerSize,
                    stride=1,
                    padding='same'
                ))
            self.cnns.append(nn.BatchNorm1d(NumKer))
            self.cnns.append(nn.ReLU())
        self.pooling = nn.AvgPool1d(NumKernels[-1])
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.dim, 1)
    
    def forward(self, input_ids=None, token_type_ids=None):

        x = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0]
        #x = x[:, 0, :].view(-1, 1, self.dim) # only use the first token
        x = torch.mean(x, dim=1) # average over all tokens
        x = x.view(-1, 1, self.dim)
        for layer in self.cnns: x = layer(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear1(x)

        return x

class SequenceData(Dataset):

    def __init__(self, encodings, counts):

        self.encodings = encodings
        self.counts = counts

    def __getitem__(self, index):
        result = {}
        result['encodings'] = {key: val[index].clone().detach() for key, val in self.encodings.items()}
        result['counts'] = torch.as_tensor(self.counts[index])
        return result

    def __len__(self):
        return len(self.counts)

def load_data(dataset='train'):
    sequences = []
    dir = os.path.dirname(os.path.realpath(__file__))
    with gzip.open('%s/../../STARR-data/%s.fasta.gz'%(dir, dataset),'rb') as f:
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
    with gzip.open('%s/../../STARR-data/%s-counts.txt.gz'%(dir, dataset),'rb') as f:
        cur_seq = ''
        for line in f:
            if cur_line == 0: 
                cur_line += 1
                continue
            line = line.decode('utf-8')
            counts.append(list(map(int, line.split())))
            cur_line += 1
    #sequences = sequences[0:500000]
    #counts = counts[0:500000]
    print('%s length: %s'%(dataset, len(sequences)))
    #print(sequences[0:10])
    #print(counts[0:10])
    #print(sequences[-1])
    #print(counts[-1])
    return sequences, counts

def encoded_data(dataset, tokenizer):
    sequences, counts = load_data(dataset)
    encodings = tokenizer(sequences, padding=True, return_tensors='pt')
    print(len(sequences),flush=True)
    sequences = None
    gc.collect()
    data = SequenceData(encodings=encodings, counts=counts)
    encodings = None
    gc.collect()
    data_loader = DataLoader(data, batch_size=32, shuffle=True)
    data = None
    gc.collect()

    print('here')
    print(len(data_loader))

    return data_loader

def naiveCorrelation(y_true, y_pred):
    NUM_DNA = 3

    DNA = y_true[:, 0:NUM_DNA] + 1
    RNA = y_true[:, NUM_DNA:] + 1
    #DNA=y_true[:,0:(int(y_true.shape[1]/2))]+1
    #RNA=y_true[:,(int(y_true.shape[1]/2)):]+1
    sumX = tf.reduce_sum(DNA, axis=1)
    sumY = tf.reduce_sum(RNA, axis=1)
    naiveTheta = sumY / sumX
    print("Y true shape")
    print(naiveTheta.shape)
    print("naiveTheta=", naiveTheta)
    print("mean naiveTheta=", tf.math.reduce_mean(naiveTheta))
    print("sd naiveTheta=", tf.math.reduce_std(naiveTheta))
    print("y_pred=", tf.math.exp(y_pred.squeeze()))
    print("mean y_pred=", tf.math.reduce_mean(tf.math.exp(y_pred.squeeze())))
    print("sd y_pred=", tf.math.reduce_std(tf.math.exp(y_pred.squeeze())))
    print("MSE=", (np.square(naiveTheta-tf.math.exp(y_pred.squeeze())).mean(axis=None)))
    print("Pearson cor=", stats.pearsonr(tf.math.exp(y_pred.squeeze()),naiveTheta))
    cor=stats.spearmanr(tf.math.exp(y_pred.squeeze()),naiveTheta)
    print("Spearman cor=", cor)
    return cor

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    cache_dir = None # Set to None to use the default cache directory
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M", 
                                        cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", 
                                              trust_remote_code=True, config=config, 
                                              cache_dir=cache_dir)
    
    #config = BertConfig.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", cache_dir=cache_dir)
    #tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", 
    #                                          trust_remote_code=True, config=config, cache_dir=cache_dir)

    NumKernels = [1024, 512, 256, 128, 64]
    #NumKernels = [512, 256, 128, 64, 32]
    KernelSizes = [8, 16, 32, 64, 128]
    model = BlueBERT(NumKernels, KernelSizes).to(device)

    print(model)

    train_loader = encoded_data('train', tokenizer)
    gc.collect()
    valid_sequences, validation_counts = load_data('validation')
    gc.collect()
    test_sequences, test_counts = load_data('test')
    gc.collect()

    #epochs = 25
    epochs = 1
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # count the number of batches in the training set
    num_batches = 0
    for batch in train_loader:
        num_batches += 1
    print("Number of batches: ", num_batches)

    model.train()
    for epoch in range(epochs):
        
        for batch in train_loader:
            start_time = time.time()
            labels = batch['counts'].to(device).to(torch.float).clone().detach().requires_grad_(True)
            input_ids = batch['encodings']["input_ids"].to(device)
            token_type_ids = batch['encodings']["token_type_ids"].to(device)
            output = model(input_ids=input_ids, token_type_ids=token_type_ids)
            labels = torch.tensor(labels[:,0]/labels[:,1])
            loss = nn.functional.mse_loss(output.view(-1), labels.view(-1))
            #loss = torch.sum(customLoss(labels,output))

            optimizer.zero_grad()
            loss.backward()
            
            # update weights
            optimizer.step()
            #print(torch.mean(loss))
            print("Time taken for one batch: ", time.time() - start_time)
            return None

    model.eval()
    test_tokens = tokenizer(test_sequences, padding=True, return_tensors = 'pt')
    input_ids = test_tokens["input_ids"].to(device)
    token_type_ids = test_tokens["token_type_ids"].to(device)
    batch_size = 512
    test_predictions = []
    for i in range(int(np.ceil(len(input_ids)/batch_size))):
        input_ids_cur = input_ids[i*batch_size:(i+1)*batch_size]
        token_type_ids_cur = token_type_ids[i*batch_size:(i+1)*batch_size]
        test_predictions_cur = model(input_ids=input_ids_cur, token_type_ids=token_type_ids_cur).cpu().detach().numpy().tolist()
        #test_predictions_cur = model(input_ids=input_ids_cur).cpu().detach().numpy().tolist()
        test_predictions += test_predictions_cur

    #test_predictions = model(input_ids=test_tokens['input_ids']).logits.cpu().detach().numpy()
    naiveCorrelation(y_true=np.array(test_counts).astype('float32'), y_pred=np.array(test_predictions).astype('float32'))

if __name__ == '__main__':
    train()
