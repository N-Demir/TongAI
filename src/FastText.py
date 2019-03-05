import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import numpy as np
from FastTextEmbeddingBag import FastTextEmbeddingBag

EMBEDDING_DIM = 300
OUTPUT_DIM = 10 + 1 # protocols + 1 for w/o contrast

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, model_path):
        super().__init__()
        
        self.embedding = nn.FastTextEmbeddingBag(model_path)
        self.pool = nn.AvgPool1d()
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        embedded = self.embedding(x)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)