"""
Results:


With embed_dim = 100 and 25002 in vocabulary:

| Epoch: 01 | Train Loss: 2.145| Train Acc: 27.77% | Val. Loss: 1.868 | Val. Acc: 28.59% 
| Epoch: 02 | Train Loss: 1.947| Train Acc: 28.32% | Val. Loss: 1.622 | Val. Acc: 36.41% 
| Epoch: 03 | Train Loss: 1.834| Train Acc: 32.08% | Val. Loss: 1.407 | Val. Acc: 52.10% 
| Epoch: 04 | Train Loss: 1.683| Train Acc: 44.07% | Val. Loss: 1.251 | Val. Acc: 60.42% 
| Epoch: 05 | Train Loss: 1.531| Train Acc: 51.75% | Val. Loss: 1.147 | Val. Acc: 66.49% 
| Epoch: 06 | Train Loss: 1.380| Train Acc: 59.22% | Val. Loss: 1.086 | Val. Acc: 70.12% 
| Epoch: 07 | Train Loss: 1.245| Train Acc: 65.82% | Val. Loss: 1.088 | Val. Acc: 72.57% 
| Epoch: 08 | Train Loss: 1.126| Train Acc: 70.07% | Val. Loss: 1.109 | Val. Acc: 74.41% 
| Epoch: 09 | Train Loss: 1.019| Train Acc: 74.28% | Val. Loss: 1.142 | Val. Acc: 75.41% 
| Epoch: 10 | Train Loss: 0.923| Train Acc: 76.97% | Val. Loss: 1.186 | Val. Acc: 75.82% 
| Epoch: 11 | Train Loss: 0.842| Train Acc: 79.29% | Val. Loss: 1.231 | Val. Acc: 76.12% 
| Epoch: 12 | Train Loss: 0.779| Train Acc: 80.67% | Val. Loss: 1.278 | Val. Acc: 76.95% 
| Epoch: 13 | Train Loss: 0.729| Train Acc: 82.21% | Val. Loss: 1.323 | Val. Acc: 77.28% 
| Epoch: 14 | Train Loss: 0.670| Train Acc: 83.55% | Val. Loss: 1.371 | Val. Acc: 77.15% 
| Epoch: 15 | Train Loss: 0.625| Train Acc: 84.40% | Val. Loss: 1.412 | Val. Acc: 77.43% 
| Epoch: 16 | Train Loss: 0.586| Train Acc: 85.46% | Val. Loss: 1.455 | Val. Acc: 77.96% 
| Epoch: 17 | Train Loss: 0.547| Train Acc: 86.08% | Val. Loss: 1.496 | Val. Acc: 77.98% 
| Epoch: 18 | Train Loss: 0.519| Train Acc: 87.27% | Val. Loss: 1.542 | Val. Acc: 78.36% 
| Epoch: 19 | Train Loss: 0.482| Train Acc: 88.13% | Val. Loss: 1.589 | Val. Acc: 77.83% 
| Epoch: 20 | Train Loss: 0.456| Train Acc: 88.70% | Val. Loss: 1.639 | Val. Acc: 77.60% 

"""

import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import numpy as np

EPOCH_SAVE = 10
EMBEDDING_DIM = 100
OUTPUT_DIM = 11
BATCH_SIZE = 64
N_EPOCHS = 200000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
REPOSITORY_NAME = 'TongAI'

#SEED = 1234
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        embedded = self.embedding(x)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)

        # print(embedded.shape)

        # assert(embedded.shape == (BATCH_SIZE, x.shape[0], EMBEDDING_DIM))
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
        # assert(pooled.shape == (BATCH_SIZE, EMBEDDING_DIM))
        logits = self.fc(pooled)

        return self.softmax(logits)

def train(model, iterator, optimizer, loss_function):
    
    epoch_loss = 0.
    epoch_acc = 0.
    
    model.train()
    
    for batch_i, batch in enumerate(iterator):
        # print ("Starting Batch: %d" % batch_i)

        optimizer.zero_grad()

        # print("Input shape {}".format(batch.text.shape))
        
        logits = model(batch.text).squeeze(1)
        
        # print("Logits shape {}".format(logits.shape))
        loss = loss_function(logits, Variable(batch.label.long()))
        
        acc = batch_accuracy(logits, batch.label)

        # print ("Batch %d accuracy: %f" % (batch_i, acc))
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, loss_function):
    epoch_loss = 0.
    epoch_acc = 0.
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = loss_function(predictions, Variable(batch.label.long()))
            
            acc = batch_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator) 

def batch_accuracy(preds, y):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == Variable(y.long())).float()
    return correct.sum() / len(correct)

def main():
    # WTF is this supposed to do ?
    def generate_bigrams(x):
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

    # Load the data-set
    TEXT, train_itr, valid_itr, test_itr = load_data(generate_bigrams)

    input_dim = len(TEXT.vocab)

    model = FastText(vocab_size=input_dim, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM)
    
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    optimizer = optim.Adam(model.parameters())
    loss_function = nn.NLLLoss()

    # Allow for running on GPU
    model = model.to(DEVICE)
    loss_function = loss_function.to(DEVICE)

    for epoch in range(1, N_EPOCHS):

        train_loss, train_acc = train(model, train_itr, optimizer, loss_function)
        valid_loss, valid_acc = evaluate(model, valid_itr, loss_function)
        
        print(f'| Epoch: {epoch:02} | Train Loss: {train_loss:.3f}| Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% ')
        # saveMetrics('train', train_acc, train_loss, train_prec, train_recall, train_f_score, epoch)
        # saveMetrics('valid', valid_acc, valid_loss, valid_prec, valid_recall, valid_f_score, epoch)
        if (epoch % EPOCH_SAVE == 0):
            saveModel(model, epoch)

    saveModel(model, epoch)

def setupCheckpoints():
    def get_repository_path():
        """ 
        Returns the path of the project repository

        Uses the global REPOSITORY_NAME constant and searches through parent directories
        """
        p = Path(__file__).absolute().parents
        for parent in p:
            if parent.name == REPOSITORY_NAME:
                return parent

    p = get_repository_path()
    checkpoints_folder = p / 'checkpoints'
    Fast_Text_folder = checkpoints_folder / 'FastText'
    cur_folder = Fast_Text_folder / CURRENT_TIME

    
    checkpoints_folder.mkdir(exist_ok=True)
    Fast_Text_folder.mkdir(exist_ok=True)
    cur_folder.mkdir(exist_ok=True)

    return cur_folder

def saveModel(model, epoch):
    path = setupCheckpoints()
    model_folder = path / 'models'
    model_folder.mkdir(exist_ok=True)

    model_path = model_folder / '{:02d}-model.pt'.format(epoch)

    state = model.state_dict()
    torch.save(state, model_path)

def saveMetrics(prefix, accuracy, loss, precisions, recalls, f1_scores, epoch):
    real_precision, fake_precision = precisions
    real_recall, fake_recall = recalls
    real_f1_score, fake_f1_score = f1_scores

    path = setupCheckpoints()

    accuracy_path = path / '{}-accuracy.txt'.format(prefix)
    loss_path = path / '{}-loss.txt'.format(prefix)

    real_precision_path = path / '{}-real-precision.txt'.format(prefix)
    fake_precision_path = path / '{}-fake-precision.txt'.format(prefix)
    real_recall_path = path / '{}-real-recall.txt'.format(prefix)
    fake_recall_path = path / '{}-fake-recall.txt'.format(prefix)
    real_f1_score_path = path / '{}-real-f1_score.txt'.format(prefix)
    fake_f1_score_path = path / '{}-fake-f1_score.txt'.format(prefix)

    def writeMetric(metric_path, value):
        with open(metric_path, 'a+') as f:
            f.write('{},{}\n'.format(epoch, value))

    writeMetric(accuracy_path, accuracy)
    writeMetric(loss_path, loss)
    writeMetric(real_precision_path, real_precision)
    writeMetric(fake_precision_path, fake_precision)
    writeMetric(real_recall_path, real_recall)
    writeMetric(fake_recall_path, fake_recall)
    writeMetric(real_f1_score_path, real_f1_score)
    writeMetric(fake_f1_score_path, fake_f1_score)

###### DATA LOADING STUFF


import spacy
from torchtext.data import Field, LabelField
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from torchtext import datasets
import torch
from torch.nn import functional as F
import random

# English language model
# Though we should play potentially with different
# language models
spacy_en = spacy.load('en')

path = 'data/processed/final.csv'
BATCH_SIZE = 64

# Do this for testing
# To see if we match the results from online
#SEED = 1234
SEED = 229
TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]

# Sets the random number generator of torch
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
# May want to play with this for reproducability
#torch.backends.cudnn.deterministic = True

def tokenizer(text): # create a tokenizer function
    # Consider using tok.lemma to lemmatize the
    # vocabulary rather than true words

    # We can also consider removing stop words!!

    # Consider appending potentially named entity tags??
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Note that now everything is tsv but would like json!!
def load_data(preprocessing=None):
    # Fields for the dataset
    # The actual review message

    #TEXT = Field(tokenize='spacy') # -- Old way, unclear exactly what language model is used
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True, preprocessing=preprocessing)
    LABEL = LabelField(dtype=torch.float)

    # Get the entire dataset that we will then split
    data = TabularDataset(
        path=path, format='csv',
        fields=[('age', None), ('gender', None), ('text', TEXT), ('label', LABEL)],
        csv_reader_params={'delimiter':"|", 'quotechar': "\""})

    # We should probabily look at the proportion of fake to non fake in each of these
    # set to make sure it is fairly even. Though probabilistically it should be I suppose
    train_data, valid_data, test_data = data.split(split_ratio=TRAIN_VAL_TEST_SPLIT, random_state=random.seed(SEED))
    #valid_data, test_data = test_data.split(split_ratio=VAL_TEST_SPLIT, random_state=random.seed(SEED))

    print ('Size of train set: ' + str(len(train_data.examples)))
    print ('Size of val / test: ' + str(len(valid_data.examples)))

    '''
    # Try loading in the IMB dataset to label pos or negative
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    # Get train/valid split!!
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    '''

    # Now we need to build the vocab for our actual data
    # Here we will use the pre-trained word vetors from "glove.6b.100"
    TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
    LABEL.build_vocab(train_data)

    # Print stuff for sanity checks
    print ('Size of the vocab: ' + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_itr, valid_itr, test_itr = BucketIterator.splits((train_data, valid_data, test_data),
        batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.text))

    return TEXT, train_itr, valid_itr, test_itr

if __name__ == '__main__':
    main()