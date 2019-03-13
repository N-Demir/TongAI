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

EPOCH_SAVE = 10
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
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
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        embedded = self.embedding(x)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)

def train(model, iterator, optimizer, loss_function):
    
    epoch_loss = 0.
    epoch_acc = 0.
    epoch_real_prec = 0.
    epoch_fake_prec = 0.
    epoch_real_recall = 0.
    epoch_fake_recall = 0.
    epoch_real_f_score = 0.
    epoch_fake_f_score = 0.
    #epoch_roc = 0
    
    model.train()
    
    for batch_i, batch in enumerate(iterator):
        print ("Starting Batch: %d" % batch_i)

        optimizer.zero_grad()
        
        logits = model(batch.text).squeeze(1)
        
        loss = loss_function(logits, batch.label)
        
        acc = batch_accuracy(logits, batch.label)
        precisions, recalls, f1_scores = batch_precision_recall_f_score(logits, batch.label)
        #roc_score = batch_roc_score(logits, batch.label)

        print ("Batch %d accuracy: %f" % (batch_i, acc))
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_real_prec += precisions[0]
        epoch_fake_prec += precisions[1]
        epoch_real_recall += recalls[0]
        epoch_fake_recall += recalls[1]
        epoch_real_f_score += f1_scores[0]
        epoch_fake_f_score += f1_scores[1]
        #epoch_roc += roc_score
    
    avg_prec = [epoch_real_prec / len(iterator), epoch_fake_prec / len(iterator)]
    avg_recall = [epoch_real_recall / len(iterator),  epoch_fake_recall / len(iterator)]
    avg_f_score = [epoch_real_f_score / len(iterator),  epoch_fake_f_score / len(iterator)]
    return epoch_loss / len(iterator), epoch_acc / len(iterator), avg_prec, avg_recall, avg_f_score #, epoch_roc / len(iterator)

def evaluate(model, iterator, loss_function):
    epoch_loss = 0.
    epoch_acc = 0.
    epoch_real_prec = 0.
    epoch_fake_prec = 0.
    epoch_real_recall = 0.
    epoch_fake_recall = 0.
    epoch_real_f_score = 0.
    epoch_fake_f_score = 0.
    #epoch_roc = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = loss_function(predictions, batch.label)
            
            acc = batch_accuracy(predictions, batch.label)
            precisions, recalls, f1_scores = batch_precision_recall_f_score(predictions, batch.label)
            #roc_score = batch_roc_score(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_real_prec += precisions[0]
            epoch_fake_prec += precisions[1]
            epoch_real_recall += recalls[0]
            epoch_fake_recall += recalls[1]
            epoch_real_f_score += f1_scores[0]
            epoch_fake_f_score += f1_scores[1]
            #epoch_roc += roc_score
    
    avg_prec = [epoch_real_prec / len(iterator), epoch_fake_prec / len(iterator)]
    avg_recall = [epoch_real_recall / len(iterator),  epoch_fake_recall / len(iterator)]
    avg_f_score = [epoch_real_f_score / len(iterator),  epoch_fake_f_score / len(iterator)]
    return epoch_loss / len(iterator), epoch_acc / len(iterator), avg_prec, avg_recall, avg_f_score  #, epoch_roc / len(iterator)

def batch_accuracy(preds, y):
    # Pass through sigmoid with decision bound at 0.5
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

def batch_precision_recall_f_score(preds, y):
    y_pred = torch.round(torch.sigmoid(preds))
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y.detach().cpu().numpy()

    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred)
    # In the case that y passed in is all one class, then we still want to return array with 2 elems
    if len(precisions) != 2:
        # If this happens because y is not all of one type then we have an issueeee
        for a in y:
            if y[0] != a: raise Exception('WHAT THE FUUUUUUCK')
        if y[0] == 0:
            return np.concatenate([precisions, [0.]]), np.concatenate([recalls, [0.]]), np.concatenate([f1_scores, [0.]])
        else:
            return np.concatenate([[0.], precisions]), np.concatenate([[0.], recalls]), np.concatenate([[0.], f1_scores])
    return precisions, recalls, f1_scores


def batch_roc_score(preds, y):
    y_pred = torch.round(torch.sigmoid(preds))
    y_pred = y_pred.detach().numpy()
    y_true = y.detach().numpy() 
    roc_score = roc_auc_score(y_true, y_pred)

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
    loss_function = nn.BCEWithLogitsLoss()

    # Allow for running on GPU
    model = model.to(DEVICE)
    loss_function = loss_function.to(DEVICE)

    for epoch in range(1, N_EPOCHS):

        train_loss, train_acc, train_prec, train_recall, train_f_score = train(model, train_itr, optimizer, loss_function)
        valid_loss, valid_acc, valid_prec, valid_recall, valid_f_score = evaluate(model, valid_itr, loss_function)
        
        print(f'| Epoch: {epoch:02} | Train Loss: {train_loss:.3f}| Train Acc: {train_acc*100:.2f}% | Train f1_score: {train_f_score}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Val. f1_score: {valid_f_score}%')
        saveMetrics('train', train_acc, train_loss, train_prec, train_recall, train_f_score, epoch)
        saveMetrics('valid', valid_acc, valid_loss, valid_prec, valid_recall, valid_f_score, epoch)
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