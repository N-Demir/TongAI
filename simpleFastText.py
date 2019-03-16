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
import data_loader
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
import matplotlib.pyplot as plt

EPOCH_SAVE = 10
EMBEDDING_DIM = 100
OUTPUT_DIM = 11
BATCH_SIZE = 64
N_EPOCHS = 200000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
REPOSITORY_NAME = 'TongAI'

CLASSES = ['mets', 'memory', 'multiple sclerosis', 'epilepsy', 'stereo/cyberknife', 'routine brain', 'sella', 'tumor brain', 'complex headache', 'brain trauma', 'stroke']

#SEED = 1234
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()

        hidden_size = 50
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim + 1, hidden_size)
        self.hidden = nn.Linear(hidden_size, output_dim)

        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x, age):
        
        # x = [sent len, batch size]
        # age = [batch size,]

        age = torch.reshape(age, (age.shape[0], 1))
        
        embedded = self.embedding(x)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)

        # print(embedded.shape)

        # assert(embedded.shape == (BATCH_SIZE, x.shape[0], EMBEDDING_DIM))
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]

        concat = torch.cat([pooled, age], dim=1)

        # assert(pooled.shape == (BATCH_SIZE, EMBEDDING_DIM))
        out = self.fc(concat)
        logits = self.hidden(out)

        return self.softmax(logits)

def train(model, iterator, optimizer, loss_function):
    
    epoch_loss = 0.
    epoch_acc = 0.

    epoch_class_correct = torch.zeros(OUTPUT_DIM) # Num classes
    epoch_class_counts = torch.zeros(OUTPUT_DIM)
    
    model.train()
    
    for batch_i, batch in enumerate(iterator):
        # print ("Starting Batch: %d" % batch_i)

        optimizer.zero_grad()

        # print("Input shape {}".format(batch.text.shape))

        x = batch.text
        age = batch.age
        
        logits = model(x, age).squeeze(1)
        
        # print("Logits shape {}".format(logits.shape))
        loss = loss_function(logits, Variable(batch.label.long()))
        
        acc = batch_accuracy(logits, batch.label)

        # print ("Batch %d accuracy: %f" % (batch_i, acc))
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # Class accuracies
        class_correct, class_counts = batch_class_accuracy(logits, batch.label)

        epoch_class_correct += class_correct
        epoch_class_counts += class_counts

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_class_correct, epoch_class_counts

def evaluate(model, iterator, loss_function):
    epoch_loss = 0.
    epoch_acc = 0.

    epoch_class_correct = torch.zeros(OUTPUT_DIM) # Num classes
    epoch_class_counts = torch.zeros(OUTPUT_DIM)
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:

            x = batch.text
            age = batch.age

            predictions = model(x, age).squeeze(1)
            
            loss = loss_function(predictions, Variable(batch.label.long()))
            
            acc = batch_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # Class accuracies
            class_correct, class_counts = batch_class_accuracy(predictions, batch.label)

            epoch_class_correct += class_correct
            epoch_class_counts += class_counts


    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_class_correct, epoch_class_counts

def batch_accuracy(preds, y):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == Variable(y.long())).float()
    return correct.sum() / len(correct)

def batch_class_accuracy(preds, y):
    class_correct = torch.zeros(OUTPUT_DIM)
    class_counts = torch.zeros(OUTPUT_DIM)
    preds = torch.argmax(preds, dim=1)
    for idx, pred in enumerate(preds):
        if pred == y[idx].long():
            class_correct[pred] += 1
        class_counts[pred] += 1

    return class_correct, class_counts

def main():
    # WTF is this supposed to do ?
    def generate_bigrams(x):
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

    # Load the data-set
    TEXT, train_itr, valid_itr = data_loader.load_data(generate_bigrams)

    input_dim = len(TEXT.vocab)

    model = FastText(vocab_size=input_dim, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM)
    
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    optimizer = optim.Adam(model.parameters())
    loss_function = nn.NLLLoss()

    # Allow for running on GPU
    model = model.to(DEVICE)
    loss_function = loss_function.to(DEVICE)

    best_valid_acc = 0
    best_train_stats = (None, None) # loss, acc
    best_valid_stats = (None, None, None, None) # loss, acc, class_acc, class_counts
    best_model = None
    best_epoch = 0

    try:
        for epoch in range(1, N_EPOCHS):

            train_loss, train_acc, train_class_correct, train_class_counts = train(model, train_itr, optimizer, loss_function)
            valid_loss, valid_acc, valid_class_correct, valid_class_counts = evaluate(model, valid_itr, loss_function)

            train_class_acc = train_class_correct / train_class_counts
            valid_class_acc = valid_class_correct / valid_class_counts

            if(valid_acc > best_valid_acc):
                best_valid_acc = valid_acc
                best_model = model
                best_epoch = epoch
                best_train_stats = (train_loss, train_acc)
                best_valid_stats = (valid_loss, valid_acc, valid_class_acc, valid_class_counts)

            
            print(f'| Epoch: {epoch:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% ')
            print()
            print('Val class accuracies: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(accuracy.item()) for idx, accuracy in enumerate(valid_class_acc) ] ))
            print('Val class counts: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(count.item()) for idx, count in enumerate(valid_class_counts) ] ))
            print()
            print()


    finally:
        train_loss, train_acc= best_train_stats
        valid_loss, valid_acc, valid_class_acc, valid_class_counts = best_valid_stats
        saveModel(best_model, best_epoch)
        plt.bar([klass for klass in CLASSES], [acc for acc in valid_class_acc], 1.0, color='#8F1500')
        axes = plt.gca()
        axes.set_ylim([0.01,1.0])
        plt.xticks(rotation='vertical')
        
        #print stats
        print(f'| Best Epoch: {best_epoch:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% ')
        print()
        print('Val class accuracies: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(accuracy.item()) for idx, accuracy in enumerate(valid_class_acc) ] ))
        print('Val class counts: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(count.item()) for idx, count in enumerate(valid_class_counts) ] ))
        print()
        print()

        #show plot
        plt.show()

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

if __name__ == '__main__':
    main()