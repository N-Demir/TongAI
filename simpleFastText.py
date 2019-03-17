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

With age + gender:
Best result was 79.59 so far

With the concat embeddings and age + gender discretized:
80.61

Just age
32 %

Just gender
28 % (it learns to just classify mets)

With fasttext 300d
81.79

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
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt
import sys

EPOCH_SAVE = 10
EMBEDDING_DIM = 300
OUTPUT_DIM = 11
BATCH_SIZE = 64
N_EPOCHS = 200000
DROPOUT_RATE = 0.5
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
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.embedding.weight.requires_grad = False

        self.fc = nn.Linear(3 * embedding_dim + 11 + 2, output_dim)
        # self.fc_2 = nn.Linear(3 * embedding_dim, 3 * embedding_dim)
        # self.fc_3 = nn.Linear(3 * embedding_dim, embedding_dim)
        # self.fc_4 = nn.Linear(embedding_dim, output_dim)


        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x, age, gender):
        
        # x = [sent len, batch size]
        # age = [batch size,]
        # gender = [batch size,]
        
        embedded = self.embedding(x)

        sent_len = x.size(0)

        nonzeros = (x != 1).sum(dim = 0)
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)

        # assert(embedded.shape == (BATCH_SIZE, x.shape[0], EMBEDDING_DIM))
        
        #embedded = [batch size, sent len, emb dim]

        # averaged ignoring pad elements
        summed = torch.sum(embedded, dim = 1)
        mean = torch.addcdiv(torch.zeros(EMBEDDING_DIM, x.size(1)), torch.t(summed), nonzeros.float())
        mean = torch.t(mean)
        #mean = [batch size, embedding_dim]
        
        maxed = F.max_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        #maxed = [batch size, embedding_dim]
        mined = -F.max_pool2d(-embedded, (embedded.shape[1], 1)).squeeze(1) #because there's no min pool
        #mined = [batch size, embedding_dim]
        pooled = torch.cat([maxed, mean, mined], dim = 1)
        #pooled = [batch size, 3* embedding_dim]

        concat = torch.cat([pooled, age, gender], dim=1)

        # assert(pooled.shape == (BATCH_SIZE, EMBEDDING_DIM))
        logits = self.fc(concat)
        # logits = F.dropout(logits, p = DROPOUT_RATE)
        # logits = F.relu(self.fc_2(logits))
        # logits = F.dropout(logits, p = DROPOUT_RATE)
        # logits = F.relu(self.fc_3(logits))
        # logits = F.dropout(logits, p = DROPOUT_RATE)
        # logits = self.fc_4(logits)

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
        gender = batch.gender
        
        logits = model(x, age, gender).squeeze(1)
        
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
            gender = batch.gender

            predictions = model(x, age, gender).squeeze(1)
            
            loss = loss_function(predictions, Variable(batch.label.long()))
            
            acc = batch_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # Class accuracies
            class_correct, class_counts = batch_class_accuracy(predictions, batch.label)

            epoch_class_correct += class_correct
            epoch_class_counts += class_counts


    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_class_correct, epoch_class_counts

def error_analysis(model, iterator):

    model.eval()
    y_preds = []
    labels = []
    for batch_i, batch in enumerate(iterator):
        x = batch.text
        age = batch.age
        gender = batch.gender

        predictions = model(x, age, gender).squeeze(1)
        y_pred = predictions.argmax(dim = 1).cpu().numpy()
        y_preds.append(y_pred)
        labels.append(batch.label.int().cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(y_preds)
    print(y_true.shape)
    print(y_pred.shape)
    print(batch.dataset.examples)
    plot_confusion_matrix(y_true, y_pred, CLASSES, normalize = True)
    plt.show()



# Taken from sklearn
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax        

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
        class_counts[y[idx].long()] += 1

    return class_correct, class_counts

def plot(class_acc):
    plt.cla()

    plt.bar([klass for klass in CLASSES], [acc for acc in class_acc], 1.0, color='#8F1500')
    axes = plt.gca()
    axes.set_ylim([0.01,1.0])
    plt.xticks(rotation='vertical')

    #show plot
    plt.draw()
    plt.pause(0.01)

def main():
    # WTF is this supposed to do ?
    # It generates bigrams dawgggg
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

    if len(sys.argv) == 2:
        model_path = sys.argv[1]
        model.load_state_dict(torch.load(model_path))
        error_analysis(model, valid_itr)
        return

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

            plot(valid_class_acc)

            
            print(f'| Epoch: {epoch:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% ')
            print()
            print('Val class accuracies: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(accuracy.item()) for idx, accuracy in enumerate(valid_class_acc) ] ))
            print('Val class counts: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(count.item()) for idx, count in enumerate(valid_class_counts) ] ))
            print()
            print()


    finally:
        train_loss, train_acc = best_train_stats
        valid_loss, valid_acc, valid_class_acc, valid_class_counts = best_valid_stats
        saveModel(best_model, best_epoch)

        plot(valid_class_acc)
        
        #print stats
        print(f'| Best Epoch: {best_epoch:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% ')
        print()
        print('Val class accuracies: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(accuracy.item()) for idx, accuracy in enumerate(valid_class_acc) ] ))
        print('Val class counts: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(count.item()) for idx, count in enumerate(valid_class_counts) ] ))
        print()
        print()
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