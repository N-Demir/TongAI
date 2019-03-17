import spacy
from torchtext.data import Field, LabelField
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from torchtext import datasets
import torch
from torch.nn import functional as F
import random
import numpy as np
import math

# English language model
# Though we should play potentially with different
# language models
spacy_en = spacy.load('en')

train_path = 'data/processed/train.csv'
valid_path =  'data/processed/valid.csv'
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

def genderToNum(gender):
    assert(gender == 'm' or gender == 'f')
    return [1, 0] if gender == 'f' else [0, 1]

def ageToFloat(age):
    # # Divide by 100 to sort of normalize it
    # return float(age) / 100
    arr = np.zeros(11)
    arr[math.floor(float(age) / 10)] = 1
    return arr



def load_data(preprocessing=None):

    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True, preprocessing=preprocessing)
    AGE = Field(sequential = False, use_vocab = False, preprocessing = ageToFloat, dtype = torch.float)
    GENDER = Field(sequential = False, use_vocab = False, preprocessing = genderToNum, dtype = torch.float)
    LABEL = LabelField(dtype=torch.float)

    train_data = TabularDataset(
        path=train_path, format='csv',
        fields=[('age', AGE), ('gender', GENDER), ('text', TEXT), ('label', LABEL)],
        csv_reader_params={'delimiter':"|", 'quotechar': "\""})

    valid_data = TabularDataset(
        path=valid_path, format='csv',
        fields=[('age', AGE), ('gender', GENDER), ('text', TEXT), ('label', LABEL)],
        csv_reader_params={'delimiter':"|", 'quotechar': "\""})

    print ('Size of train set: ' + str(len(train_data.examples)))
    print ('Size of val / test: ' + str(len(valid_data.examples)))


    TEXT.build_vocab(train_data, max_size=250000, vectors="fasttext.en.300d")
    LABEL.build_vocab(train_data)

    print ('Size of the vocab: ' + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_itr, valid_itr = BucketIterator.splits((train_data, valid_data),
        batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.text))

    return TEXT, train_itr, valid_itr