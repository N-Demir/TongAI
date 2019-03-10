import torch
import torch.nn as nn
from FastTextEmbeddingBag import FastTextEmbeddingBag
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score
from math import ceil
import sys

MODEL_PATH = '../outputs/bestFastText.pth.tar'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FastText(nn.Module):
    def __init__(self, num_classes, hidden_size, model_path):
        super().__init__()
        
        self.embedding = FastTextEmbeddingBag(model_path, DEVICE)
        self.embedding.weight.requires_grad = False

        self.fc_1 = nn.Linear(self.embedding.embedding_dim, hidden_size)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim = -1)

        # self.fc_contrast = nn.Linear(self.embedding.embedding_dim, 1)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        embeded = self.embedding(X)
        avg = torch.mean(embeded, dim = 1) #average each sentence embeddings
        output = self.softmax(self.fc_2(self.relu_1(self.fc_1(avg))))
        # contrast = self.sigmoid(self.fc_contrast(avg))
        # output = torch.cat((classes_prob, contrast), dim = -1)
        return output

DELIMITER = '||'
def readData(file_path):
	# Let's dejsonify
	X = []
	Y = []

	with open(file_path, 'r') as f:
		for line in f:
			assert(len(line.split(DELIMITER)) == 4)

			X.append(line.split(DELIMITER)[2])
			Y.append(line.split(DELIMITER)[3])

	return (X, Y)

def processX(X, max_len = 100):
	num_sentences = len(X)
	X_new = [['<pad>' for i in range(max_len)] for j in range(num_sentences)]
	for i, sentence in enumerate(X):
		assert(sentence == X[i])
		words = sentence.replace("\n", "").split()
		numWords = len(words)
		final_ind = min(numWords, max_len)
		X_new[i][:final_ind] = words[:final_ind]
	return X_new

def processY(Y):
	Y_new = [int(index.rstrip('\n')) for index in Y]
	num_classes = len(set(Y_new))
	return num_classes, torch.tensor(Y_new, dtype = torch.long, device = DEVICE)


def main(continue_training):
	# get Data
	X, Y = readData("../data/processed/processed_train.csv")
	X_eval, Y_eval = readData("../data/processed/processed_test.csv") 
	X_processed = processX(X)
	X_eval = processX(X_eval)
	num_classes, Y_processed = processY(Y)
	_, Y_eval = processY(Y_eval)

	num_train = len(X)

	fastText = FastText(num_classes, num_classes * 2, "../outputs/embeddings/fasttext_brain_embed.bin")
	if continue_training:
		print("Loading model from ", MODEL_PATH)
		fastText.load_state_dict(torch.load(MODEL_PATH))
	criterion = nn.NLLLoss()
	optimizer = Adam(fastText.parameters())

	#training with GPU
	fastText = fastText.to(DEVICE)
	criterion = criterion.to(DEVICE)

	# training params
	num_epochs = 100
	batch_size = 128

	bestAccuracy = 0
	bestEpoch = 0
	for epoch in range(num_epochs):
		print("Beginning epoch ", epoch)
		running_loss = 0
		fastText.train()
		for i in range(ceil(num_train / batch_size)):
			beginIndex = i* batch_size
			endIndex = min(beginIndex + batch_size, num_train)
			X_input = X_processed[beginIndex: endIndex]	
			Y_input = Y_processed[beginIndex: endIndex]
			optimizer.zero_grad()

			outputs = fastText(X_input)
			loss = criterion(outputs, Y_input)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print("epoch ", epoch, " loss: ", running_loss)
		fastText.eval()
		outputs = fastText(X_eval)
		loss = criterion(outputs, Y_eval)
		predictions = torch.argmax(outputs, dim = -1)
		accuracy = accuracy_score(Y_eval.numpy(), predictions.numpy())
		print("epoch ", epoch, " eval_loss: ", loss.item(), " eval_accuracy : ", accuracy)
		if accuracy > bestAccuracy:
			torch.save(fastText.state_dict(), MODEL_PATH)
			bestAccuracy = accuracy
			bestEpoch = epoch
	print("Best accuracy: ", bestAccuracy, " on epoch ", bestEpoch)


if __name__ == "__main__":
	continue_training = False
	if len(sys.argv) > 1:
		if sys.argv[1] == 'continue':
			continue_training = True
		elif sys.argv[1] != 'new':
			print("Arg ", sys.argv[1], " not supported. Please use continue or new")
	else:
		print("Provide argument continue or new")
	main(continue_training)