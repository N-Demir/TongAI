import torch
import torch.nn as nn
from FastTextEmbeddingBag import FastTextEmbeddingBag
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score
from math import ceil
import sys
import os
import gc
import data_loader


MODEL_PATH = 'outputs/bestFastText_4l_retrnemb.pth.tar'
SECOND_MODEL_PATH = 'outputs/bestFastText_4l_retrnemb_2nd.pth.tar'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 100
BATCH_SIZE = 128
num_classes = 11



class FastText(nn.Module):
    def __init__(self, num_classes, hidden_size, model_path):
        super().__init__()
        
        self.embedding = FastTextEmbeddingBag(model_path, DEVICE)
        # self.embedding.weight.requires_grad = False

        self.fc_1 = nn.Linear(self.embedding.embedding_dim, hidden_size)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.relu_2 = nn.ReLU()
        self.fc_3 = nn.Linear(hidden_size, hidden_size)
        self.relu_3 = nn.ReLU()
        self.fc_4 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim = -1)

        # self.fc_contrast = nn.Linear(self.embedding.embedding_dim, 1)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        embedded = self.embedding(X)

        output = self.relu_1(self.fc_1(embedded))
        output = self.relu_2(self.fc_2(output))
        output = self.relu_3(self.fc_3(output))
        output = self.softmax(self.fc_4(output))
        # contrast = self.sigmoid(self.fc_contrast(avg))
        # output = torch.cat((classes_prob, contrast), dim = -1)
        return output

DELIMITER = '|'

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
	X_new = []
	for i, sentence in enumerate(X):
		words = data_loader.tokenizer(sentence)
		X_new.append(words)
	return X_new


def processY(Y):
	Y_new = [int(index.rstrip('\n')) for index in Y]
	num_classes = len(set(Y_new))
	return num_classes, torch.tensor(Y_new, dtype = torch.long, device = DEVICE)

# WTF is this supposed to do ?
def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def train(model_path):
	# WTF is this supposed to do ?
	def generate_bigrams(x):
		n_grams = set(zip(*[x[i:] for i in range(2)]))
		for n_gram in n_grams:
			x.append(' '.join(n_gram))
		return x


	X, Y = readData("data/processed/train.csv")
	X_eval, Y_eval = readData("data/processed/valid.csv") 
	X_processed = processX(X)
	X_eval = processX(X_eval)
	num_classes, Y_processed = processY(Y)
	_, Y_eval = processY(Y_eval)

	# model
	fastText = FastText(num_classes, num_classes * 2, "outputs/embeddings/fasttext_brain_embed.bin")

	if model_path != None:
		print("Loading model from ", model_path)
		fastText.load_state_dict(torch.load(model_path))
	criterion = nn.NLLLoss()
	#training with GPU
	fastText = fastText.to(DEVICE)
	criterion = criterion.to(DEVICE)

	optimizer = Adam(fastText.parameters())

	bestAccuracy = 0
	bestEpoch = 0
	num_train = len(X)
	for epoch in range(NUM_EPOCHS):
		print("Beginning epoch ", epoch)
		running_loss = 0
		fastText.train()
		numberCorrect = 0
		for i in range(ceil(num_train / BATCH_SIZE)):
			print("batch: ", i)
			beginIndex = i * BATCH_SIZE
			endIndex = min(beginIndex + BATCH_SIZE, num_train)
			X_input = X_processed[beginIndex: endIndex]	
			Y_input = Y_processed[beginIndex: endIndex]
			optimizer.zero_grad()

			outputs = fastText(X_input)

			loss = criterion(outputs, Y_input)
			loss.backward()
			optimizer.step()

			#calculate num correct
			predictions = torch.argmax(outputs, dim = -1)
			for i in range(len(predictions)):
				numberCorrect += int(predictions[i] == Y_input[i])

			running_loss += loss.item()
		accuracy = numberCorrect / len(Y)
		print("epoch ", epoch, " loss: ", running_loss, " train_accuracy: ", accuracy)

		# evaluation
		fastText.eval()

		numberCorrect = 0
		outputs = fastText(X_eval)

		loss = criterion(outputs, Y_eval)
		#calculate num correct
		predictions = torch.argmax(outputs, dim = -1)
		for i in range(len(predictions)):
			numberCorrect += int(predictions[i] == Y_eval[i])

		accuracy = numberCorrect / len(Y_eval)
		print("epoch ", epoch, " eval_loss: ", loss, " eval_accuracy: ", accuracy)

		if accuracy > bestAccuracy:
			if epoch != 0:
				os.rename(MODEL_PATH, SECOND_MODEL_PATH)
			torch.save(fastText.state_dict(), MODEL_PATH)
			bestAccuracy = accuracy
			bestEpoch = epoch
			gc.collect()
	print("Best accuracy: ", bestAccuracy, " on epoch ", bestEpoch)

def eval(model_path):
	fastText = FastText(num_classes, num_classes * 2, "outputs/embeddings/fasttext_brain_embed.bin")
	fastText.load_state_dict(torch.load(model_path))
	criterion = nn.NLLLoss()
	#evaluating with GPU
	fastText = fastText.to(DEVICE)
	criterion = criterion.to(DEVICE)

	# evaluation
	fastText.eval()

	numberCorrect = 0
	outputs = fastText(X_eval)

	loss = criterion(outputs, Y_eval)
	#calculate num correct
	predictions = torch.argmax(outputs, dim = -1)
	for i in range(len(predictions)):
		numberCorrect += int(predictions[i] == Y_eval[i])

	accuracy = numberCorrect / len(Y_eval)
	print("epoch ", epoch, " eval_loss: ", loss, " eval_accuracy: ", accuracy)



def main():
	if len(sys.argv) == 1:
		print("Provide an argument train or eval")
	else:
		mode = sys.argv[1]
		if mode == "train":
			if len(sys.argv) == 2:
				print("Provide a second argument new/continue")
			else:
				if sys.argv[2] == "continue":
					if len(sys.argv) == 3:
						print("Please add what model to train: first/second")
					elif sys.argv[3] == "first" or sys.argv[3] == "second":
						model_path = MODEL_PATH if sys.argv[3] == "first" else SECOND_MODEL_PATH
						train(model_path)
					else: 
						print("If continuing to train, only options are first/second")
				elif sys.argv[2] == "new":
					train(None)
				else:
					print("Train argument ", sys.argv[2], " not recognized.")
		elif mode == "eval":
			if len(sys.argv) == 2:
				print("Provide what model to evaluate (first/second)")
			else:
				if sys.argv[2] == "first" or sys.argv[2] == "second":
					model_path = MODEL_PATH if sys.argv[2] == "first" else SECOND_MODEL_PATH
					eval(model_path)
				else:
					print("Can only evaluate the first or second model")
		else:
			print("Mode ", mode, " not recognized.")
	

if __name__ == "__main__":
	main()