import torch
import torch.nn as nn
from FastTextEmbeddingBag import FastTextEmbeddingBag
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score

class FastText(nn.Module):
    def __init__(self, num_classes, hidden_size, model_path):
        super().__init__()
        
        self.embedding = FastTextEmbeddingBag(model_path)
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

			X.append(line.split(DELIMITER)[3])
			Y.append(line.split(DELIMITER)[2])

	return (X, Y)

def processX(X, max_len = 50):
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
	num_examples = len(Y)
	protocol_indices = {protocol: ind for ind, protocol in enumerate(set(Y))}
	num_classes = len(protocol_indices)
	Y_new = [protocol_indices[protocol] for protocol in Y]
	return num_classes, Y_new


def main():
	X,Y = readData("../data/processed/small_processed.csv")
	X_processed = processX(X)
	num_classes, Y_processed = processY(Y)
	Y_processed = torch.LongTensor(Y_processed)
	fastText = FastText(num_classes, num_classes * 2, "../outputs/embeddings/fasttext_train_embed.bin")
	criterion = nn.NLLLoss()
	optimizer = Adam(fastText.parameters())

	# training params
	num_epochs = 2000

	for epoch in range(num_epochs):
		optimizer.zero_grad()

		outputs = fastText(X_processed)
		loss = criterion(outputs, Y_processed)
		loss.backward()
		optimizer.step()
		if epoch % 10 == 0:
			predictions = torch.argmax(outputs, dim = -1)
			accuracy = accuracy_score(Y_processed.numpy(), predictions.numpy())
			print("epoch ", epoch, " loss: ", loss.item(), " accuracy: ", accuracy)

	outputs = fastText(X_processed)
	predictions = torch.argmax(outputs, dim = -1)
	accuracy = accuracy_score(Y_processed.numpy(), predictions.numpy())
	print("Final Accuracy: ", accuracy)

if __name__ == "__main__":
 	main()