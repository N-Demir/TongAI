import fastText

TRAIN_FILE_PATH = 'data/processed/fastTextInput_train.txt'
VALID_FILE_PATH = 'data/processed/fastTextInput_valid.txt'


def main():
	model = fastText.load_model("outputs/model_baseline_subwords.bin")
	total = 0
	correct = 0
	with open(VALID_FILE_PATH, 'r') as f:
		for line in f:
			Y, X = line.split(" ", 1)
			X = X.rstrip() # remove newline
			prediction = model.predict(X)[0][0]
			total += 1
			if Y == prediction:
				correct += 1

	acc = correct/total
	print("Accuracy is ", acc)

if __name__ == "__main__":
	main()