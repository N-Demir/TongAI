from sklearn.model_selection import train_test_split
import sys

FILE_PATH = 'data/processed/final.csv'
DELIMITER = '||'

TRAIN_FILE_PATH = 'data/processed/fastTextInput_train.txt'
VALID_FILE_PATH = 'data/processed/fastTextInput_valid.txt'
LABEL_TAG = "__label__"

def readInData(file_path):
	# Let's dejsonify
	X = []
	Y = []

	with open(file_path, 'r') as f:
		for line in f:
			assert(len(line.split(DELIMITER)) == 4)

			X.append(line.split(DELIMITER)[3])
			Y.append(line.split(DELIMITER)[2])

	return (X, Y)

def saveToNewFile(X, Y, file_path):
	numLines = len(X)
	assert(len(X) == len(Y))

	with open(file_path, 'w') as f:
		for i in range(numLines):
			f.write('%s%s %s' % (LABEL_TAG, Y[i], X[i]))


def main():
	print("Reading in data")
	X,Y = readInData(FILE_PATH)
	print("Read ", len(X), " examples")

	X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
	print("Creating train file with ", len(X_train), " examples.")
	saveToNewFile(X_train, Y_train, TRAIN_FILE_PATH)
	print("Creating valid file with ", len(X_valid), " examples.")
	saveToNewFile(X_valid, Y_valid, VALID_FILE_PATH)

def _remove_label(source_file_path):
	target_file_path = source_file_path[:-4] + "_noLabel.txt"
	
	with open(source_file_path, 'r') as source:
		with open(target_file_path, 'w') as target:
			for line in source:
				label, text = line.split(" ", 1)
				target.write(text)

def remove_label():
	source_file_path = TRAIN_FILE_PATH 
	_remove_label(source_file_path)


if __name__ == "__main__":
	if len(sys.argv) > 1:
		if sys.argv[1] != "remove_label":
			print("argument not supported")
		else:
			remove_label()
	else: 
		main()