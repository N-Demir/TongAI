# Random got accuracy of 0.1686 haha

import numpy as np
import data_loader

OUTPUT_DIM = 11

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

# Load the data-set
TEXT, train_itr, valid_itr = data_loader.load_data(generate_bigrams)

class_counts = np.zeros(OUTPUT_DIM)

# Get distribution
for batch in train_itr:
	y = batch.label

	for elem in y:
		class_counts[int(elem)] += 1


probabilities = class_counts / sum(class_counts)

correct_count = 0.0
count = 0.0

for batch in valid_itr:
	y = batch.label

	for elem in y:
		prediction = np.random.choice(OUTPUT_DIM, p=probabilities)

		if prediction == elem:
			correct_count += 1

		count += 1


print('Random\'s accuracy: {}'.format(correct_count / count))

