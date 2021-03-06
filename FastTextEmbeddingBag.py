"""
Credit to facebookresearc/fastText

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from torch.nn.modules.sparse import EmbeddingBag
import numpy as np
import torch
import random
import string
import time
from fastText import load_model
from torch.autograd import Variable


# to do:
# use spacy
# have variable length

class FastTextEmbeddingBag(EmbeddingBag):
	def __init__(self, model_path, device):
		self.model = load_model(model_path)
		self.device = device
		input_matrix = self.model.get_input_matrix()
		input_matrix_shape = input_matrix.shape
		self.embedding_dim = input_matrix_shape[1]
		super().__init__(input_matrix_shape[0], self.embedding_dim)
		self.weight.data.copy_(torch.FloatTensor(input_matrix))

	def forward(self, sentences):
		"""
		expecting sentences to be of the shape batch_size * max_sentence_len
		"""
		batch_size = len(sentences)
		max_sentence_len = len(sentences[0])

		embeddings = torch.empty((batch_size, self.embedding_dim), device = self.device)
		for i, sentence in enumerate(sentences):
			word_subinds = np.empty([0], dtype=np.int64)
			word_offsets = [0]
			for word in sentence:
				_, subinds = self.model.get_subwords(word)
				word_subinds = np.concatenate((word_subinds, subinds))
				word_offsets.append(word_offsets[-1] + len(subinds))
			word_offsets = word_offsets[:-1]
			ind = Variable(torch.tensor(word_subinds, dtype = torch.long, device = self.device))
			offsets = Variable(torch.tensor(word_offsets, dtype = torch.long, device = self.device))
			embeddings[i] = torch.mean(super().forward(ind, offsets), dim = 0) #embedding averaged before storing 
		return embeddings


def random_word(N):
    return ''.join(
        random.choices(
            string.ascii_uppercase + string.ascii_lowercase + string.digits,
            k=N
        )
    )


if __name__ == "__main__":
    ft_emb = FastTextEmbeddingBag("fil9.bin")
    model = load_model("fil9.bin")
    num_lines = 200
    total_seconds = 0.0
    total_words = 0
    for _ in range(num_lines):
        words = [
            random_word(random.randint(1, 10))
            for _ in range(random.randint(15, 25))
        ]
        total_words += len(words)
        words_average_length = sum([len(word) for word in words]) / len(words)
        start = time.clock()
        words_emb = ft_emb(words)
        total_seconds += (time.clock() - start)
        for i in range(len(words)):
            word = words[i]
            ft_word_emb = model.get_word_vector(word)
            py_emb = np.array(words_emb[i].data)
            assert (np.isclose(ft_word_emb, py_emb).all())
    print(
        "Avg. {:2.5f} seconds to build embeddings for {} lines with a total of {} words.".
        format(total_seconds, num_lines, total_words)
    )