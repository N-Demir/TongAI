"""
This file is where Nikita compile's random data viz things

Dataset things:

Class gender counts
male: [1415.  383.  357.  273.  391.  757.  438. 1185.   63.   85.   72.]
female: [2274.  429.  821.  305.  491.  994.  661. 1380.  155.   61.   98.]

Class age vis you just have to run

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import math

# Visualize gender and age across classes

DELIMITER = '|'
DATA_PATH = 'data/processed/final.csv'

CLASSES = ['mets', 'memory', 'multiple sclerosis', 'epilepsy', 'stereo/cyberknife', 'routine brain', 'sella', 'tumor brain', 'complex headache', 'brain trauma', 'stroke']

## Age:
inFile = open(DATA_PATH,'r')

age_class_arr = np.zeros((len(CLASSES), 11))

for line in inFile:
	row = line.split(DELIMITER)

	age = int(float(row[0]))
	gender = row[1]
	reason = row[2]
	protocol = int(row[3])

	age = math.floor(age / 10)

	age_class_arr[protocol, age] += 1

for col_idx in range(11):
	age_class_arr[:,col_idx] /= sum(age_class_arr[:,col_idx])

df_cm = pd.DataFrame(age_class_arr, index = [i for i in CLASSES],
                  columns = [i for i in range(11)])
plt.figure(figsize = (10,7))
ax = sn.heatmap(df_cm, annot=True, fmt='.2g')

ax.set_ylabel('Classes')
ax.set_title('Class \\ Age (Decade) Matrix Normalized by Columns')

plt.show()

## Gender:
# inFile = open(DATA_PATH,'r')

# male_classes_count = np.zeros(len(CLASSES))
# female_classes_count = np.zeros(len(CLASSES))

# for line in inFile:
# 	row = line.split(DELIMITER)

# 	age = row[0]
# 	gender = row[1]
# 	reason = row[2]
# 	protocol = row[3]

# 	if gender == 'm':
# 		male_classes_count[int(protocol)] += 1
# 	elif gender == 'f':
# 		female_classes_count[int(protocol)] += 1

# print(male_classes_count)
# print(female_classes_count)

# male_classes_percentages = male_classes_count / sum(male_classes_count)
# female_classes_percentages = female_classes_count / sum(female_classes_count)


# indices = np.arange(len(CLASSES))
# width = 0.3

# fig = plt.figure()
# ax = fig.add_subplot(111)

# bars1 = plt.bar(indices + width / 2, [count for count in male_classes_percentages], width=width, color='b')
# bars2 = plt.bar(indices - width / 2, [count for count in female_classes_percentages], width=width, color='r')


# ax.set_ylabel('Percent')
# ax.set_title('Percent of gender per class')
# ax.set_xticks(indices)
# ax.set_xticklabels(CLASSES)

# plt.xticks(rotation='vertical')

# ax.legend( (bars1[0], bars2[0]), ('Men', 'Women') )

# plt.show()


# # 19546 brain

# save_folder_path = 'data/processed/'
# DELIMITER = '|'

# inFile = open(save_folder_path + 'brain.csv','r')

# variations = dict()

# for line in inFile:
# 	row = line.split(DELIMITER)

# 	age = row[0]
# 	gender = row[1]
# 	reason = row[2]
# 	protocol = row[3]

# 	protocol_parts = protocol.split(",")

# 	flag = False
# 	for protocol in protocol_parts:
# 		if protocol in ["brain trauma", "memory", "mets", "tumor brain", "sella", "stroke", "multiple sclerosis", "epilepsy", "stereo/cyberknife", "complex headache", "routine brain"]:
# 			print(protocol)
# 			flag = True
# 	if flag:
# 		variations[protocol_parts[0]] = variations.get(protocol_parts[0], 0) + 1

# print(variations)

# inFile.close()

"""
Inside brain we have the following classes by the first index of protocol:

brain trauma
memory
mets
tumor brain
sella
stroke (?)
multiple sclerosis
epilepsy
stereo/cyberknife
complex headache
routine brain (?)
"""