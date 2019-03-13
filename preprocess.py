"""
Run once to get nicely formatted data

IMPORTANT: You need to first create a folder 'data' and inside there a folder called 'processed' (feel free to change)
		   Then, move the 'neuro_protocol_sent_021419.xlsx' file into data and run this script.

The final representation is:

age||gender||protocol_class_number||reason_for_visit

Output is final.csv (Other intermediate files don't matter) 

"""

import numpy as np
import re
import xlrd

file_path = 'data/neuro_protocol_sent_021419.xlsx'
save_folder_path = 'data/processed/'

DELIMITER = '|' # Hopefully this doesn't increase the size much haha

#### Preprocess and format

print('Reading in excel file and reformatting')

book = xlrd.open_workbook(file_path)
sheet = book.sheets()[0]

outFile = open(save_folder_path + 'processed.csv', 'w')

for idx_row, row in enumerate(sheet.get_rows()):
	if idx_row == 0:
		continue

	age = row[1].value
	gender = row[2].value.lower()
	reason = str(row[6].value).lower()
	protocol = str(row[7].value).lower()

	if protocol == '' or reason == '':
		print('\033[91m' +  'Skipping this row because missing protocol or reason: ' + '\033[0m')
		print(row)
	else:
		outFile.write(str(age) + DELIMITER + str(gender) + DELIMITER + str(reason).replace('\n', '').replace('\r', '') + DELIMITER + str(protocol).replace('\n', '').replace('\r', '') + '\n')


outFile.close()

#### Duplicates

print('Removing duplicates')

inFile = open(save_folder_path + 'processed.csv','r')

outFile = open(save_folder_path + 'no_duplicates_processed.csv','w')

listLines = set()


for line in inFile:

    if line in listLines:
        continue

    else:
        outFile.write(line)
        listLines.add(line)

outFile.close()

inFile.close()

#### Remove the 19548 that reference brain in protocols

print('Separating the brain ones')

inFile = open(save_folder_path + 'no_duplicates_processed.csv','r')

outFile = open(save_folder_path + 'brain.csv','w')

for line in inFile:

	row = line.split(DELIMITER)

	age = row[0]
	gender = row[1]
	reason = row[2]
	protocol = row[3]

	protocol_parts = protocol.split(",")

	if 'brain' in protocol_parts:
		outFile.write(line)

outFile.close()

inFile.close()

#### From brain ones separate the ones of the classes we want

print('Separating the brain ones with nicely formatted protocols')

inFile = open(save_folder_path + 'brain.csv','r')

outFile = open(save_folder_path + 'correct_brain.csv','w')

for line in inFile:

	row = line.split(DELIMITER)

	age = row[0]
	gender = row[1]
	reason = row[2]
	protocol = row[3]

	protocol_parts = protocol.split(",")

	protocol = protocol_parts[0]

	if protocol in ["brain trauma", "memory", "mets", "tumor brain", "sella", "stroke", "multiple sclerosis", "epilepsy", "stereo/cyberknife", "complex headache", "routine brain"]:
		outFile.write(line)

outFile.close()

inFile.close()

#### Turning them into classes

print('Creating the classes')

inFile = open(save_folder_path + 'correct_brain.csv','r')

outFile = open(save_folder_path + 'final.csv','w')

classes = []
class_length = {}

for line in inFile:
	split = line.split(DELIMITER)

	class_name = split[3].split(",")[0]


	if class_name in classes:
		index = classes.index(class_name)
		class_length[class_name] += 1
	else:
		classes.append(class_name)
		index = len(classes) - 1
		class_length[class_name] = 1

	outFile.write(split[0] + DELIMITER + split[1] + DELIMITER + split[2] + DELIMITER + str(index) + '\n')

print('Created classes are: {}'.format(classes))
print('Total number: {}'.format(len(class_length)))
print('For histogram purposes {}'.format(class_length))

#### Plotting

import matplotlib.pyplot as plt
from collections import Counter

a = Counter(class_length)

plt.bar([k for k, v in a.most_common(100)], [v for k, v in a.most_common(100)], 1.0, color='g')
plt.show()

outFile.close()

inFile.close()
