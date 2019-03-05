"""
Run once to get nicely formatted data

IMPORTANT: You need to first create a folder 'data' and inside there a folder called 'processed' (feel free to change)
		   Then, move the 'neuro_protocol_sent_021419.xlsx' file into data and run this script.

The final representation is:

age||gender||protocol_class_number||reason_for_visit

"""

import numpy as np
import re
import xlrd

file_path = 'data/neuro_protocol_sent_021419.xlsx'
save_folder_path = 'data/processed/'

DELIMITER = '||' # Hopefully this doesn't increase the size much haha

#### Preprocess and format

print('Reading in excel file and reformatting')

book = xlrd.open_workbook(file_path)
sheet = book.sheets()[0]

outFile = open(save_folder_path + 'processed.csv', 'w')

for idx_row, row in enumerate(sheet.get_rows()):
	if idx_row == 0:
		continue

	age = row[1].value
	gender = row[2].value
	protocol = row[4].value
	reason = row[6].value

	if protocol == '' or reason == '':
		print('\033[91m' +  'Skipping this row because missing protocol or reason: ' + '\033[0m')
		print(row)
	else:
		outFile.write(str(age) + DELIMITER + str(gender) + DELIMITER + str(protocol) + DELIMITER + str(reason).replace('\n', '') + '\n')


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

#### Turning them into classes

print('Creating the classes')

inFile = open(save_folder_path + 'no_duplicates_processed.csv','r')

outFile = open(save_folder_path + 'final.csv','w')

classes = []
class_length = {}

for line in inFile:
	split = line.split(DELIMITER)

	class_name = split[2].lower()

	if class_name in classes:
		index = classes.index(class_name)
		class_length[class_name] += 1
	else:
		classes.append(class_name)
		index = len(classes) - 1
		class_length[class_name] = 1

	outFile.write(split[0] + DELIMITER + split[1] + DELIMITER + str(index) + DELIMITER + split[3])

print('Created classes are: {}'.format(classes))
print('Total number: {}'.format(len(class_length)))
print('For histogram purposes {}'.format(class_length))

outFile.close()

inFile.close()
