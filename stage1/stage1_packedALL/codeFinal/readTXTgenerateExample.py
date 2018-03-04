import re
import csv
from os import listdir
from os.path import isfile, join
#import nltk
from nltk.corpus import stopwords

# Updating this path to get the file set
mypath = "/Users/mingrenshen/Develop/CS839ClassProject/stage1/DataSet/shuffledtext/"

files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-4:] == ".txt"]
combo = [1,2,3,4]
fields = ['docID','word', 'startPos', 'endPos', 'label','bag','preWord','postWord']
stops = set(stopwords.words("english"))

# generate examples
with open("data_shuffled.csv", 'w') as csvFile:
    csvWriter = csv.DictWriter(csvFile, fieldnames=fields)
    csvWriter.writeheader()
    for f in files:
        docID = f[:-4]
        with open(mypath + f,'r') as file:
            text = file.read().replace('\n', '')
            name = re.findall("<>.*?</>", text)
            name = [re.sub(r'[^\w\s]', '', x).strip() for x in name]
            text = re.sub(r'[^\w\s]', "", text)
            text = re.sub(r'[^a-zA-Z]', " ", text)
            text = re.sub("\s\s+", " ", text)
            #text = [w for w in text.split()]
            text = [w for w in text.split() if not w.lower() in stops]
            for n in combo:
                Pos = 0
                for index in range(len(text)-n):
                    word = ""
                    if index==0:
                        preWord="NA"
                    else:
                        preWord=text[index-1]
                    if index==len(text)-n-1:
                        postWord="NA"
                    else:
                        postWord=text[index+n]
                    for j in range(index, index+n):
                        word = word + text[j] + " "
                    word = word.rstrip()
                    startPos = Pos
                    endPos = Pos + len(word) - 1
                    Pos = startPos + len(text[index]) + 1
                    label = 0
                    if word in name:
                        label = 1
                    row = {'docID': docID, 'word': word, 'startPos': startPos, 'endPos': endPos, 'label': label, 'bag': n, 'preWord':preWord,'postWord':postWord}
                    csvWriter.writerow(row)
