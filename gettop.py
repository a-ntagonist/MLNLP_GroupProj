import csv
import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
from openpyxl import load_workbook
import matplotlib.pyplot as plt

from statsmodels.sandbox.regression.predstd import wls_prediction_std

def readkeys(fn):
    f = open('data\\'+fn+'.txt').read().split()
    return f

def getstats():
    l = []
    wb = load_workbook('data\\adj_words.xlsx')
    ws = wb.active
    for row in ws.rows:
        if len(row)>1:
            words = tuple(filter(lambda x:x!=None, map(lambda x:x.value, row[1:])))
            l.extend(words)
    return l

def getcossim(vecA, vecB):
    return np.linalg.norm(vecA-vecB)


class wordvec:
    def __init__(self, fname, need):
        fin = io.open('data\\'+fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            if tokens[0] in need:
                data[tokens[0]] = list(map(float, tokens[1:]))
        self.vecdict = data# dictionary token:nparray

    def word(self, word): # return nparray or None for OOV
        if word in self.vecdict:
            return self.vecdict[word]
        else:
            return None

def avgvec(words, vectors):
    K = list(filter(lambda x:x!=None, list(map(vectors.word, words))))
    if len(K) == 0:
        print('None of {} are in word vector!'.format(words))
        return np.array([None for i in range(300)])
    K = np.mean(K, axis=0)
    #print(len(K), K[0], type(K))
    return K


def dist(vecA, vecB):
    return np.linalg.norm(vecA-vecB)

if __name__ == '__main__':
    group_A = readkeys('female_pairs') #list [w1, ... wD]
    group_B = readkeys('male_pairs') #list [w1, ... wD]
    need = set(group_A).union(set(group_B))
    alljobs = getstats() #dict{(keyword, (words)):float(0~1)}
    need = need.union(set(alljobs))

    vectors = wordvec('cc.ko.300.vec', need) # wordvec object
    groupvec_A = avgvec(group_A, vectors)
    groupvec_B = avgvec(group_B, vectors)
    c = 0
    results = []
    for job in alljobs:
        jobvec = avgvec([job], vectors)
        if jobvec.any() != None:
            distB = getcossim(groupvec_B, jobvec)
            distA = getcossim(groupvec_A, jobvec)
            results.append([job, distB, distA, distB-distA])
        else:
            c+=1
    print(len(alljobs))
    print(c, c/len(alljobs))
    results = pd.DataFrame(results, columns=['job', 'distA', 'distB', 'bias']).drop_duplicates(subset=['job'])

    s = results.sort_values(by=['distA'])
    print('여성', s.job[:10])
    s = results.sort_values(by=['distB'])
    print('남성', s.job[:10])
    s = results.sort_values(by=['bias'])
    print('bias', s.job[:10], s.job[-10:])
