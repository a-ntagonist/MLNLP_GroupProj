import csv
import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
from openpyxl import load_workbook

def readkeys(fn):
    f = open('data\\'+fn+'.txt').read().split()
    return f

def getcands():
    cands = []
    wb = load_workbook('data\\list_jobs.xlsx')
    ws = wb.active
    for row in ws.rows:
        if len(row)>1:
            cands.extend(list(map(lambda x:x.value, row[1:])))
    return list(filter(lambda x:x!=None, cands))


class wordvec:
    def __init__(self, fname, need):
        fin = io.open('data\\'+fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            if tokens[0] in need:
                data[tokens[0]] = np.array(list(map(float, tokens[1:])))
        self.vecdict = data# dictionary token:nparray

    def word(self, word): # return nparray or None for OOV
        if word in self.vecdict:
            return self.vecdict[word]
        else:
            return np.array([None for i in range(300)])



def avgvec(words, vectors):
    K = list(filter(lambda x:x.any()!=None, list(map(vectors.word, words))))
    if len(K) == 0:
        print('None of {} are in word vector!'.format(words))
        return np.array([None for i in range(300)])
    K = np.mean(K, axis=0)
    #print(len(K), K[0], type(K))
    return K


def getcossim(vecA, vecB):
    return np.linalg.norm(vecA-vecB)

if __name__ == '__main__':
    group_A = readkeys('female_pairs') #list [w1, ... wD]
    group_B = readkeys('male_pairs') #list [w1, ... wD]
    need = set(group_A).union(set(group_B))
    alljobs = getcands() #dict{(keyword, (words)):float(0~1)}
    need = need.union(set(alljobs))

    vectors = wordvec('cc.ko.300.vec', need) # wordvec object
    groupvec_A = avgvec(group_A, vectors)
    groupvec_B = avgvec(group_B, vectors)
    distB = dict()
    results = []
    for job in alljobs:
        jobvec = vectors.word(job)
        if jobvec.any() != None:
            distB[job] = getcossim(groupvec_B, jobvec)-getcossim(groupvec_A, jobvec)

    distB = sorted(distB.items(), key=lambda x:x[1])
    print(distB[:10])
    print(distB[-10:])
