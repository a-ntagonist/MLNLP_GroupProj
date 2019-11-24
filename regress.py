import csv
import io
import numpy as np
import pandas as pd
import statsmodels.api as sm

def readkeys(fn):
    f = open('data\\'+fn+'.txt').read().split()
    return f

def getstats():
    stats = dict()
    with open('data\\women_job_statistics.csv', encoding='utf-8') as f:
        r = csv.reader(f)
        for row in r:
            stats[row[0]] = row[1]
    ans = dict()
    with open('data\\job_words.csv', encoding='utf-8') as f:
        r = csv.reader(f)
        for row in r:
            ans[(row[0], tuple(row[1:]))] = stats[row[0]]
    return ans


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
        return [0 for i in range(300)]
    K = np.mean(K, axis=0)
    #print(len(K), K[0], type(K))
    return K


def getcossim(vecA, vecB):
    return np.linalg.norm(vecA-vecB)

if __name__ == '__main__':
    group_A = readkeys('female_pairs') #list [w1, ... wD]
    group_B = readkeys('male_pairs') #list [w1, ... wD]
    need = set(group_A).union(set(group_B))
    alljobs = getstats() #dict{(keyword, (words)):float(0~1)}
    for jobs, stat in alljobs.items():
        need = need.union(set(jobs[1]))

    vectors = wordvec('cc.ko.300.vec', need) # wordvec object
    groupvec_A = avgvec(group_A, vectors)
    groupvec_B = avgvec(group_B, vectors)

    results = []
    for jobs, stat in alljobs.items():
        jobvec = avgvec(jobs[1], vectors)
        bias = getcossim(groupvec_B, jobvec)-getcossim(groupvec_A, jobvec)
        results.append([jobs[0], bias, stat])

    results = pd.DataFrame(results, columns=['job', 'bias', 'stat'])
    results.to_csv('results.csv', encoding='utf-8')

    X = sm.add_constant(results.bias.astype(float))
    Y = results.stat.astype(float)

    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)

    print(model.summary())
