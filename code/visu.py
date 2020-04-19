import csv
import io
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from openpyxl import load_workbook
import seaborn as sns

from statsmodels.sandbox.regression.predstd import wls_prediction_std

def readkeys(fn):
    f = open(fn+'.txt').read().split()
    return f

def getstats(lang, statname):
    l = []
    wb = load_workbook('..\\Data\\{}\\{}_words.xlsx'.format(lang, statname))
    ws = wb.active
    for row in ws.rows:
        if len(row)>1:
            words = tuple(filter(lambda x:x!=None, map(lambda x:x.value, row[1:])))
            l.extend(words)
    return l

def getcossim(vecA, vecB):
    return np.linalg.norm(vecA-vecB)


class wordvec:
    def __init__(self, lang, fname, need):
        fin = io.open('..\\data\\{}\\'.format(lang)+fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
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


    lang = 'korean' #or 'japanese'
    statname = 'adjective' #or 'participation' or 'mturk'


    group_A = readkeys('..\\data\\{}\\female_keys'.format(lang)) #list [w1, ... wD]
    group_B = readkeys('..\\data\\{}\\male_keys'.format(lang)) #list [w1, ... wD]
    need = set(group_A).union(set(group_B)) #words we need vectors for
    alljobs = getstats(lang, statname) #dict{(keyword, (lemmas)):stat as float} dictionary
    need = need.union(set(alljobs))

    results = []
    vectors = wordvec(lang, 'cc.{}.300.vec'.format(lang[:2]), need) #maps words to vectors, with only needed words to save memory
    groupvec_A = avgvec(group_A, vectors)
    groupvec_B = avgvec(group_B, vectors)
    results.append(groupvec_A)
    results.append(groupvec_B)
    c = 0
    for job in alljobs:
        jobvec = avgvec([job], vectors)
        if jobvec.any() != None:
            results.append(jobvec)
        else:
            c+=1
    print('Total number of key concepts:', len(alljobs))
    print('{} OOV concepts, {:.3f}% of total concepts'.format(c, c/len(alljobs)))
    feat_cols = ['dim'+str(i) for i in range(300)]
    results = pd.DataFrame(results, columns=feat_cols)
    labels = ['Female', 'Male']+['Jobs' for i in range(len(results)-2)]
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(results[feat_cols].values)

    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pca_result)
    tsne_results = pd.DataFrame(tsne_results, columns = ['1', '2', '3'])
    tsne_results['Vectors'] = labels
    sns.scatterplot(
        x="1", y="2",
        hue="Vectors",
        palette=sns.xkcd_palette(['windows blue', 'amber', 'greyish']),
        data=tsne_results,
        legend="full",
    )
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (20,10)
    plt.title('3D Visualization of {}'.format(statname))
    plt.savefig('..\\results\\{}_{}_TSNE.png'.format(lang, statname))
