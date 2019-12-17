import numpy as np

vecs = np.loadtxt("wordVectors.txt")
word = open("vocab.txt").read().split('\n')
words_to_find = ['dog', 'england', 'john', 'explode', 'office']

for w in words_to_find:
    i = word.index(w)
    vector = vecs[i]
    res = np.full(len(vecs), -np.inf)
    for j, vec in enumerate(vecs):
        if i == j:
            continue
        res[j] = (vector.dot(vec))/(np.linalg.norm(vec)*np.linalg.norm(vector))
    print(f"{w} similar words: {[word[i] for i in res.argsort()[-5:][::-1]]}")



