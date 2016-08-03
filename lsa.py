import numpy
import pylab
from scipy import linalg
from matplotlib import pyplot, font_manager
import codecs
import math
import jieba
import re
import argparse
import os

class LSA:

    def __init__(self, documentsFilePath, modelDirectory = '', outputDirectory = '', stopwordsFilePath = None, dimension = 2):
        
        self.documentsFilePath = documentsFilePath
        self.modelDirectory = modelDirectory
        self.outputDirectory = outputDirectory
        self.stopwordsFilePath = stopwordsFilePath
        self.dimension = dimension
        if os.path.exists('C:/Windows/Fonts/msyh.ttf'):
            self.zhFont = font_manager.FontProperties(fname = 'C:/Windows/Fonts/msyh.ttf')
        elif os.path.exists('C:/Windows/Fonts/msyh.ttc'):
            self.zhFont = font_manager.FontProperties(fname = 'C:/Windows/Fonts/msyh.ttc')

    def train(self):
        
         # read stopwords
        self.stopwords = set()
        if self.stopwordsFilePath != None:
            for word in codecs.open(self.stopwordsFilePath, 'r', 'utf-8'):
                self.stopwords.add(word.lower().strip())
                
        # read documents
        self.docs = []
        for document in codecs.open(self.documentsFilePath, 'r', 'utf-8'):
            self.docs.append(document)
        self.N = len(self.docs)
        
        # word's document count
        worddc = {}

        # calculate word's document count
        for i in range(0, self.N):
            doc = self.docs[i]
            words = []
            segment = jieba.cut(doc)
            tokens = [word for word in segment]
            for word in tokens:
                word = word.lower().strip()
                if word not in self.stopwords and len(word) > 1 and not re.search('[0-9]', word):
                    words.append(word)
            for w in words:
                if w in worddc.keys():
                    worddc[w].append(i)
                else:
                    worddc[w] = [i]

        # generate keywords
        self.keywords = [word for word in worddc.keys() if len(worddc[word]) > 1]
        self.keywords.sort()
        self.M = len(self.keywords)
        
        # generate word-document matrix
        self.X = numpy.zeros([self.M, self.N])
        self.id2word = {}
        self.word2id = {}
        for i, word in enumerate(self.keywords):
            self.id2word[i] = word
            self.word2id[word] = i
            for d in worddc[word]:
                self.X[i, d] += 1

        # svd
        self.U, self.sigma, self.V = linalg.svd(self.X, full_matrices=True)
        
        # dimension reduction
        self._U = self.U[0:, 0:self.dimension]
        self._V = self.V[0:self.dimension, 0:]
        self._sigma = numpy.diag(self.sigma[0:self.dimension])
        self._X = numpy.dot(numpy.dot(self._U, self._sigma), self._V);


    # save the model and output some analysis results
    def saveModel(self):
        
        numpy.save(self.modelDirectory + '/_U.npy', self._U)
        numpy.save(self.modelDirectory + '/_V.npy', self._V)
        
        # save keywords
        file = codecs.open(self.modelDirectory + '/keywords.txt', 'w', 'utf-8')
        for i in range(0, self.M):
            file.write(self.keywords[i] + "\n")
        file.close()
        
        # save coordinates of documents
        file = codecs.open(self.outputDirectory + '/document_coordinate.txt', 'w', 'utf-8')
        for i in range(0, self.N):
            file.write("doc" + str(i) + " " + str(self._V[0][i]) + " " + str(self._V[1][i]) + "\n")
        file.close()
        
        # save coordinates of words
        file = codecs.open(self.outputDirectory + '/words_coordinate.txt', 'w', 'utf-8')
        for i in range(0, self.M):
            file.write(self.keywords[i] + " " + str(self._U[i][0]) + " " + str(self._U[i][1]) + "\n")
        file.close()
        
        # save visualization result
        pyplot.title("LSA")
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        for i in range(len(self._U)):
            pylab.text(self._U[i][0], self._U[i][1],  self.keywords[i], fontsize=10, fontproperties=self.zhFont)
        pylab.plot(self._U.T[0], self._U.T[1], '.')
        for i in range(len(self._V[0])):
            pylab.text(self._V[0][i], self._V[1][i], ('doc%d' %(i)), fontsize=10)
        pylab.plot(self._V[0], self._V[1], 'x')
        pylab.plot([0], [0], 'o')
        pylab.savefig(self.outputDirectory + "/visualization.png", dpi=120)
        
    def loadModel(self):
        self._U = numpy.load(self.modelDirectory + '/_U.npy')
        self._V = numpy.load(self.modelDirectory + '/_V.npy')
        self.dimension = len(self._V)
        self.N = len(self._V[0])
        self.keywords = []
        self.word2id = {}
        self.id2word = {}
        for keyword in codecs.open(self.modelDirectory + '/keywords.txt', 'r', 'utf-8'):
                self.keywords.append(keyword.strip())
        self.M = len(self.keywords)
        for i in range(0, self.M):
            self.word2id[self.keywords[i]] = i
            self.id2word[i] = self.keywords[i]

    def infer(self):
        self.docs = []
        for document in codecs.open(self.documentsFilePath, 'r', 'utf-8'):
            self.docs.append(document)
        file = codecs.open(self.outputDirectory + '/infer.txt', 'w', 'utf-8')
        for i in range(0, len(self.docs)):
            doc = self.docs[i]
            words = []
            segment = jieba.cut(doc)
            tokens = [word for word in segment]
            for word in tokens:
                word = word.lower().strip()
                if word in self.keywords:
                    words.append(word)
            coordinate = numpy.zeros((self.dimension))
            for word in words:
                for i in range(0, self.dimension):
                    coordinate[i] += self._U[self.word2id[word]][i] * 1.0 / len(words)
            coordinates = []
            for i in range(0, self.N):
                tmp = numpy.zeros((self.dimension))
                for k in range(0, self.dimension):
                    tmp[k] = self._V[k][i]
                coordinates.append(tmp)
            degrees = numpy.zeros((self.N))
            for i in range(0, self.N):
                degrees[i] = math.acos(numpy.dot(coordinate, coordinates[i]) / (linalg.norm(coordinate) * linalg.norm(coordinates[i])))
            similar = numpy.argsort(degrees)[0:5]
            s = ''
            for i in range(0, len(similar)):
                if i != 0:
                    s += ' ,'
                s += str(similar[i]) + '(' + str(degrees[similar[i]]) + ')'
            file.write(s + '\n')
        file.close()         
                
            
def readParamsFromCmd():
    parser = argparse.ArgumentParser(description = "This is a python implementation of lsa(latent semantic analysis). Support both English and Chinese. A line in the input represents a document.")
    parser.add_argument('mode', help = 'train or infer')
    parser.add_argument('documentsFilePath', help = 'The file path of input documents for training or testing')
    parser.add_argument('-m', '--modelDirectory', help = 'The directory of model.(default current dir)', default = '')
    parser.add_argument('-o', '--outputDirectory', help = 'The directory of other outputs. (default current dir).', default = '')
    parser.add_argument('-s', '--stopwordsFilePath', help = 'The file path of stopwords, each line represents a word.', default = None)
    parser.add_argument('-d', '--dimension', type = int, help = 'The dimension to reduce to (default 2).', default = 2)
    return parser.parse_args()

params = readParamsFromCmd().__dict__
lsa = LSA(params['documentsFilePath'], modelDirectory = params['modelDirectory'], outputDirectory = params['outputDirectory'], stopwordsFilePath = params['stopwordsFilePath'], dimension = params['dimension'])
if params['mode'] == 'train':
    lsa.train()
    lsa.saveModel()
    print('train complete and model is saved!')
elif params['mode'] == 'infer':
    lsa.loadModel()
    lsa.infer()
    print('infer complete!')
else:
    print('unknown mode ' + params['mode'])