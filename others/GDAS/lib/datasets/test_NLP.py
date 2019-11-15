import os, sys, torch

from LanguageDataset import SentCorpus, BatchSentLoader

if __name__ == '__main__':
  path = '../../data/data/penn'
  corpus = SentCorpus( path )
  loader = BatchSentLoader(corpus.test, 10)
  for i, d in enumerate(loader):
    print('{:} :: {:}'.format(i, d.size()))
