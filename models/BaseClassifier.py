import os, pickle
import numpy as np
import pandas as pd
import heapq
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
import matplotlib.pyplot as plt

from util.helpers import playlistToSparseMatrixEntry, getPlaylistTracks

class BaseClassifier:
    def __init__(self, playlists, songs, reTrain=False, name="BaseClassifier.pkl"):
        self.pathName = name
        self.name = "Base Classifier"
        self.playlists = playlists 
        self.songs = songs
        self.popularity = self.getPopularity()

    def getPopularity(self):
        popularity = defaultdict(int)
        for playlist in self.playlists['tracks']: 
            for track in playlist:
                popularity[track] += 1
        return popularity
        
    def predict(self, X, numPredictions, songs, k=0):
        scores = heapq.nlargest(numPredictions, self.popularity, key=self.popularity.get) 
        # scores = [songs.loc[x]['track_name'] for x in scores]
        return scores