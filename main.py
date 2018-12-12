import json, argparse, os, random

import pprint as pp
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score

from models.NNeighClassifier import NNeighClassifier
from models.BaseClassifier import BaseClassifier
from util import vis, dataIn
from util.helpers import playlistToSparseMatrixEntry, getPlaylistTracks, getTrackandArtist, obscurePlaylist

class SpotifyExplorer:
    """
    Args:
        numFiles (int): CLI variable that determines how many MPD files to read
        retrainNNC (bool): determines whether to retrain NNC or read from file

    Attributes:
        NNC (NNeighClassifier): NNeighbor Classifier used for predictions
        baseClassifier (BaseClassifier): Baseline classifier for comparison
        playlists (DataFrame): contains all playlists read into memory
        songs (DataFrame): all songs read into memory
        playlistSparse (scipy.CSR matrix) playlists formatted for predictions
    """
    def __init__(self, numFiles, retrainNNC=True):
        self.readData(numFiles)
        self.buildClassifiers(retrainNNC)

    def buildClassifiers(self, retrainNNC):
        """
        Init classifiers and set initial classifier as main
        """
        self.NNC = self.buildNNC(retrainNNC)
        self.baseClassifier = self.buildBaseClassifier()
        self.classifier = self.NNC

    def buildNNC(self, shouldRetrain): 
        """
        Init NNC classifier
        """
        self.NNC = NNeighClassifier(
            sparsePlaylists=self.playlistSparse,
            songs=self.songs,
            playlists=self.playlists,
            reTrain=shouldRetrain) 
        return self.NNC

    def buildBaseClassifier(self):
        """
        Init base classifier
        """
        self.baseClassifier = BaseClassifier(
            songs=self.songs,
            playlists=self.playlists)  
        return self.baseClassifier
    
    def setClassifier(self, classifier="NNC"):
        """
        Select classifier to set as main classifier
        """
        if classifier == "NNC":
            self.classifier = self.NNC
        elif classifier == "Base":
            self.classifier = self.baseClassifier

    def readData(self, numFilesToProcess):
        """
        Read song and playlist data
        Either read from MPD data or pickled dataframe
        """
        # don't have to write every time
        if numFilesToProcess > 0:
            # extract number from file
            def sortFile(f):
                f = f.split('.')[2].split('-')[0]
                return int(f)
            files = os.listdir("data/data")
            files.sort(key=sortFile)

            dataIn.createDFs(idx=0, 
                numFiles=numFilesToProcess,
                path="data/data/",
                files=files)

        # Read data
        print("Reading data")
        self.playlists = pd.read_pickle("lib/playlists.pkl")
        self.songs = pd.read_pickle("lib/tracks.pkl")
        self.playlistSparse = pd.read_pickle("lib/playlistSparse.pkl")
        print(f"Working with {len(self.playlists)} playlists " + \
            f"and {len(self.songs)} songs")
    
    def getRandomPlaylist(self): 
        return self.playlists.iloc[random.randint(0,len(self.playlists) - 1)]

    def predictNeighbour(self, playlist, numPredictions, songs):
        """
        Use currently selected predictor to predict neighborings songs
        """
        return self.classifier.predict(playlist, numPredictions, songs)
        
    def obscurePlaylist(self, playlist, obscurity): 
        """
        Obscure a portion of a playlist's songs for testing
        """
        k = len(playlist['tracks']) * obscurity // 100
        indices = random.sample(range(len(playlist['tracks'])), k)
        obscured = [playlist['tracks'][i] for i in indices]
        tracks = [i for i in playlist['tracks'] + obscured if i not in playlist['tracks'] or i not in obscured]
        return tracks, obscured

    def evalAccuracy(self, numPlaylists, percentToObscure=0.5): 
        """
        Obscures a percentage of songs
        Iterates and sees how many reccomendations match the missing songs
        """
        print()
        print(f"Selecting {numPlaylists} playlists to test and obscuring {int(percentToObscure * 100)}% of songs")

        def getAcc(pToObscure):
            playlist = self.getRandomPlaylist()

            keptTracks, obscured = obscurePlaylist(playlist, pToObscure)
            playlistSub = playlist.copy()
            obscured = set(obscured)
            playlistSub['tracks'] = keptTracks

            predictions = self.predictNeighbour(playlistSub, 
                500, 
                self.songs)

            overlap = [value for value in predictions if value in obscured]

            return len(overlap)/len(obscured)
        
        accuracies = [getAcc(percentToObscure) for _ in tqdm(range(numPlaylists))]
        avgAcc = round(sum(accuracies) / len(accuracies), 4) * 100
        print(f"Using {self.classifier.name}, we predicted {avgAcc}% of obscured songs")
    
    def displayRandomPrediction(self):
        playlist = self.getRandomPlaylist()
        while len(playlist["tracks"]) < 10:
            playlist = self.getRandomPlaylist()

        predictions = self.predictNeighbour(playlist=playlist,
            numPredictions=50,
            songs=self.songs)


        playlistName = playlist["name"]
        playlist = [getTrackandArtist(trackURI, self.songs) for trackURI in playlist["tracks"]]
        predictions = [getTrackandArtist(trackURI, self.songs) for trackURI in predictions]
        return {
            "name": playlistName,
            "playlist": playlist,
            "predictions": predictions
        }
    
    def createRandomPredictionsDF(self, numInstances):
        print(f"Generating {numInstances} data points")
        data = [self.displayRandomPrediction() for _ in tqdm(range(numInstances))]
        df = pd.DataFrame(data)
        df.to_csv("predictionData.csv")
        

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--parseData')
    args = parser.parse_args()
    if args.parseData:
        numToParse = int(args.parseData)
    else:
        numToParse = 0

    """
    Builds explorer
    numFiles: Number of files to load (each with 1000 playlists)
    parse:    Boolean to load in data
    """

    # Init class
    spotify_explorer = SpotifyExplorer(numToParse)

    #Run tests on NNC
    spotify_explorer.evalAccuracy(30)
    
    # # # Run tests on base
    spotify_explorer.setClassifier("Base")
    spotify_explorer.evalAccuracy(30)

    # Generate prediction CSV
    spotify_explorer.createRandomPredictionsDF(100)
