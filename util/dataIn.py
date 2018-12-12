import json, display, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import dok_matrix

def parseTrackURI(uri):
    return uri.split(":")[2]

def processPlaylistForClustering(playlists, tracks):
    """
    Create sparse matrix mapping playlists to track
    lists that are consumable by most clustering algos
    """

    # List of all track IDs in db
    trackIDs = list(tracks["tid"])
    
    # Map track id to matrix index
    IDtoIDX = {k:v for k,v in zip(trackIDs,range(0,len(trackIDs)))}
    
    playlistIDs = list(playlists["pid"])
    
    print("Create sparse matrix mapping playlists to tracks")
    playlistSongSparse = dok_matrix((len(playlistIDs), len(trackIDs)), dtype=np.float32)

    for i in tqdm(range(len(playlistIDs))):
        # Get playlist and track ids from DF
        playlistID = playlistIDs[i]
        trackID = playlists.loc[playlistID]["tracks"]
        playlistIDX = playlistID
        
        # Get matrix index for track id
        trackIDX = [IDtoIDX.get(i) for i in trackID]
        
        # Set index to 1 if playlist has song
        playlistSongSparse[playlistIDX, trackIDX] = 1 

    return playlistSongSparse.tocsr(), IDtoIDX

def createDFs(idx, numFiles, path, files):
    """
    Creates playlist and track DataFrames from
    json files
    """
    # Get correct number of files to work with
    files = files[idx:idx+numFiles]

    tracksSeen = set()
    playlistsLst = []
    trackLst = []

    print("Creating track and playlist DFs")
    for i, FILE in enumerate(tqdm(files)):
        # get full path to file
        name = path + FILE 
        with open(name) as f:
            data = json.load(f)
            playlists = data["playlists"]

            # for each playlist
            for playlist in playlists:
                for track in playlist["tracks"]:
                    if track["track_uri"] not in tracksSeen:
                        tracksSeen.add(track["track_uri"])
                        trackLst.append(track)
                playlist["tracks"] = [parseTrackURI(x["track_uri"]) for x in playlist["tracks"]]
                playlistsLst.append(playlist)
    
    playlistDF = pd.DataFrame(playlistsLst)

    playlistDF.set_index("pid")

    tracksDF = pd.DataFrame(trackLst)
    # Split id from spotifyURI for brevity
    tracksDF["tid"] = tracksDF.apply(lambda row: parseTrackURI(row["track_uri"]), axis=1)

    playlistClusteredDF, IDtoIDXMap = processPlaylistForClustering(playlists=playlistDF,
                                                       tracks=tracksDF)

    # Add sparseID for easy coercision to sparse matrix for training data
    tracksDF["sparse_id"] = tracksDF.apply(lambda row: IDtoIDXMap[row["tid"]], axis=1)
    tracksDF = tracksDF.set_index("tid")
    
    # Write DFs to CSVs
    print(f"Pickling {len(playlistDF)} playlists")
    playlistDF.to_pickle("lib/playlists.pkl")
    print(f"Pickling {len(tracksDF)} tracks")
    tracksDF.to_pickle("lib/tracks.pkl")
    print(f"Pickling clustered playlist")
    pickle.dump(playlistClusteredDF, open(f"lib/playlistSparse.pkl", "wb"))
    