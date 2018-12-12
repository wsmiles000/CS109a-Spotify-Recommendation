import random
from scipy.sparse import dok_matrix

def playlistToSparseMatrixEntry(playlist, songs):
    """
    Converts a playlist with list of songs
    into a sparse matrix with just one row
    """
    # print(songs.iloc[1:5])
    playlistMtrx = dok_matrix((1, len(songs)))
    tracks = [songs.loc[str(x)]["sparse_id"] for x in list(playlist["tracks"])]
    playlistMtrx[0, tracks] = 1
    return playlistMtrx.tocsr()

def getPlaylistTracks(playlist, songs):
    return [songs.loc[x] for x in playlist["tracks"]]

def getTrackandArtist(trackURI, songs):
    song = songs.loc[trackURI]
    return (song["track_name"], song["artist_name"])


def obscurePlaylist(playlist, percentToObscure): 
    """
    Obscure a portion of a playlist's songs for testing
    """
    k = int(len(playlist['tracks']) * percentToObscure)
    indices = random.sample(range(len(playlist['tracks'])), k)
    obscured = [playlist['tracks'][i] for i in indices]
    tracks = [i for i in playlist['tracks'] + obscured if i not in playlist['tracks'] or i not in obscured]
    return tracks, obscured