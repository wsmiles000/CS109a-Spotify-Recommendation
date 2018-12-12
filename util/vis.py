import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plotHist(ax, title, xlabel, ylabel, data):
    """
    Helper used to plot histogram
    """
    ax.hist(data, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plotBar(ax, title, xlabel, ylabel, spacing, heights, labels):
    """
    Helper used to plot bar chart
    """
    # Plot data
    ax.bar(spacing, heights)
    plt.xticks(
            ticks=spacing,
            labels=labels,
            rotation="vertical",
            fontsize=5
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def displayPopularArtists(df, lim=100):
    """
    Display the number of playlist inclusions for
    X most popular artists
    """
    # Initialize dictionary of artist names
    artists = {}
    for playlist in df.tracks:
        for song in playlist:
            artsist = song["artist_name"]
            if artsist in artists:
                artists[artsist] += 1
            else: 
                artists[artsist] = 1
    
    # Sort artists by popularity
    sortedArtists = sorted(artists.items(), 
                           key=lambda x: x[1],
                           reverse=True)
    mostPopular = sortedArtists[:lim]
    artists, count = zip(*mostPopular)
    xvals = np.arange(len(mostPopular))

    # Plot data
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plotBar(ax, "Number of Playlist Appearences by Top 100 Artists",
        xlabel="Artist",
        ylabel="Number of Appearences",
        spacing=xvals,
        heights=count,
        labels=artists)
    plt.savefig("figs/popularArtists.png")

def displayMostCommonKeyWord(df):
    """
    Displays bar chart of most common keywords
    used in playlist names
    """
    playlist_names = df["name"]
    dic = {}
    
    # Break into keywords
    keywords = [list(filter(None,name.split(" "))) for name in playlist_names]
    flat_list = [item for sublist in keywords for item in sublist]

    # Sort dict and break up into consumable form
    key_dict = dict((x,flat_list.count(x)) for x in set(flat_list))
    key_dict = sorted(key_dict.items(),
                        key=lambda x:x[1], 
                        reverse=True)

    word, count = zip(*key_dict[:20])

    xvals = np.arange(len(word))
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.bar(xvals, count)
    plotBar(ax=ax,
        title="Most Common Words in Playlist Titles",
        xlabel="KeyWord",
        ylabel="Frequency",
        spacing=xvals,
        heights=count,
        labels=word)
    plt.savefig("figs/kwF.png")

def displayPlaylistLengthDistribution(df):
    """
    Displays histogram of playlist lengths,
    allows us to get a sense of the size of
    a typical playlist 
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    numTracks = [len(x) for x in df.tracks]
    plotHist(
        ax=ax, 
        title= "Distribution of Number of Tracks per Playlist",
        xlabel="Number of Tracks",
        ylabel="Distribution",
        data=numTracks)
    plt.savefig("figs/playlistLengthDist.png")
