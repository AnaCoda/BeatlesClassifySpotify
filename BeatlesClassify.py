import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Get the Spotify credentials (I had to use my own Spotify developer Client ID and Secret)
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
scope = 'user-library-read'

# This excel document is from https://www.kaggle.com/chadwambles/allbeatlesspotifysongdata2009remaster and uses the same Spotify metrics as I get for whatever song is input
raw_data = pd.read_excel(R'TheBeatlesData.xlsx')

# Get the relevant columns from the data and turn it into a Pandas dataframe
SongInfo = pd.DataFrame(raw_data, columns=['song', 'album', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence'])

# Album names
Albums = SongInfo.iloc[:,1]
# Song names
Songs = SongInfo.iloc[:,0]
# Song metrics (danceability, tempo, etc.)
X = SongInfo.iloc[:,2:9]

# Get the audio features of a track (can just paste Spotify URL) and turn it into Pandas Dataframe
songFeatures = pd.DataFrame.from_dict(sp.audio_features(tracks=['https://open.spotify.com/track/0eI5SOdxDTSEGkUppeWKls']))
# Only keep the metrics
songFeatures = songFeatures[['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]

# Create a song dataframe with the song names
ySong = pd.DataFrame()
ySong['song'] = Songs
# Create a column with numerical versions of the song names
ySong['cats'] = pd.Categorical(Songs).codes

# Do the same thing with the albums
yAlbums = pd.DataFrame()
yAlbums['album'] = Albums
yAlbums['cats'] = pd.Categorical(Albums).codes

# Create a KNeighborClassifier (only 1 nearest, because there's only 1 of each song)
neighSong = KNeighborsClassifier(n_neighbors=1)
# Fit the Classifier on the Excel song info with the number categories as labels. Ravel() to shape it correctly
neighSong.fit(X, ySong.cats.ravel())

# Same thing with album, except we use 5 nearest neighbours (there are many songs in each album)
neighAlbum = KNeighborsClassifier(n_neighbors=5)
neighAlbum.fit(X, yAlbums.cats.ravel())

while True:
    spotTrack = input("Enter song URL: ")
    # Get the audio features of a track (can just paste Spotify URL) and turn it into Pandas Dataframe
    songFeatures = pd.DataFrame.from_dict(sp.audio_features(tracks=[spotTrack]))
    # Only keep the metrics
    songFeatures = songFeatures[['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]

    # Find the ID of the closest song and album
    ClosestSong = neighSong.predict(songFeatures)[0]
    ClosestAlbum = neighAlbum.predict(songFeatures)[0]

    # Predict the probability of the nearest album to print
    ClosestAlbumProb = neighAlbum.predict_proba(songFeatures)
    probNum = ClosestAlbumProb[0][ClosestAlbum]

    songName = ySong.loc[ySong['cats'] == ClosestSong].song.iloc[0]
    albumName = yAlbums.loc[yAlbums['cats'] == ClosestAlbum].album.iloc[0]
    # Print the results
    print('The nearest Beatles song is: ' + songName)
    print('The nearest Beatles album is: ' + albumName + ', with ' + str(probNum * 100) + '% probability' )
    print('Your song\'s features: ')
    print(songFeatures)
    print(songName + '\'s features: ')
    print(SongInfo.loc[SongInfo['song'] == songName].iloc[:,2:10])
