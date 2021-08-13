# Speech Emotion Detection Classifier

Emotion recognition is the part of speech recognition which is gaining more popularity and need for it increases enormously. In this repo, I attempt to use deep learning to recognize the emotions from data.

## Dataset
- Crowd-sourced Emotional Mutimodal Actors Dataset (Crema-D)
- Emotions included in this dataset: ``sad, angry, disgust, neutral, happy, and fear``
  - Each path to the audio is extracted with it's associated emotion.
#### Emotions count in the dataset
![emo](/images/emo.png)
#### Waveplot of a sample audio
`` Waveplots let us know the loudness of the audio at a given time.``

![waveplot](/images/wave.png)
#### Spectogram of a sample audio
``A spectrogram is a visual representation of the spectrum of frequencies of sound or other signals as they vary with time. ``

![spec](/images/spec.png)

## How it works
In this repo, I extract features like **MFCC** and **mel-spectogram** from each audio file in the dataset. The extracted data is added to a new dataframe with it's assiciated emotion.
I use this dataframe of extracted features to train the model later.
