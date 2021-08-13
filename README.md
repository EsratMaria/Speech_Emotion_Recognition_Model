# Speech Emotion Detection Classifier

Emotion recognition is the part of speech recognition which is gaining more popularity and need for it increases enormously. In this repo, I attempt to use deep learning to recognize the emotions from data.

## Dataset
- Crowd-sourced Emotional Mutimodal Actors Dataset (Crema-D)
- Emotions included in this dataset: ``sad, angry, disgust, neutral, happy, and fear``
  - Each path to the audio is extracted with it's associated emotion.
### Emotions in the dataset
![emo](/images/emo.png)
### Waveplot of a sample audio
![waveplot](/images/wave.png)
### Spectogram of a sample audio
![spec](/images/spec.png)

## How it works
I use this dataframe of extracted features to train the model later.
