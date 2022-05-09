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
First I extract features from each audio data in the dataset since provided audio cannot be understood by the models directly so I need to convert them into an understandable format.

In this repo, I extract features like **MFCC** and **mel-spectogram** from each audio file in the dataset. The extracted data is added to a new dataframe with it's assiciated emotion.
I use this dataframe of extracted features to train the model later.

> MFCC: Mel Frequency Cepstral Coefficients form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale.

Later, this dataframe is normalized to prepare it for training and testing. 
- StandardScalar() - fit transform
- train_test_split()

### Data Augmentation Techniques
With this technique, I try to create new synthetic data samples by adding minor modifications on the initial training set. I apply **Noise Injection**  to make synthetic data in this repo. 
```
noise_amp = 0.035 * np.random.uniform() * np.amax(value)
```
## Model & Prediction accuracy:
> Used Model: MLPClassifier

> MLPClassifier(alpha=0.839903176695813, batch_size=150, hidden_layer_sizes=100,
              learning_rate='adaptive', max_iter=100000, solver='sgd')
              
The prediction made by above model to detect the emotion of a given audio is given below:
```
<<<===========================================>>>
       Actual  Predict
1011    angry    angry
1689  neutral  neutral
6092    angry    angry
6231    angry  disgust
7334  neutral  disgust

```

