# Spotter+Biometry Project

## Data

We have 3 open datasets:
- English Mozilla Common Voice: https://commonvoice.mozilla.org/en/datasets
- All languages Mozilla Common Voice
- Our own recordings in Russian: TODO

We need split all sets on train/test by speakers to train/test biometry model. Also we need get only top words/phrases (top 100 or top 200 for example) by frequency from sets to train/test keyword spotting model.

## Model

Main paper on this theme: https://arxiv.org/abs/2104.13970

### Features

Most popular features for speech recognition is FBANK: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html.

After features transform we get tensor of shape (frames x features), so we can use different models:
- conv1d (frames as picture shape, features as features)
- conv2d (framex x features as picture shape, unsqueeze as features shape)
- rnn (frames as sequence, features as features)

MFCC features also can bring some profit, but rather to to spotter, that to biometry. Also MFCC loses spatial dependence in features shape, which leads to a deterioration in quality of conv2d models.

### Spotter

Some papers about spotter:
- https://arxiv.org/pdf/1812.02802.pdf
- https://arxiv.org/pdf/2005.06720.pdf
- https://arxiv.org/pdf/2004.08531.pdf

### Biometry

Some papers about biometry:
- https://arxiv.org/pdf/2104.01989.pdf

## Inference


- Enrollment: speakers pronounce some count of phrases and biometry model calculate speaker embedding
- Solve keyword spotting task with biometry for given speaker on some set of wav files
- Solve keyword spotting task with biometry for given speaker on sound flow