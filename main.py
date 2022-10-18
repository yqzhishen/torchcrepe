import numpy as np

import torchcrepe


# Load audio
audio, sr = torchcrepe.load.audio(r'data/xtgg_mono_16k_denoise.wav')

# Here we'll use a 5 millisecond hop length
hop_length = int(sr / 200.)

# Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
# This would be a reasonable range for speech
fmin = 50
fmax = 1100

# Select a model capacity--one of "tiny" or "full"
model = 'full'

# Choose a device to use for inference
device = 'cpu'

# Pick a batch size that doesn't cause memory errors on your gpu
batch_size = 512

# Compute pitch using first gpu
pitch = torchcrepe.predict(audio,
                           sr,
                           hop_length,
                           fmin,
                           fmax,
                           model,
                           batch_size=batch_size,
                           device=device)
pitch = pitch.detach().cpu().numpy()
print(np.mean(pitch))
print(np.var(pitch))
