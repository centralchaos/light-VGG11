# data prep by CarloSia

import os
import numpy as np
import librosa
import torch

# Parameters
audio_file_path = './sound_data/rsif20210921_si_001.wav'  # Path to the audio file
output_directory = 'processed_data'  # Directory to save spectrogram images and data
segment_length = 1  # Segment length in seconds

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Load the audio file
audio, sr = librosa.load(audio_file_path, sr=None)

# Calculate the number of segments
num_segments = int(len(audio) // (sr * segment_length))

# List to hold all normalized Log-Mel spectrograms
normalized_spectrograms = []

for i in range(num_segments):
    # Get the start and end of the segment
    start_sample = i * sr * segment_length
    end_sample = start_sample + sr * segment_length
    segment = audio[start_sample:end_sample]

    # Check if the segment is less than 1 second
    if len(segment) < sr * segment_length:
        continue  # Skip this segment if it's shorter than 1 second

    # Convert to Log-Mel Spectrogram
    n_fft = 2048  # Length of the FFT window
    hop_length = 512  # Number of samples between each frame
    n_mels = 64  # Number of Mel bands

    mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert to Log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Normalize the Log-Mel spectrogram to range [0, 1]
    normalized_log_mel = (log_mel_spectrogram - log_mel_spectrogram.min()) / (log_mel_spectrogram.max() - log_mel_spectrogram.min())

    # Convert to PyTorch tensor and add a channel dimension
    normalized_spectrograms.append(torch.tensor(normalized_log_mel).unsqueeze(0))  # Add a channel dimension

# Convert list to tensor for easy indexing
normalized_spectrograms = torch.stack(normalized_spectrograms)

# Tagging
labels = torch.full((normalized_spectrograms.size(0),), 2, dtype=torch.long)  # Tag all data as '2'

# Save all data in a single .pt file
archive_path = os.path.join(output_directory, 'archive.pt')
torch.save((normalized_spectrograms, labels), archive_path)

print(f'Spectrograms and archive saved to {output_directory}')

