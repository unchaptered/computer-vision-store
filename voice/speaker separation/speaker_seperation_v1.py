import torch
import torchaudio
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    import os
    
    absPath = os.getcwd()
    
    vidPathList = [
        './assets/sounds/interview.wav',
    ]
    
    for vidPath in vidPathList:    
        totalPath = os.path.join(absPath, vidPath)
        
        waveform, sample_rate = torchaudio.load(vidPath)
        
        # Mel-spectrogram 변환
        transform = transforms.MelSpectrogram(sample_rate=sample_rate)
        mel_specgram = transform(waveform)

        # Mel-spectrogram 시각화
        plt.figure()
        plt.imshow(torch.log(mel_specgram[0]), cmap='gray')
        plt.title('Mel spectrogram')
        plt.show()
