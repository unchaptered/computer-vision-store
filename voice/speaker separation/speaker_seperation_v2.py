# # MODULE : https://github.com/juanmc2005/diart
# # INSTALLATION : conda install -c conda-forge sox

# import sounddevice as sd
# print(sd.query_devices())

import torchaudio
torchaudio.set_audio_backend("soundfile")

from diart import OnlineSpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import RealTimeInference
from diart.sinks import RTTMWriter

pipeline = OnlineSpeakerDiarization()
mic = MicrophoneAudioSource(pipeline.config.sample_rate)
# mic = MicrophoneAudioSource(
#     sample_rate=pipeline.config.sample_rate,
#     block_size=1000,
#     device=1)
inference = RealTimeInference(pipeline, mic, do_plot=True)
inference.attach_observers(RTTMWriter(mic.uri, "/assets/sounds/out/file.rttm"))
prediction = inference()
