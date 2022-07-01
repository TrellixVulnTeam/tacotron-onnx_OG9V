#pip install numpy scipy librosa unidecode inflect librosa
# brew update
# brew install libsndfile1
# move the libsndfiles manually to cona environment
import sys
sys.path.append("/Users/leon.luithlen/miniconda3/envs/ml/lib/python3.10/site-packages/")
import torch
from inference import prepare_input_sequence


tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', pretrained=False)
checkpoint = torch.hub.load_state_dict_from_url('https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_fp32/versions/1/files/nvidia_tacotron2pyt_fp32_20190306.pth', map_location="cpu")
state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
tacotron2.load_state_dict(state_dict)
tacotron2 = tacotron2.to('cpu')
tacotron2.eval()



waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16', pretrained=False)
checkpoint = torch.hub.load_state_dict_from_url("https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth", map_location='cpu',)
state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
waveglow.load_state_dict(state_dict)
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cpu')
waveglow.eval()

text = "Hello world, Labour’s Yvette Cooper said that removing the whip from the former Conservative deputy whip needed to be the “first step that takes place” but did not call for his resignation as an MP. Pincher, 52, resigned from his role as deputy chief whip on Thursday night after admitting he had “embarrassed myself and other people” following reports that he drunkenly groped two men at the Carlton Club in Piccadilly, London, on Wednesday."
sequences, lengths = prepare_input_sequence([text])

with torch.no_grad():
    mel, _, _ = tacotron2.infer(sequences, lengths)
    audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050

from scipy.io.wavfile import write
write("audio.wav", rate, audio_numpy)
