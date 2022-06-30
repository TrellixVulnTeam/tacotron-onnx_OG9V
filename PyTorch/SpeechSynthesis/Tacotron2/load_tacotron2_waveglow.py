#pip install numpy scipy librosa unidecode inflect librosa
# brew update
# brew install libsndfile1
# move the libsndfiles manually to cona environment



tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', pretrained=False)
checkpoint = torch.hub.load_state_dict_from_url('https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_fp32/versions/1/files/nvidia_tacotron2pyt_fp32_20190306.pth', map_location="cpu")
state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
tacotron2.load_state_dict(state_dict)


waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16', pretrained=False)
checkpoint = torch.hub.load_state_dict_from_url("https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth", map_location='cpu',)
state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
waveglow.load_state_dict(state_dict)