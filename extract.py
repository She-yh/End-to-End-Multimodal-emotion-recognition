import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
import librosa
import functools
import moviepy.editor as mp

import transforms
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=(720, 1280), device=device)
save_frames = 15
input_fps = 30

save_length = 3.6  # seconds

select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]

def extract_faces(filename):
    if filename.endswith('.mp4'):
        cap = cv2.VideoCapture(filename)
        # calculate length in frames
        framen = 0
        while True:
            i, q = cap.read()
            if not i:
                break
            framen += 1
        cap = cv2.VideoCapture(filename)

        if save_length * input_fps > framen:
            skip_begin = int((framen - (save_length * input_fps)) // 2)
            for i in range(skip_begin):
                _, im = cap.read()

        framen = int(save_length * input_fps)
        frames_to_select = select_distributed(save_frames, framen)

        numpy_video = []
        frame_ctr = 0

        while True:
            ret, im = cap.read()
            if not ret:
                break
            if frame_ctr not in frames_to_select:
                frame_ctr += 1
                continue
            else:
                frames_to_select.remove(frame_ctr)
                frame_ctr += 1

            temp = im[:, :, -1]
            im_rgb = im.copy()
            im_rgb[:, :, -1] = im_rgb[:, :, 0]
            im_rgb[:, :, 0] = temp
            im_rgb = torch.tensor(im_rgb)
            im_rgb = im_rgb.to(device)

            bbox = mtcnn.detect(im_rgb)
            if bbox[0] is not None:
                bbox = bbox[0][0]
                bbox = [round(x) for x in bbox]
                x1, y1, x2, y2 = bbox
            im = im[y1:y2, x1:x2, :]
            im = cv2.resize(im, (224, 224))
            numpy_video.append(im)
        if len(frames_to_select) > 0:
            for i in range(len(frames_to_select)):
                numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))

        video = np.array(numpy_video)
        video_data = []
        for i in range(np.shape(video)[0]):
            video_data.append(Image.fromarray(video[i, :, :, :]))
        inputs_visual = video_data
        spatial_transform = transforms.Compose([
            transforms.ToTensor(255)])
        spatial_transform.randomize_parameters()
        inputs_visual = [spatial_transform(img) for img in inputs_visual]
        inputs_visual = torch.stack(inputs_visual, 0).permute(1, 0, 2, 3)
        inputs_visual = inputs_visual.unsqueeze(0)
        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1], inputs_visual.shape[2],
                                              inputs_visual.shape[3], inputs_visual.shape[4])
        return inputs_visual

def extract_audio(audiofile):
    target_time = 3.6
    audio_mp = mp.VideoFileClip(audiofile).audio
    audio_mp.write_audiofile('./raw_data/audio_mp.wav')
    audios = librosa.core.load('./raw_data/audio_mp.wav',sr=22050)
    y = audios[0]
    sr = audios[1]
    target_length = int(sr * target_time)
    # make sure the length of the audio is 3.6s.
    if len(y) < target_length:
        y = np.array(list(y) + [0 for i in range(target_length - len(y))])
    else:
        remain = len(y) - target_length
        y = y[remain // 2:-(remain - remain // 2)]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return torch.FloatTensor(mfcc).unsqueeze(0)

def extract(filename):
    inputs_visual = extract_faces(filename)
    inputs_audio = extract_audio(filename)
    return inputs_visual, inputs_audio

def video_loader(video_dir_path):
    video = np.load(video_dir_path)
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)
