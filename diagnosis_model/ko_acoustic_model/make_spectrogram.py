import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


def make_spectrogram_chunks(audio_samples, sr, image_transform):
    chunk_len = 5 * sr
    chunks = []
    for i in range(0, len(audio_samples), chunk_len):
        y_part = audio_samples[i:i + chunk_len]
        if len(y_part) < chunk_len:
            continue
        mel = librosa.feature.melspectrogram(y=y_part, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        fig = Figure(figsize=(2, 2), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        librosa.display.specshow(log_mel, sr=sr, ax=ax)
        ax.axis('off')
        fig.tight_layout(pad=0)
        canvas.draw()

        width, height = canvas.get_width_height()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
        img = img[:, :, :3]

        pil_img = Image.fromarray(img)
        img_tensor = image_transform(pil_img)
        chunks.append(img_tensor)

        plt.close(fig)
    return chunks