# 🎙️ Chapter 29: Audio & Speech Processing with Neural Networks

## 29.1 Audio Signal Processing Fundamentals
Audio signals are continuous 1D pressure waveforms sampled at high frequencies (e.g. 16,000 Hz or 44,100 Hz).

```
1D Waveform s(t) ──► Short-Time Fourier Transform (STFT) ──► 2D Mel-Spectrogram (Time x Frequency)
```

### 1. Short-Time Fourier Transform (STFT)
Applies Discrete Fourier Transform (DFT) over overlapping sliding time windows $w(t)$:

$$\mathbf{STFT}\{x[n]\}(m, \omega) = \sum_{n=-\infty}^\infty x[n] w[n - m] e^{-j \omega n}$$

### 2. Mel-Spectrogram
Maps linear hertz frequencies $f$ onto the non-linear human auditory **Mel Scale**:

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

Converts 1D raw audio waveforms into 2D image-like spectrogram matrices $\mathbf{S} \in \mathbb{R}^{F \times T}$, enabling standard 2D CNNs and ViTs to process audio.

---

## 29.2 Connectionist Temporal Classification (CTC Loss)

In Automatic Speech Recognition (ASR), 1D audio frame sequences are much longer than output text character sequences, and frame-level character alignments are unobserved.

```
Audio Frames:  [ h  h  h  e  e  _  _  l  l  l  o  o ]  (N = 12)
CTC Collapsed: [ h  e  l  o ]                         (T = 4)
```

### CTC Loss Formulation:
Introduces a blank token $\epsilon$. Defines a many-to-one collapsing mapping $\mathcal{B}$ that removes repeated adjacent characters and blanks:

$$P(Y \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^T y_{\pi_t}^t$$

---

## 29.3 Audio Foundation Models: OpenAI Whisper

Whisper (Radford et al. 2022) is an Encoder-Decoder Transformer trained on 680,000 hours of multilingual audio.

```
80-Channel Log-Mel Spectrogram ──► Conv1D Subsampling ──► Transformer Encoder ──► Transformer Decoder ──► Text Tokens
```

- **Tasks**: Automatic Speech Recognition (ASR), Multilingual Translation, Voice Activity Detection (VAD), and Speaker Identification.

---

## ⚓ Repository Code Reference
- See [`src/sentiment_analysis/pipeline.py`](file:///e:/Downloads/ML_only/src/sentiment_analysis/pipeline.py) for sequence classification principles.
