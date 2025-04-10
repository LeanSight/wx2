# WX2 - Fast Audio Transcription and Diarization System with Whisper and Pyannote

WX2 is a functional Python tool for transcribing audio and video, with speaker identification (diarization) capabilities. It uses state-of-the-art models like Whisper for transcription and PyAnnote for diarization. This project is inspired by [Insanely Fast Whisper](https://github.com/Vaibhavs10/insanely-fast-whisper).

## 📋 Main Features

- **Multiple Source Processing**: Local files, URLs, binary data
- **Video Support**: Automatic audio extraction from video files
- **Multi-language Transcription**: Compatible with all languages supported by Whisper
- **Speaker Diarization**: Automatically identifies different speakers in the conversation
- **Multiple Output Formats**: JSON, SRT, VTT, and plain text
- **Speaker Customization**: Allows assigning names to detected speakers
- **Performance Optimized**: Support for CUDA, MPS (Apple Silicon), and CPU

## 🔧 Prerequisites

- Python 3.10 (tested version)
- FFmpeg (for audio processing)
- CUDA-compatible graphics card (optional, for better performance)

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/LeanSight/wx2.git
cd wx2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For CUDA support (recommended for GPU acceleration):
   - Follow this tutorial: [Installing CUDA for PyTorch Easily Explained](https://medium.com/@fernandopalominocobo/installing-cuda-for-pytorch-easily-explained-windows-users-4d3b7db5f2e0)
   - Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

4. To use diarization, you'll need a Hugging Face token:
   - Register at [Hugging Face](https://huggingface.co/)
   - Get an access token
   - Accept the terms of the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model

## 🚀 Basic Usage

### Simple Transcription

```bash
python wx2.py file.mp3
```

### Transcription with Specific Format

```bash
python wx2.py video.mp4 -f srt
```

### Transcription with Diarization (Speaker Identification)

```bash
python wx2.py meeting.wav --diarize --token=hf_xxx
```

### Transcription with Custom Speaker Names

```bash
python wx2.py interview.mp3 --diarize --token=hf_xxx --speaker-names="John,Mary,Peter"
```

### Translation to English Instead of Transcription

```bash
python wx2.py speech.mp3 --task translate
```

### Using Specific Hardware Resources

```bash
# Use specific GPU
python wx2.py audio.mp3 --device 0

# Use CPU
python wx2.py audio.mp3 --device cpu

# Use Apple Silicon (M1/M2)
python wx2.py audio.mp3 --device mps
```

## 🛠️ Advanced Options

```
Basic options:
  -o, --output         Path to save the result
  -f, --format         Output file format (json, srt, vtt, txt)

Transcription options:
  -m, --model          Transcription model to use
  -l, --lang           Audio language (ISO code, e.g. 'es', 'en', 'fr')
  -t, --task           Task to perform: transcribe or translate
  --chunk-length       Duration in seconds of each audio fragment

Performance options:
  -d, --device         GPU device ID or 'cpu' or 'mps'
  -b, --batch-size     Batch size for GPU processing
  --attn-type          Attention implementation type (sdpa, eager, flash)

Diarization options:
  --diarize            Enable speaker identification
  --dmodel             Diarization model to use
  --token              HuggingFace token for diarization models
  --num-speakers       Exact number of speakers in the audio
  --min-speakers       Minimum number of speakers to detect
  --max-speakers       Maximum number of speakers to detect
  --speaker-names      List of names to replace speaker labels
```

## 🔄 Format Conversion

You can also convert existing transcriptions to different formats using the `output_convert.py` script:

```bash
python output_convert.py transcription.json -f srt
```

## 📦 Project Structure

```
├── wx2.py               # Main script
├── audio.py             # Audio processing
├── transcription.py     # Transcription functions
├── diarization.py       # Diarization functions
├── formatters.py        # Conversion to different formats
├── output_convert.py    # Independent conversion tool
├── data_types.py        # Data type definitions
├── helpers.py           # Utilities and decorators
├── requirements.txt     # Dependencies
└── wx2.spec             # PyInstaller configuration
```

## 🏗️ Packaging Possibilities

The project includes a `.spec` file for packaging the tool with PyInstaller:

```bash
pyinstaller wx2.spec
```

This will create a standalone executable in the `dist/wx2` folder.

## 📝 Notes

- Diarization requires a Hugging Face token.
- For video files, FFmpeg must be installed on the system.
- The default transcription model is `openai/whisper-large-v3`.
- The default diarization model is `pyannote/speaker-diarization-3.1`.

## 🤝 Contributions

Contributions are welcome. Please feel free to submit a Pull Request.

## 📄 License

[Apache License](https://www.apache.org/licenses/LICENSE-2.0)

## 🙏 Acknowledgements

This project is inspired by [Insanely Fast Whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) and builds upon its approach to efficient audio transcription.

---

For any issues or suggestions, please open an issue in the repository: https://github.com/LeanSight/wx2