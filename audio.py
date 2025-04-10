"""
Functions for audio processing.
"""
import os
import requests
from pathlib import Path
from typing import Dict, Any, Union

from helpers import log_time, with_imports, logger, format_path
from data_types import AudioData, AudioSourceInfo

@log_time
@with_imports("numpy", "torch", "transformers.pipelines.audio_utils", "torchaudio.functional", "os", "subprocess")
def process_audio(
    input_src: Union[str, Path, bytes, Dict[str, Any]],
    *,
    dynamic_imports: Dict[str, Any] = {}
) -> AudioData:
    """
    Processes audio from various sources for ASR and diarization.
    Automatically detects video files and extracts audio.
    
    Args:
        input_src: String (path/URL), bytes or dictionary with audio data.
        dynamic_imports: Dynamically imported modules.
        
    Returns:
        AudioData: Dictionary with numpy_array, torch_tensor and sampling_rate.
    """
    # Initialize source information
    source_info: AudioSourceInfo = {
        "path": None,
        "type": "unknown",
        "file_name": None,
        "format": "unknown",
        "is_video": False,
        "duration_seconds": None,
        "content_size": None
    }
    
    # Extract modules
    np = dynamic_imports["numpy"]
    torch = dynamic_imports["torch"]
    audio_utils = dynamic_imports["audio_utils"]
    functional = dynamic_imports["functional"]
    os = dynamic_imports["os"]
    subprocess = dynamic_imports["subprocess"]

    # Processing according to input type
    if isinstance(input_src, (str, Path)):
        input_str: str = str(input_src)
        source_info["path"] = input_str
        source_info["file_name"] = os.path.basename(input_str)
        
        _, file_ext = os.path.splitext(input_str.lower())
        source_info["format"] = file_ext.lstrip('.')
        source_info["is_video"] = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if input_str.startswith(("http://", "https://")):
            source_info["type"] = "url"
            logger.info(f"Processing audio from URL: {format_path(input_str)} ({'' if source_info['is_video'] else 'audio, '}{source_info['format']})")
            logger.info("Downloading audio from URL")
            
            response = requests.get(input_str, stream=True)
            content_size = int(response.headers.get('content-length', 0))
            source_info["content_size"] = content_size
            
            input_src = response.content
            logger.info(f"Download completed: {content_size / (1024*1024):.1f} MB received")
        else:
            source_info["type"] = "file"
            logger.info(f"Processing audio from file: {format_path(input_str)} ({source_info['format']})")
            logger.info("Loading file from local storage")
            
            if source_info["is_video"]:
                logger.info(f"Detected video file: {file_ext}")
                temp_audio = f"{os.path.splitext(input_str)[0]}_temp.wav"
                logger.info(f"Extracting audio to temporary file: {format_path(temp_audio)}")
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", input_str, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio],
                        check=True,
                        capture_output=True
                    )
                    # Check that the file was generated correctly
                    if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
                        raise ValueError("The temporary audio file was not generated correctly or is empty")
                    
                    logger.info(f"Audio extracted successfully, size: {os.path.getsize(temp_audio)/1024:.1f} KB")
                    with open(temp_audio, "rb") as f:
                        input_src = f.read()
                        logger.info(f"Bytes read from temporary file: {len(input_src)} bytes")
                    os.remove(temp_audio)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error extracting audio with FFmpeg: {e}")
                    logger.error(f"Error output: {e.stderr.decode() if e.stderr else 'No stderr'}")
                    raise ValueError(f"Could not extract audio from video file: {input_str}") from e
                except FileNotFoundError:
                    logger.error("FFmpeg is not installed or not found in PATH")
                    raise ValueError("FFmpeg is necessary to process video files")
            else:
                with open(input_str, "rb") as f:
                    input_src = f.read()

    # Processing binary data with robust error handling
    if isinstance(input_src, bytes):
        source_info["type"] = "bytes"
        logger.info(f"Processing audio from binary data: {len(input_src) / 1024:.1f} KB")
        source_info["content_size"] = len(input_src)
        logger.info("Decoding audio with FFmpeg")
        
        try:
            # Add detailed logging for diagnostics
            logger.info(f"Bytes size before FFmpeg: {len(input_src)} bytes")
            decoded_audio = audio_utils.ffmpeg_read(input_src, 16000)
            
            # Explicitly verify it was decoded correctly
            if not isinstance(decoded_audio, np.ndarray):
                logger.error(f"FFmpeg did not return a numpy array, but {type(decoded_audio).__name__}")
                raise ValueError(f"Incorrect decoding: expected numpy.ndarray, got {type(decoded_audio).__name__}")
                
            input_src = decoded_audio
            logger.info(f"Audio decoded successfully: {input_src.shape[0]} samples")
            
        except Exception as e:
            logger.error(f"Error decoding audio with FFmpeg: {str(e)}", exc_info=True)
            raise ValueError("Could not decode audio correctly. Check the file format and FFmpeg installation.") from e

    elif isinstance(input_src, dict):
        source_info["type"] = "dict"
        if "path" in input_src:
            source_info["path"] = input_src["path"]
            source_info["file_name"] = os.path.basename(str(input_src["path"]))
        
        logger.info(f"Processing audio from dictionary: {source_info.get('path', 'in-memory data')}")
        
        if not ("sampling_rate" in input_src and ("raw" in input_src or "array" in input_src)):
            raise ValueError("Dictionary must contain 'raw' or 'array' with audio and 'sampling_rate'")
        
        _inputs: Any = input_src.pop("raw", None)
        if _inputs is None:
            input_src.pop("path", None)
            _inputs = input_src.pop("array", None)
        
        in_sampling_rate: int = input_src.pop("sampling_rate")
        input_src = _inputs
        
        if in_sampling_rate != 16000:
            logger.info(f"Resampling from {in_sampling_rate} to 16000 Hz")
            input_src = functional.resample(torch.from_numpy(input_src), in_sampling_rate, 16000).numpy()

    # Final data type verification
    if not isinstance(input_src, np.ndarray):
        error_msg = f"Expected a numpy array, got `{type(input_src).__name__}`"
        logger.error(error_msg)
        if isinstance(input_src, bytes):
            logger.error(f"Data is still bytes of size {len(input_src)}. FFmpeg decoding failed.")
        raise ValueError(error_msg)
        
    if len(input_src.shape) != 1:
        raise ValueError("Expected single-channel audio")

    # Prepare tensor for diarization
    input_src = input_src.copy() 
    torch_tensor = torch.from_numpy(input_src).float().unsqueeze(0)
    
    # Calculate duration
    duration_seconds = len(input_src) / 16000
    source_info["duration_seconds"] = duration_seconds
    duration_str = f"{int(duration_seconds//60)}m {int(duration_seconds%60)}s"
    
    logger.info(f"Audio processed: {input_src.shape[0]} samples, SR=16000Hz (duration: {duration_str})")
    
    return {
        "numpy_array": input_src,
        "torch_tensor": torch_tensor,
        "sampling_rate": 16000,
        "source_info": source_info
    }