"""
Data type definitions for the transcription system.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TypedDict, Dict, List, Any, Union, Optional, Tuple, Literal
from pathlib import Path

# Specific types for data
TaskType = Literal["transcribe", "translate"]

class AudioSourceInfo(TypedDict):
    """Information about the source of processed audio."""
    path: Optional[str]  # Original path or URL
    type: str  # "file", "url", "bytes" or "dict"
    file_name: Optional[str]  # File name if available
    format: str  # File format (mp3, wav, mp4, etc.)
    is_video: bool  # Whether the source is a video file
    duration_seconds: Optional[float]  # Duration in seconds
    content_size: Optional[int]  # Size in bytes if available

class AudioData(TypedDict):
    """Processed audio data ready for transcription/diarization."""
    numpy_array: "np.ndarray"  # Mono 16kHz audio array
    torch_tensor: "torch.Tensor"  # Tensor for diarization
    sampling_rate: int  # Always 16000Hz
    source_info: AudioSourceInfo  # Source information

class AudioInputDict(TypedDict):
    """Audio input dictionary with metadata."""
    raw: Optional["np.ndarray"]  # Raw audio data
    array: Optional["np.ndarray"]  # Alternative to raw
    sampling_rate: int  # Sampling rate
    path: Optional[str]  # Audio source (if available)

class TranscriptChunk(TypedDict):
    """Individual transcription chunk with timestamp."""
    text: str
    timestamp: Tuple[Optional[float], Optional[float]]

class TranscriptOutput(TypedDict):
    """Complete transcription result."""
    text: str  # Complete text
    chunks: List[TranscriptChunk]  # Chunks with timestamps

class SpeakerSegment(TypedDict):
    """Segment with speaker information."""
    segment: Dict[str, float]  # start, end in seconds
    speaker: str  # Speaker identifier (SPEAKER_00, etc.)

class DiarizedChunk(TranscriptChunk):
    """Transcription chunk with speaker information."""
    speaker: str  # Speaker identifier

class ProcessingMetadata(TypedDict):
    """Metadata about the processing performed."""
    transcription_model: str
    language: str
    device: str
    timestamp: str
    diarization: Optional[bool]
    diarization_model: Optional[str]

class FinalResult(TypedDict):
    """Final result structure."""
    speakers: List[DiarizedChunk]  # Transcription with speakers
    chunks: List[TranscriptChunk]  # Original chunks
    text: str  # Complete text
    metadata: Dict[str, Any]  # Metadata, includes source_info and processing

@dataclass(frozen=True)
class TranscriptionConfig:
    """Immutable configuration for the transcription process."""
    file_name: Union[str, Path]
    device_id: str = "0"
    transcript_path: Union[str, Path] = "output.json"
    model_name: str = "openai/whisper-large-v3"
    task: TaskType = "transcribe"
    language: str = "es"
    batch_size: int = 8
    output_format: str = "json"
    chunk_length: int = 30
    attn_type: str = "sdpa"
    hf_token: str = "no_token"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarize: bool = False
    speaker_names: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError(
                "--num-speakers cannot be used together with --min-speakers or --max-speakers"
            )