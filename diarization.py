"""
Functions for diarization (speaker identification) of audio.
"""
import sys
from typing import Dict, Any, List, Tuple, cast

from helpers import log_time, with_imports, with_progress_bar, logger, format_path
from data_types import (
    TranscriptionConfig, AudioData, TranscriptOutput, 
    TranscriptChunk, DiarizedChunk
)

@with_imports("torch", "pyannote.audio")
def load_diarization_pipeline(
    config: TranscriptionConfig,
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> Any:
    """
    Loads the diarization pipeline.
    
    Args:
        config: Transcription/diarization configuration
        dynamic_imports: Dynamically imported modules
        
    Returns:
        Pipeline: Loaded diarization pipeline
    """
    torch = dynamic_imports["torch"]
    pyannote_audio = dynamic_imports["audio"]
    
    pipeline = pyannote_audio.Pipeline.from_pretrained(
        checkpoint_path=config.diarization_model,
        use_auth_token=config.hf_token,
    )
    
    device = torch.device("mps" if config.device_id == "mps" 
                         else "cpu" if config.device_id == "cpu" 
                         else f"cuda:{config.device_id}")
    pipeline.to(device)
    
    return pipeline

def process_diarization_segments(diarization: Any) -> List[Dict[str, Any]]:
    """
    Processes diarization segments and combines them by speaker.
    
    Args:
        diarization: Diarization result
        
    Returns:
        List[Dict[str, Any]]: Combined segments by speaker
    """
    # Extract segments
    segments: List[Dict[str, Any]] = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append({
            "segment": {"start": segment.start, "end": segment.end},
            "track": track,
            "label": label,
        })
    
    # Combine consecutive segments from the same speaker
    new_segments: List[Dict[str, Any]] = []
    if segments:
        prev_segment: Dict[str, Any] = segments[0]
        
        for i in range(1, len(segments)):
            cur_segment: Dict[str, Any] = segments[i]
            
            # If speaker changed, add the combined segment
            if cur_segment["label"] != prev_segment["label"]:
                new_segments.append({
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                })
                prev_segment = segments[i]
        
        # Add the last segment
        new_segments.append({
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": segments[-1]["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        })
    
    return new_segments

@with_imports("numpy")
def align_segments_with_transcript(
    new_segments: List[Dict[str, Any]], 
    transcript_chunks: List[TranscriptChunk],
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> List[DiarizedChunk]:
    """
    Aligns diarization segments with the transcript.
    
    Args:
        new_segments: Diarization segments
        transcript_chunks: Transcript fragments
        dynamic_imports: Dynamically imported modules
        
    Returns:
        List[DiarizedChunk]: Fragments with speaker information
    """
    np = dynamic_imports["numpy"]
    segmented_preds: List[DiarizedChunk] = []
    
    if not new_segments or not transcript_chunks:
        return segmented_preds
    
    # Get final timestamps of each transcribed chunk
    end_timestamps = np.array([
        chunk["timestamp"][-1] if chunk["timestamp"][-1] is not None 
        else sys.float_info.max for chunk in transcript_chunks
    ])
    
    # Align diarization and ASR timestamps
    for segment in new_segments:
        end_time: float = segment["segment"]["end"]
        # Find the closest ASR timestamp
        upto_idx: int = np.argmin(np.abs(end_timestamps - end_time))
        
        # Add chunks with speaker information
        for i in range(upto_idx + 1):
            chunk: TranscriptChunk = transcript_chunks[i]
            segmented_preds.append({
                "text": chunk["text"],
                "timestamp": chunk["timestamp"],
                "speaker": segment["speaker"]
            })
        
        # Trim the transcript and timestamps for the next segment
        transcript_chunks = transcript_chunks[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]
        
        if len(end_timestamps) == 0:
            break
    
    return segmented_preds

@log_time
@with_imports("torch", "pyannote.audio", "numpy")
def diarize_audio(
    config: TranscriptionConfig, 
    audio_data: AudioData, 
    transcript: TranscriptOutput,
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> List[DiarizedChunk]:
    """
    Performs diarization to identify different speakers.
    
    Args:
        config: Transcription/diarization configuration
        audio_data: Processed audio for diarization
        transcript: Transcription results
        dynamic_imports: Dynamically imported modules
        
    Returns:
        List[DiarizedChunk]: Transcription with speaker information
    """
    # If no token, skip diarization
    if config.hf_token == "no_token":
        logger.info("Diarization skipped (no token provided)")
        return []
    
    logger.info(f"Starting diarization with model {config.diarization_model}")
    
    # Show source information if available
    if "source_info" in audio_data and audio_data["source_info"].get("path"):
        logger.info(f"Source: {format_path(audio_data['source_info']['path'])}")
    
    def execute_diarization() -> Tuple[List[DiarizedChunk], List[Dict[str, Any]]]:
        # 1. Load pipeline
        diarization_pipeline = load_diarization_pipeline(
            config, dynamic_imports={"torch": dynamic_imports["torch"], 
                                    "pyannote.audio": dynamic_imports["audio"]}
        )
        
        # 2. Run diarization
        diarization = diarization_pipeline(
            {"waveform": audio_data["torch_tensor"], "sample_rate": 16000},
            num_speakers=config.num_speakers,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
        )
        
        # 3. Process segments
        new_segments: List[Dict[str, Any]] = process_diarization_segments(diarization)
        
        # 4. Align with transcript
        transcript_chunks: List[TranscriptChunk] = transcript["chunks"].copy()
        segmented_preds: List[DiarizedChunk] = align_segments_with_transcript(
            new_segments, transcript_chunks, 
            dynamic_imports={"numpy": dynamic_imports["numpy"]}
        )
        
        return segmented_preds, new_segments
    
    segmented_preds, new_segments = with_progress_bar("Segmenting speakers...", execute_diarization)
    
    num_speakers: int = len(set(segment["speaker"] for segment in new_segments)) if new_segments else 0
    logger.info(f"Diarization complete: {num_speakers} speakers detected")
    
    return segmented_preds