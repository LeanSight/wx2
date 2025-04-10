"""
Module for converting transcriptions to different formats.
Optimized version to eliminate code duplication.
"""
import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

from helpers import logger, format_path

class OutputFormat(Enum):
    """Supported output formats for transcriptions."""
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    TXT = "txt"

def output_format_type(value: str) -> OutputFormat:
    """Convert string to OutputFormat enum."""
    try:
        return OutputFormat(value.lower())
    except ValueError:
        raise ValueError(f"Unsupported format: {value}")

def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    """
    Format seconds to timestamp format.
    
    Args:
        seconds: Time in seconds
        format_type: Format type ("srt" or "vtt")
        
    Returns:
        str: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    if format_type == "vtt":
        return f"{hours:02}:{minutes:02}:{secs:02}.{millisecs:03}"
    else:  # srt default
        return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"

def get_speaker_display(segment: Dict[str, Any], speaker_names: Optional[Dict[str, str]]) -> Optional[str]:
    """
    Gets the display name of the speaker.
    
    Args:
        segment: Transcription segment
        speaker_names: Mapping of speaker ID to custom name
        
    Returns:
        Optional[str]: Speaker display name or None if no information
    """
    if 'speaker' not in segment:
        return None
        
    speaker_id = segment['speaker']
    if speaker_names:
        return speaker_names.get(speaker_id, speaker_id)
    return speaker_id

def get_segments_from_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts segments from transcription data.
    
    Args:
        data: Transcription data
        
    Returns:
        List[Dict[str, Any]]: List of segments
    """
    # Use 'speakers' if available, otherwise 'chunks'
    speakers = data.get('speakers', [])
    return speakers if speakers else data.get('chunks', [])

def format_subtitle_content(
    segments: List[Dict[str, Any]],
    speaker_names: Optional[Dict[str, str]],
    format_type: str
) -> str:
    """
    Formats a list of segments in subtitle format.
    
    Args:
        segments: List of transcription segments
        speaker_names: Mapping of speaker ID to custom name
        format_type: Format type ("srt" or "vtt")
        
    Returns:
        str: Formatted content
    """
    result = []
    # Add VTT header if needed
    if format_type == "vtt":
        result.append("WEBVTT\n")
    
    index = 1
    
    for segment in segments:
        # Get timestamps from segment
        start_time, end_time = segment['timestamp']
        if start_time is None or end_time is None:
            continue
            
        # Format text with speaker name if available
        text = segment['text'].strip()
        speaker_display = get_speaker_display(segment, speaker_names)
        if speaker_display:
            text = f"{speaker_display}: {text}"
        
        # Format timestamps
        start_timestamp = format_timestamp(start_time, format_type)
        end_timestamp = format_timestamp(end_time, format_type)
        
        # Add entry according to format
        if format_type == "vtt":
            result.append(f"\n{start_timestamp} --> {end_timestamp}\n{text}")
        else:  # srt
            result.append(f"{index}\n{start_timestamp} --> {end_timestamp}\n{text}\n")
            index += 1
    
    return "\n".join(result)

def convert_to_srt(data: Dict[str, Any], speaker_names: Optional[Dict[str, str]] = None) -> str:
    """Convert transcription data to SRT format."""
    segments = get_segments_from_data(data)
    return format_subtitle_content(segments, speaker_names, "srt")

def convert_to_vtt(data: Dict[str, Any], speaker_names: Optional[Dict[str, str]] = None) -> str:
    """Convert transcription data to VTT format."""
    segments = get_segments_from_data(data)
    return format_subtitle_content(segments, speaker_names, "vtt")

def convert_to_txt(data: Dict[str, Any], speaker_names: Optional[Dict[str, str]] = None) -> str:
    """Convert transcription data to plain text format."""
    result = []
    
    # Determine which list to use - with speakers if available, otherwise use full text
    if 'speakers' in data and data['speakers']:
        segments = data['speakers']
        
        current_speaker = None
        current_text = []
        
        for segment in segments:
            speaker_id = segment['speaker']
            # If speaker changes, add accumulated text and reset
            if current_speaker is not None and current_speaker != speaker_id:
                speaker_display = get_speaker_display({'speaker': current_speaker}, speaker_names)
                result.append(f"{speaker_display}: {' '.join(current_text)}")
                current_text = []
            
            current_speaker = speaker_id
            current_text.append(segment['text'].strip())
        
        # Add the last segment
        if current_speaker and current_text:
            speaker_display = get_speaker_display({'speaker': current_speaker}, speaker_names)
            result.append(f"{speaker_display}: {' '.join(current_text)}")
    
    # If no speaker information or it's empty, use the full text
    else:
        result = [data['text']]
    
    return "\n\n".join(result)

def create_speaker_map(speaker_names_str: Optional[str] = None) -> Dict[str, str]:
    """Create a mapping of speaker IDs to custom names."""
    if not speaker_names_str:
        return {}
        
    names = [name.strip() for name in speaker_names_str.split(',')]
    return {f"SPEAKER_{i:02d}": name for i, name in enumerate(names)}

def convert_output(
    input_path: Union[str, Path], 
    output_format: OutputFormat, 
    output_dir: Optional[Union[str, Path]] = None,
    speaker_names: Optional[str] = None
) -> Path:
    """
    Convert a JSON transcription file to the specified format.
    
    Args:
        input_path: Path to the input JSON file
        output_format: Desired output format
        output_dir: Directory to save the output file (optional)
        speaker_names: Comma-separated list of speaker names (optional)
    
    Returns:
        Path: Path to the generated output file
    """
    # Handle paths
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        
    # Create speaker map if provided
    speaker_map = create_speaker_map(speaker_names)
    if speaker_map:
        logger.info(f"Using speaker mapping: {speaker_map}")
    
    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine base name and extension of output file
    stem = input_path.stem
    output_ext = f".{output_format.value}"
    output_path = output_dir / f"{stem}{output_ext}"
    
    # If the format is JSON, simply copy the file
    if output_format == OutputFormat.JSON:
        if str(output_path) != str(input_path):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path
    
    # Convert according to requested format
    converter_map = {
        OutputFormat.SRT: convert_to_srt,
        OutputFormat.VTT: convert_to_vtt,
        OutputFormat.TXT: convert_to_txt
    }
    
    converter = converter_map.get(output_format)
    if not converter:
        raise ValueError(f"No converter implemented for format {output_format}")
    
    # Perform the conversion
    output_content = converter(data, speaker_map)
    
    # Save the result
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    logger.info(f"File converted and saved to: {format_path(str(output_path))}")
    return output_path