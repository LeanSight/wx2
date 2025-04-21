#!/usr/bin/env python3
"""
Audio transcription and diarization system with functional approach.
Uses specific static typing and decorators for dynamic imports.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, List, Union, Optional, Tuple, cast

# Environment configuration
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=FutureWarning)

# Add these new filters
warnings.filterwarnings("ignore", module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", module="pyannote.audio.models.blocks.pooling")

# Filter SpeechBrain logs (they use the logging module)
logging.getLogger("speechbrain.utils.quirks").setLevel(logging.WARNING)

# Filter Torio ffmpeg import warnings
logging.getLogger('torio._extension.utils').setLevel(logging.ERROR)

# Filter checkpoint.py debug 
logging.getLogger("checkpoint").setLevel(logging.DEBUG)

# Local imports
from data_types import (
    TranscriptionConfig, TaskType, FinalResult, 
    TranscriptOutput, DiarizedChunk, ProcessingMetadata
)
from helpers import log_time, logger, format_path, set_log_level
from audio import process_audio
from transcription import transcribe_audio
from diarization import diarize_audio
from formatters import OutputFormat, convert_output, output_format_type

@log_time
def parse_arguments() -> Tuple[TranscriptionConfig, str]:
    """
    Parse command line arguments and return typed configuration.
    
    Returns:
        Tuple[TranscriptionConfig, str]: Configuración inmutable con valores validados y nivel de logging
    """
    parser = argparse.ArgumentParser(
        description="Audio transcription and diarization system with functional approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Basic transcription of a local file
  transcribe.py file.mp3

  # Transcription with SRT format
  transcribe.py video.mp4 -f srt

  # Transcription with diarization and speaker names
  transcribe.py meeting.wav --diarize --token=hf_xxx --speaker-names="John,Mary,Peter"

  # Transcription in English with optimized attention
  transcribe.py interview.mp3 -l en --attn-type flash
  
  # Transcription with debug logs enabled
  transcribe.py audio.mp3 --debug debug
        """
    )
    
    # Create argument groups for better organization
    basic_group = parser.add_argument_group('basic options')
    transc_group = parser.add_argument_group('transcription options')
    perf_group = parser.add_argument_group('performance options')
    diar_group = parser.add_argument_group('diarization options')
    debug_group = parser.add_argument_group('debug options')
    
    # Positional argument for input file
    parser.add_argument(
        "file_name", 
        type=str, 
        help="Path or URL to the audio/video file to process"
    )
    
    # Basic options
    basic_group.add_argument(
        "-o", "--output",
        dest="transcript_path",
        type=str,
        default=None,  # Changed from "output.json" to None to handle dynamically
        help="Path to save the result (default: [input_name]-transcribe.[format])"
    )
    
    basic_group.add_argument(
        "-f", "--format",
        dest="output_format",
        type=str,
        choices=["json", "srt", "vtt", "txt"],
        default="json",
        help="Output file format (default: json)"
    )
    
    # Transcription options
    transc_group.add_argument(
        "-m", "--model",
        dest="model_name",
        type=str,
        default="openai/whisper-large-v3",
        help="Transcription model to use (default: openai/whisper-large-v3)"
    )
    
    transc_group.add_argument(
        "-l", "--lang",
        dest="language",
        type=str,
        default=None,
        help="Audio language (ISO code, e.g. 'es', 'en', 'fr') (default: es)"
    )
    
    transc_group.add_argument(
        "-t", "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform: transcribe or translate to English (default: transcribe)"
    )
    
    transc_group.add_argument(
        "--chunk-length",
        type=int,
        default=10,
        help="Duration in seconds of each audio chunk (default: 30)"
    )
    
    # Performance options
    perf_group.add_argument(
        "-d", "--device",
        dest="device_id",
        type=str,
        default="0",
        help="GPU device ID (0, 1, etc.) or 'cpu' or 'mps' (default: 0)"
    )
    
    perf_group.add_argument(
        "-b", "--batch-size",
        type=int,
        default=8,
        help="Batch size for GPU processing (default: 8)"
    )
    
    perf_group.add_argument(
        "--attn-type",
        choices=["sdpa", "eager", "flash"],
        default="sdpa",
        help="Attention implementation type (default: sdpa)"
    )
    
    # Debug options
    debug_group.add_argument(
        "-dbg", "--debug",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set logging level (default: info)"
    )
    
    # Diarization options
    diar_group.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker identification (requires --token)"
    )
    
    diar_group.add_argument(
        "--dmodel",
        dest="diarization_model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="Diarization model to use (default: pyannote/speaker-diarization-3.1)"
    )
    
    diar_group.add_argument(
        "--token",
        dest="hf_token",
        type=str,
        default="no_token",
        help="HuggingFace token for diarization models"
    )
    
    # Mutually exclusive group for speaker number options
    speakers_group = diar_group.add_mutually_exclusive_group()
    
    speakers_group.add_argument(
        "--num-speakers",
        type=int,
        help="Exact number of speakers in the audio"
    )
    
    speakers_group.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers to detect"
    )
    
    diar_group.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers to detect"
    )
    
    diar_group.add_argument(
        "--speaker-names",
        type=str,
        help="Comma-separated list of names to replace speaker labels (e.g.: \"John,Mary,Peter\")"
    )
    
    args = parser.parse_args()
    
    # Additional validations
    if args.max_speakers is not None and args.min_speakers is None:
        parser.error("--max-speakers requires --min-speakers")
    
    if args.diarize and args.hf_token == "no_token":
        parser.error("The --diarize option requires a HuggingFace token (--token)")
    
    # Extract logging level
    log_level = args.debug
    
    # NEW: Calculate default output path if not specified
    if args.transcript_path is None:
        # Get the full path of the input file
        input_path = Path(args.file_name)
        
        # If it's a URL, use only the filename
        if args.file_name.startswith(("http://", "https://")):
            import urllib.parse
            file_name = urllib.parse.urlparse(args.file_name).path.split("/")[-1]
            input_path = Path(file_name)
        
        # Get the directory and base name of the input file
        input_dir = input_path.parent
        input_stem = input_path.stem
        
        # Generate the output filename
        output_name = f"{input_stem}-transcribe.{args.output_format}"
        
        # Combine with the directory for the full path
        args.transcript_path = str(input_dir / output_name)
        logger.info(f"Default output path: {format_path(args.transcript_path)}")
    
    # Adjust the output file extension according to the format
    if args.output_format != "json":
        output_path = Path(args.transcript_path)
        # If the extension doesn't match the format, change it
        if output_path.suffix.lower() != f".{args.output_format}":
            stem = output_path.stem
            args.transcript_path = str(output_path.parent / f"{stem}.{args.output_format}")
    
    # Create typed and validated configuration
    config = TranscriptionConfig(
        file_name=args.file_name,
        device_id=args.device_id,
        transcript_path=args.transcript_path,
        model_name=args.model_name,
        task=cast(TaskType, args.task),
        language=args.language,
        batch_size=args.batch_size,
        output_format=args.output_format,
        chunk_length=args.chunk_length,
        attn_type=args.attn_type,
        hf_token=args.hf_token,
        diarization_model=args.diarization_model,
        diarize=args.diarize,
        speaker_names=args.speaker_names,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    
    return config, log_level

@log_time
def build_and_save_transcript(
    config: TranscriptionConfig,
    transcript: TranscriptOutput,
    audio_data: Any,
    *,
    speakers_transcript: List[DiarizedChunk] = [],
    is_intermediate: bool = False
) -> Union[Path, FinalResult]:
    """
    Construye y guarda un resultado de transcripción, ya sea intermedio o final.
    
    Args:
        config: Configuración de transcripción
        transcript: Resultados de transcripción
        audio_data: Datos de audio procesados
        speakers_transcript: Transcripción con info de hablantes (solo para resultado final)
        is_intermediate: Indica si es un resultado intermedio
        
    Returns:
        Union[Path, FinalResult]: Ruta al archivo (si es intermedio) o resultado completo (si es final)
    """
    # Determinar el tipo de resultado para los mensajes
    result_type = "intermediate" if is_intermediate else "final"
    logger.info(f"Building {result_type} result")
    
    # Crear metadatos de procesamiento
    processing_meta: ProcessingMetadata = {
        "transcription_model": config.model_name,
        "language": config.language,
        "device": "mps" if config.device_id == "mps" else "cpu" if config.device_id == "cpu" else f"cuda:{config.device_id}",
        "timestamp": datetime.now().isoformat(),
        "diarization": config.diarize and not is_intermediate,
        "diarization_model": config.diarization_model if config.diarize and not is_intermediate else None
    }
    
    # Incluir metadatos completos
    metadata = {
        "source": audio_data.get("source_info", {}),
        "processing": processing_meta
    }
    
    # Construir el resultado
    result = {
        "chunks": transcript["chunks"],
        "text": transcript["text"],
        "metadata": metadata
    }
    
    # Añadir información de hablantes si está disponible y no es intermedio
    if speakers_transcript and not is_intermediate:
        result["speakers"] = speakers_transcript
    
    # Determinar ruta de salida
    base_path = Path(str(config.transcript_path))
    if is_intermediate:
        # Modificar nombre para resultado intermedio
        json_output_path = base_path.parent / f"{base_path.stem}-intermediate.json"
    else:
        # Usar nombre estándar para resultado final
        json_output_path = Path(str(config.transcript_path).replace(f".{config.output_format}", ".json")) \
            if config.output_format != "json" else Path(config.transcript_path)
    
    logger.info(f"Saving {result_type} JSON result to: {format_path(str(json_output_path))}")
    
    # Mostrar información de origen si está disponible
    if "source" in metadata and metadata["source"].get("path"):
        logger.info("Source information included in metadata:")
        logger.info(f"- File: {format_path(metadata['source']['path'])}")
        
        format_info = []
        if metadata["source"].get("is_video"):
            format_info.append("video")
        elif metadata["source"].get("type") == "url":
            format_info.append("remote")
        else:
            format_info.append("audio")
            
        if metadata["source"].get("format"):
            format_info.append(f"format: {metadata['source']['format']}")
            
        logger.info(f"- Type: {', '.join(format_info)}")
        
        if metadata["source"].get("duration_seconds"):
            duration = metadata["source"]["duration_seconds"]
            duration_str = f"{int(duration//60)}m {int(duration%60)}s"
            logger.info(f"- Duration: {duration_str}")
            
        if metadata["source"].get("sampling_rate"):
            logger.info(f"- Samples: {metadata['source'].get('numpy_array', {}).shape[0] if 'numpy_array' in metadata['source'] else 'N/A'} at {metadata['source']['sampling_rate']}Hz")
    
    # Guardar el JSON
    with open(json_output_path, "w", encoding="utf8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    
    # Si el formato solicitado no es JSON, convertir
    if config.output_format != "json":
        try:
            output_format = OutputFormat(config.output_format)
            # Usar nombres de hablantes solo para resultado final
            speaker_names = config.speaker_names if not is_intermediate else None
            
            converted_path = convert_output(
                input_path=json_output_path,
                output_format=output_format,
                output_dir=json_output_path.parent,
                speaker_names=speaker_names
            )
            
            # No eliminar el archivo JSON
            if is_intermediate:
                logger.info(f"{result_type.capitalize()} files saved:")
                logger.info(f"- JSON: {format_path(str(json_output_path))}")
                logger.info(f"- {config.output_format.upper()}: {format_path(str(converted_path))}")
            else:
                logger.info(f"[green]Voila!✨[/] Both files saved:")
                logger.info(f"- JSON: {format_path(str(json_output_path))}")
                logger.info(f"- {config.output_format.upper()}:  {format_path(str(converted_path))}")
        except Exception as e:
            logger.error(f"[red]Error converting format[/]: {str(e)}")
            logger.info(f"Keeping result in JSON format: {format_path(str(json_output_path))}")
    else:
        if is_intermediate:
            logger.info(f"{result_type.capitalize()} file saved to: {format_path(str(json_output_path))}")
        else:
            logger.info(f"[green]Voila!✨[/] File saved to: {format_path(str(json_output_path))}")
    
    # Para resultado final, devolver el objeto completo como FinalResult
    if not is_intermediate:
        return cast(FinalResult, result)
    
    # Para intermedio, solo devolver la ruta
    return json_output_path

@log_time
def main() -> FinalResult:
    """
    Main function that coordinates the entire process.
    
    Returns:
        FinalResult: Final result of the process
    """
    try:
        # 1. Process arguments
        config, log_level = parse_arguments()
        
        # 1.5 Configure logging level
        set_log_level(log_level)
        
        # 2. Process audio
        audio_data = process_audio(config.file_name)
        
        # 3. Transcribe audio
        transcript = transcribe_audio(config, audio_data)
        
        # 3.5. Guardar transcripción intermedia
        build_and_save_transcript(
            config=config,
            transcript=transcript,
            audio_data=audio_data,
            is_intermediate=True
        )
        
        # 4. Diarize audio (optional)
        speakers_transcript = []
        if config.diarize:
            speakers_transcript = diarize_audio(config, audio_data, transcript)
        
        # 5. Build and save final result
        result = build_and_save_transcript(
            config=config,
            transcript=transcript,
            audio_data=audio_data,
            speakers_transcript=speakers_transcript,
            is_intermediate=False
        )
        
        return result
    except Exception as e:
        logger.error(f"[red]Error during execution:[/] {str(e)}")
        logger.debug("Error details:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()