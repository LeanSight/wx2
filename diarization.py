"""
Funciones para diarización (identificación de hablantes) utilizando la lógica de v1.
Mantiene la API de v2 para compatibilidad.
"""
import sys
from typing import Dict, Any, List, Tuple, cast, Optional

import requests
import torch
import numpy as np
from torch.nn import functional as F
from transformers.pipelines.audio_utils import ffmpeg_read

from helpers import log_time, with_imports, with_progress_bar, logger, format_path
from data_types import (
    TranscriptionConfig, AudioData, TranscriptOutput, 
    TranscriptChunk, DiarizedChunk
)

def preprocess_inputs(input_src: str, audio_data: AudioData) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Preprocesa los datos de audio utilizando la lógica de v1.
    
    Args:
        input_src: Ruta al archivo de audio o URL
        audio_data: Datos de audio procesados por v2
    
    Returns:
        Tuple[np.ndarray, torch.Tensor]: Datos de audio procesados en formato NumPy y Torch
    """
    # Usar directamente los datos ya procesados por audio.py de v2
    inputs = audio_data["numpy_array"]
    
    # Diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = audio_data["torch_tensor"]
    
    return inputs, diarizer_inputs

def diarize_audio_v1(
    diarizer_inputs: torch.Tensor, 
    diarization_pipeline: Any, 
    num_speakers: Optional[int], 
    min_speakers: Optional[int], 
    max_speakers: Optional[int]
) -> List[Dict[str, Any]]:
    """
    Realiza la diarización utilizando la lógica exacta de v1.
    
    Args:
        diarizer_inputs: Tensor de audio para diarización
        diarization_pipeline: Pipeline de diarización
        num_speakers: Número exacto de hablantes (opcional)
        min_speakers: Número mínimo de hablantes (opcional)
        max_speakers: Número máximo de hablantes (opcional)
    
    Returns:
        List[Dict[str, Any]]: Segmentos de diarización
    """
    diarization = diarization_pipeline(
        {"waveform": diarizer_inputs, "sample_rate": 16000},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append({
            "segment": {"start": segment.start, "end": segment.end},
            "track": track,
            "label": label,
        })

    # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
    new_segments = []
    
    if not segments:
        return new_segments
    
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append({
                "segment": {
                    "start": prev_segment["segment"]["start"],
                    "end": cur_segment["segment"]["start"],
                },
                "speaker": prev_segment["label"],
            })
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append({
        "segment": {
            "start": prev_segment["segment"]["start"],
            "end": cur_segment["segment"]["end"],
        },
        "speaker": prev_segment["label"],
    })

    return new_segments

def post_process_segments_and_transcripts(
    new_segments: List[Dict[str, Any]], 
    transcript_chunks: List[TranscriptChunk]
) -> List[DiarizedChunk]:
    """
    Alinea los segmentos de diarización con la transcripción usando la lógica de v1.
    
    Args:
        new_segments: Segmentos de diarización
        transcript_chunks: Fragmentos de transcripción
    
    Returns:
        List[DiarizedChunk]: Transcripción con información de hablantes
    """
    if not new_segments or not transcript_chunks:
        return []
    
    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array([
        chunk["timestamp"][-1] if chunk["timestamp"][-1] is not None 
        else sys.float_info.max for chunk in transcript_chunks
    ])
    
    segmented_preds = []

    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        # Añadir chunks con información de hablante (equivalente a group_by_speaker=False en v1)
        for i in range(upto_idx + 1):
            segmented_preds.append({
                "text": transcript_chunks[i]["text"],
                "timestamp": transcript_chunks[i]["timestamp"],
                "speaker": segment["speaker"]
            })

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript_chunks = transcript_chunks[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

        if len(end_timestamps) == 0:
            break

    return segmented_preds

@with_imports("torch", "pyannote.audio")
def load_diarization_pipeline(
    config: TranscriptionConfig,
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> Any:
    """
    Carga el pipeline de diarización.
    
    Args:
        config: Configuración de transcripción/diarización
        dynamic_imports: Módulos importados dinámicamente
        
    Returns:
        Pipeline: Pipeline de diarización cargado
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
    Realiza la diarización para identificar diferentes hablantes.
    Implementa la API de v2 pero utiliza la lógica de v1.
    
    Args:
        config: Configuración de transcripción/diarización
        audio_data: Audio procesado para diarización
        transcript: Resultados de transcripción
        dynamic_imports: Módulos importados dinámicamente
        
    Returns:
        List[DiarizedChunk]: Transcripción con información de hablantes
    """
    # Si no hay token, omitir diarización
    if config.hf_token == "no_token":
        logger.info("Diarización omitida (no se proporcionó token)")
        return []
    
    logger.info(f"Iniciando diarización con el modelo {config.diarization_model}")
    
    # Mostrar información de origen si está disponible
    if "source_info" in audio_data and audio_data["source_info"].get("path"):
        logger.info(f"Origen: {format_path(audio_data['source_info']['path'])}")
    
    def execute_diarization() -> List[DiarizedChunk]:
        # 1. Cargar pipeline
        diarization_pipeline = load_diarization_pipeline(
            config, dynamic_imports={"torch": dynamic_imports["torch"], 
                                    "pyannote.audio": dynamic_imports["audio"]}
        )
        
        # 2. Preprocesar los datos de audio (utilizando audio ya procesado por v2)
        _, diarizer_inputs = preprocess_inputs(config.file_name, audio_data)
        
        # 3. Ejecutar diarización con la lógica de v1
        segments = diarize_audio_v1(
            diarizer_inputs, 
            diarization_pipeline, 
            config.num_speakers, 
            config.min_speakers, 
            config.max_speakers
        )
        
        # 4. Alinear con la transcripción usando la lógica de v1
        transcript_chunks: List[TranscriptChunk] = transcript["chunks"].copy()
        segmented_preds = post_process_segments_and_transcripts(segments, transcript_chunks)
        
        return segmented_preds
    
    segmented_preds = with_progress_bar("Segmentando hablantes...", execute_diarization)
    
    # Count unique speakers
    speakers = set()
    for segment in segmented_preds:
        if "speaker" in segment:
            speakers.add(segment["speaker"])
    
    num_speakers = len(speakers)
    logger.info(f"Diarización completada: {num_speakers} hablantes detectados")
    
    return segmented_preds
