"""
Functions for audio transcription.
"""
from typing import Dict, Any, cast

from helpers import log_time, with_imports, with_progress_bar, logger, format_path
from data_types import TranscriptionConfig, AudioData, TranscriptOutput

def _build_whisper_pipeline(config: TranscriptionConfig, torch: Any, transformers: Any) -> Any:
    model_kwargs = {}
    if config.attn_type == "sdpa":
        model_kwargs["attn_implementation"] = "sdpa"
    elif config.attn_type == "eager":
        model_kwargs["attn_implementation"] = "eager"
    elif config.attn_type == "flash":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    device = (
        "cuda:" + config.device_id 
        if config.device_id not in ["mps", "cpu"] 
        else config.device_id
    )

    return transformers.pipeline(
        "automatic-speech-recognition",
        model=config.model_name,
        torch_dtype=torch.float16,
        device=device,
        model_kwargs=model_kwargs,
    )

def _run_transcription(config: TranscriptionConfig, pipe: Any, audio_data: AudioData) -> Any:
    audio_input = {
        "raw": audio_data["numpy_array"].copy(),
        "sampling_rate": audio_data["sampling_rate"]
    }

    return pipe(
        audio_input,
        chunk_length_s=config.chunk_length,
        batch_size=config.batch_size,
        generate_kwargs={
            "task": config.task,
            "language": config.language
        },
        return_timestamps=True,
    )


def _detect_language_if_needed(config: TranscriptionConfig, pipe: Any, audio_data: AudioData) -> None:
    if config.language is not None:
        return

    logger.info("Language not specified. Detecting automatically...")
    snippet = {
        "raw": audio_data["numpy_array"][:int(30 * audio_data["sampling_rate"])],
        "sampling_rate": audio_data["sampling_rate"]
    }

    result = pipe(snippet, return_language=True, return_timestamps=False)
    language_code = result.get("language")

    if not language_code:
        raise ValueError("Could not detect the audio language.")

    config.language = language_code
    logger.info(f"Detected language: '{language_code}'")


@log_time
@with_imports("torch", "transformers")
def transcribe_audio(
    config: TranscriptionConfig, 
    audio_data: AudioData,
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> TranscriptOutput:
    """
    Transcribe the audio using the Whisper model.
    If language is not specified, it will be automatically detected.
    """
    torch = dynamic_imports["torch"]
    transformers = dynamic_imports["transformers"]

    logger.info(f"Starting transcription with model {config.model_name}")
    if "source_info" in audio_data and audio_data["source_info"].get("path"):
        logger.info(f"Source: {format_path(audio_data['source_info']['path'])}")

    # 1. Prepare pipeline
    pipe = _build_whisper_pipeline(config, torch, transformers)

    # 2. Detect language if needed
    _detect_language_if_needed(config, pipe, audio_data)

    # 3. Run full transcription
    outputs = with_progress_bar("Transcribing...", lambda: _run_transcription(config, pipe, audio_data))
    logger.info(f"Transcription complete: {len(outputs.get('chunks', []))} fragments")

    return cast(TranscriptOutput, outputs)