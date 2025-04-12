from typing import Dict, Any, cast
from helpers import log_time, with_imports, with_progress_bar, logger, format_path
from data_types import TranscriptionConfig, AudioData, TranscriptOutput


def _build_whisper_pipeline(
    config: TranscriptionConfig, torch: Any, transformers: Any
) -> Any:
    """
    Build the Whisper-based automatic speech recognition pipeline.
    
    This function configures model parameters such as the attention implementation
    based on the provided configuration and sets up the device accordingly.

    Args:
        config (TranscriptionConfig): Configuration for transcription.
        torch (Any): The PyTorch module.
        transformers (Any): The HuggingFace Transformers module.

    Returns:
        Any: An ASR pipeline from HuggingFace transformers.
    """
    model_kwargs: Dict[str, Any] = {}
    if config.attn_type == "sdpa":
        model_kwargs["attn_implementation"] = "sdpa"
    elif config.attn_type == "eager":
        model_kwargs["attn_implementation"] = "eager"
    elif config.attn_type == "flash":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Set device: if device_id is not "mps" or "cpu", use CUDA with the specified device number
    device = "cuda:" + config.device_id if config.device_id not in ["mps", "cpu"] else config.device_id

    return transformers.pipeline(
        "automatic-speech-recognition",
        model=config.model_name,
        torch_dtype=torch.float16,
        device=device,
        model_kwargs=model_kwargs,
    )


def _run_transcription(
    config: TranscriptionConfig, pipe: Any, audio_data: AudioData
) -> Any:
    """
    Run the transcription process using the provided ASR pipeline.
    
    The function prepares the input by directly using the original audio numpy array.
    It constructs the generate_kwargs including the language parameter, passing it as-is.

    Args:
        config (TranscriptionConfig): Transcription configuration.
        pipe (Any): The ASR pipeline.
        audio_data (AudioData): Processed audio data.

    Returns:
        Any: The output from the ASR pipeline, typically containing transcribed text and timestamps.
    """
    # Directly use the original audio numpy array without referencing a non-existent attribute.
    audio_np = audio_data["numpy_array"]

    audio_input = {
        "raw": audio_np.copy(),
        "sampling_rate": audio_data["sampling_rate"],
    }
    # Build additional arguments; language is passed as provided (even if None)
    generate_kwargs = {"task": config.task, "language": config.language}

    return pipe(
        audio_input,
        chunk_length_s=config.chunk_length,
        batch_size=config.batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )


@log_time
@with_imports("torch", "transformers")
def transcribe_audio(
    config: TranscriptionConfig, 
    audio_data: AudioData,
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> TranscriptOutput:
    """
    Transcribe audio using a Whisper-based model.
    
    This function logs the start of transcription, builds the ASR pipeline,
    runs the transcription while showing a progress bar, and logs the number of segments generated.

    Args:
        config (TranscriptionConfig): Transcription configuration.
        audio_data (AudioData): Processed audio data.
        dynamic_imports (Dict[str, Any]): Dynamically imported modules (must include "torch" and "transformers").

    Returns:
        TranscriptOutput: The transcription output containing the full text and time-marked segments.
    """
    torch = dynamic_imports["torch"]
    transformers = dynamic_imports["transformers"]

    logger.info(f"Starting transcription with model {config.model_name}")
    if "source_info" in audio_data and audio_data["source_info"].get("path"):
        logger.info(f"Source: {format_path(audio_data['source_info']['path'])}")

    # Build the Whisper pipeline with the configured model and device
    pipe = _build_whisper_pipeline(config, torch, transformers)
    
    # Run transcription with a progress bar for improved user feedback
    outputs = with_progress_bar("Transcribing...", lambda: _run_transcription(config, pipe, audio_data))
    logger.info(f"Transcription completed: {len(outputs.get('chunks', []))} segments generated")

    return cast(TranscriptOutput, outputs)
