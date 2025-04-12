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

    device = "cuda:" + config.device_id if config.device_id not in ["mps", "cpu"] else config.device_id

    return transformers.pipeline(
        "automatic-speech-recognition",
        model=config.model_name,
        torch_dtype=torch.float16,
        device=device,
        model_kwargs=model_kwargs,
    )

def _run_transcription(config: TranscriptionConfig, pipe: Any, audio_data: AudioData) -> Any:
    # Se elimina la referencia a config.vad_transcription ya que dicho atributo no existe;
    # se utiliza directamente el array de audio original.
    audio_np = audio_data["numpy_array"]

    audio_input = {
        "raw": audio_np.copy(),
        "sampling_rate": audio_data["sampling_rate"]
    }
    # Se construye generate_kwargs incluyendo el parámetro "language" tal cual,
    # lo que permite que si es None se pase sin modificaciones.
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
    torch = dynamic_imports["torch"]
    transformers = dynamic_imports["transformers"]

    logger.info(f"Inicio de transcripción con el modelo {config.model_name}")
    if "source_info" in audio_data and audio_data["source_info"].get("path"):
        logger.info(f"Fuente: {format_path(audio_data['source_info']['path'])}")

    pipe = _build_whisper_pipeline(config, torch, transformers)
    # Se ha eliminado la detección automática del idioma.
    # Si config.language es None, se pasa directamente al pipeline.
    outputs = with_progress_bar("Transcribiendo...", lambda: _run_transcription(config, pipe, audio_data))
    logger.info(f"Transcripción completada: {len(outputs.get('chunks', []))} fragmentos")

    return cast(TranscriptOutput, outputs)
