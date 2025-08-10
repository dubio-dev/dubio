import modal
import os

tag = "nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04"
image = (
    modal.Image.from_registry(tag, add_python="3.12")
    .pip_install("torch")
    .pip_install("whisperx")
    .apt_install("ffmpeg")
)

app = modal.App("whisperx-worker-dev", image=image)

@app.function(
    timeout=600,
    gpu="A100",
    volumes={
        "/mnt/models": modal.Volume.from_name("models"),
        "/mnt/outputs": modal.Volume.from_name("outputs", create_if_missing=True),
    }
)
def worker():
    print("hello world")
    os.system("ls /usr/local/cuda/lib64/libcudnn*")
    os.system("ldconfig -p | grep cudnn")
    return

    import whisperx

    device = "cuda"
    audio_file = "/mnt/outputs/golden.mp3"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    # save model to local path (optional)
    model_dir = "/mnt/models/whisperx"
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # after alignment
    return result

    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a


@app.local_entrypoint()
def main():
    worker.remote()


if __name__ == "__main__":
    main()
