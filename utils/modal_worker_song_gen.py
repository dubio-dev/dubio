# Generate song using jamify on modal

# todo

import modal
import subprocess

### MODAL APP NAME ###
modal_app_name = "song-gen-dev"
### MODAL APP NAME ###

app = modal.App(modal_app_name)

image = (
    modal.Image.debian_slim()
    .apt_install(["git", "ffmpeg"])
    .pip_install_from_requirements("jamify/requirements.txt")
    .pip_install("git+https://github.com/xhhhhang/DeepPhonemizer@dcfafbf2")
    .add_local_python_source("jamify", copy=True)
    .add_local_dir("jamify/configs", remote_path="/root/jamify/configs", copy=True)
    .add_local_dir("jamify/inputs", remote_path="/root/jamify/inputs", copy=True)
    .add_local_dir("jamify/public", remote_path="/root/jamify/public", copy=True)
    .env({"PYTHONPATH": "/root/jamify/src"})  # ?? maybe unnecessary ??
)

volumes = {
    "/mnt/models": modal.Volume.from_name("models"),
    "/mnt/outputs": modal.Volume.from_name("outputs", create_if_missing=True),
}


@app.function(image=image, volumes=volumes, gpu="H100", timeout=600)
def main():
    # Sample output in lyrics-generation/output_lyrics.json
    subprocess.run([
        "python", "-m", "jam.infer",
        "evaluation.checkpoint_path=/mnt/models/jam-0.5/jam-0_5.safetensors",
        "evaluation.output_dir=/mnt/outputs/hello-world-0",
        "config=/root/jamify/configs/jam_infer.yaml"
    ], cwd="/root/jamify", shell=False)


@app.local_entrypoint()
def local_main():
    main.remote()
