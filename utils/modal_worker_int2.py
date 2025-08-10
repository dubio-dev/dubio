# todo

import modal
import os

### MODAL APP NAME ###

modal_app_name = "jamify-worker-dev"

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
    "/mnt/outputs": modal.Volume.from_name("my-output-volume", create_if_missing=True),
}


@app.function(image=image, volumes=volumes, gpu="H100", timeout=600)
def main():
    print("First part: yt scrape")
    os.system("python /root/lyrics-aligner/yt_to_resonite_visemes.py'")

    print("Second part: robust_lyrics_gen.py")
    # TODO: fix so that it directly takes from out/ (involves shortening input video tho)
    os.system(
        "python /root/lyricsgeneration/robust_lyrics_gen.py /root/lyricsgeneration/visemes_resonite_min_short.json /root/lyricsgeneration/output_lyrics.json --topic 'basic math'"
    )

    print("\n=== Extracting Generated Files ===")
    os.system("cp -r /root/jamify/outputs /mnt/outputs/")
    volumes["/mnt/outputs"].commit()
    print("✅ Files copied to my-output-volume")


@app.local_entrypoint()
def local_main():
    main.remote()
