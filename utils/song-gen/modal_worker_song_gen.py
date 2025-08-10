# Generate song using jamify on modal
# Run `uv run modal run song-gen/modal_worker_song_gen.py` from `utils`

import modal
import subprocess
import datetime as dt
import json
import os

### MODAL APP NAME ###
modal_app_name = "song-gen-dev"
### MODAL APP NAME ###

app = modal.App(modal_app_name)

image = (
    modal.Image.debian_slim()
    .apt_install(["git", "ffmpeg"])
    .pip_install_from_requirements("jamify/requirements.txt")
    .pip_install("git+https://github.com/xhhhhang/DeepPhonemizer@dcfafbf2")
    .env({"PYTHONPATH": "/root/jamify/src"})
    .add_local_python_source("jamify", copy=False)
    .add_local_file("jamify/configs/jam_infer.yaml", remote_path="/root/jamify/configs/jam_infer.yaml", copy=False)
)

volumes = {
    "/mnt/models": modal.Volume.from_name("models"),
    "/mnt/outputs": modal.Volume.from_name("outputs", create_if_missing=True),
}


@app.function(image=image, volumes=volumes, gpu="H100", timeout=600)
def main(input_data: dict, style_prompt: str):
    """
    Generate song using jamify with provided input data and style prompt.
    
    Args:
        input_data: Dictionary containing input configuration (from input.json)
        style_prompt: String containing the style prompt for generation
    """
    # Create inputs directory in container
    os.makedirs("/root/jamify/inputs", exist_ok=True)
    
    # Write input.json to container
    input_json_path = "/root/jamify/inputs/input.json"
    with open(input_json_path, "w") as f:
        json.dump(input_data, f, indent=2)
    
    # Write style prompt to container
    style_prompt_path = "/root/jamify/inputs/style_prompt.txt"
    with open(style_prompt_path, "w") as f:
        f.write(style_prompt)
    
    # Update input data to use container paths
    container_input_data = []
    for item in input_data:
        updated_item = item.copy()
        # Update prompt_path to point to our written file
        updated_item["prompt_path"] = "inputs/style_prompt.txt"
        container_input_data.append(updated_item)
    
    # Write updated input.json
    with open(input_json_path, "w") as f:
        json.dump(container_input_data, f, indent=2)
    
    # Use direct model path to avoid Modal volume symlink issues
    output_dir = f"/mnt/outputs/{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    subprocess.run([
        "python", "-m", "jam.infer",
        "evaluation.checkpoint_path=/mnt/models/jam-0.5/jam-0_5.safetensors",
        f"evaluation.output_dir={output_dir}",
        "config=/root/jamify/configs/jam_infer.yaml"
    ], cwd="/root/jamify", shell=False, check=True)
    
    return output_dir


@app.local_entrypoint()
def local_main():
    """
    Local entrypoint that reads input files and calls the remote function.
    """
    # Read input.json
    with open("song-gen/input.json", "r") as f:
        input_data = json.load(f)
    
    # Read style prompt
    with open("song-gen/style_prompt.txt", "r") as f:
        style_prompt = f.read().strip()
    
    # Call remote function
    output_dir = main.remote(input_data, style_prompt)
    print(f"Song generation completed. Output saved to: {output_dir}")
    return output_dir
