# DO NOT USE we have model on modal volumes

from huggingface_hub import snapshot_download
import subprocess

if __name__ == "__main__":
    paths = snapshot_download(repo_id="declare-lab/jam-0.5")
    # paths = "/mnt/models/jam-0.5"
    subprocess.run([
        "python", "-m", "jam.infer",
        "config=configs/jam_infer.yaml",
        f"evaluation.checkpoint_path={paths}/jam-0_5.safetensors"
    ])
