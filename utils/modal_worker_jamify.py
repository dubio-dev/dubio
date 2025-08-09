# todo

import modal
import os

### MODAL APP NAME ###

modal_app_name = "jamify-worker-dev"

### MODAL APP NAME ###

app = modal.App(modal_app_name)

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("jamify/requirements.txt")
    .pip_install("git+https://github.com/xhhhhang/DeepPhonemizer@dcfafbf2")
    .add_local_python_source("jamify", copy=True)
    .add_local_dir("jamify/configs", remote_path="/root/jamify/configs", copy=True)
    .add_local_dir("jamify/inputs", remote_path="/root/jamify/inputs", copy=True)
    .add_local_dir("jamify/public", remote_path="/root/jamify/public", copy=True)
    .env({"PYTHONPATH": "/root/jamify/src"})    # ?? maybe unnecessary ??
)

volumes = { "/mnt/models": modal.Volume.from_name("models") }


@app.function(image=image, volumes=volumes, gpu="H100", timeout=600)
def main():
    print("Hello World")
    print("=== Volume Debug Info ===")
    print("Contents of /mnt/models/jam-0.5:")
    print(os.listdir("/mnt/models/jam-0.5"))
    
    print("\n=== Symlink Debug ===")
    os.system("stat /mnt/models/jam-0.5/jam-0_5.safetensors")
    
    print("\n=== Trying to resolve symlink ===")
    os.system("ls -la /mnt/models/jam-0.5/jam-0_5.safetensors")
    os.system("readlink -f /mnt/models/jam-0.5/jam-0_5.safetensors")
    
    print("\n=== Check if target exists ===")
    os.system("ls -la /mnt/models/jam-0.5/")
    
    print("\n=== Try to access file directly ===")
    os.system("file /mnt/models/jam-0.5/jam-0_5.safetensors")

    print("=== Check for blobs at root ===")
    os.system("find /mnt/models -name 'blobs' -type d")
    
    print("\n=== Running inference ===")
    os.system("cd /root/jamify && python inference.py")


@app.local_entrypoint()
def local_main():
    main.remote()
