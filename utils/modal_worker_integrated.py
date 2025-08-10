import modal
import os

### MODAL APP NAME ###
modal_app_name = "integrated-jamify-pipeline"

app = modal.App(modal_app_name)

# Modal image setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["ffmpeg", "git"])
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("requirements.txt", remote_path="/root/requirements.txt")
    .add_local_dir("jamify/src", remote_path="/root/jamify/src", copy=True)
    .add_local_dir("jamify/configs", remote_path="/root/jamify/configs", copy=True)
    .add_local_dir("jamify/inputs", remote_path="/root/jamify/inputs", copy=True)
    .add_local_dir("jamify/public", remote_path="/root/jamify/public", copy=True)
    .add_local_dir("lyrics-aligner", remote_path="/root/lyrics-aligner", copy=True)
    .add_local_dir("lyricsgeneration", remote_path="/root/lyricsgeneration", copy=True)
    .add_local_file("yt_to_resonite_visemes.py", remote_path="/root/yt_to_resonite_visemes.py")
    .add_local_file("stitch_audio_video.py", remote_path="/root/stitch_audio_video.py")
    .env({"PYTHONPATH": "/root/jamify/src:/root"})
    .run_commands([
        # Download NLTK data during image build instead of runtime
        "python -c 'import nltk; nltk.download(\"cmudict\", download_dir=\"/root/nltk_data\")'",
        "mkdir -p /root/nltk_data",
    ])
    .env({"NLTK_DATA": "/root/nltk_data"})
)

volumes = {
    "/mnt/models": modal.Volume.from_name("models"),
    "/mnt/outputs": modal.Volume.from_name("my-output-volume", create_if_missing=True)
}

# Import the modal-registered functions
from yt_to_resonite_visemes import extract_visemes
from lyricsgeneration.robust_lyrics_gen import generate_lyrics_from_visemes  
from stitch_audio_video import stitch_audio_with_video

@app.function(image=image, volumes=volumes, cpu=2, timeout=1200)
def prepare_jamify_inputs(youtube_url: str, max_duration: int = 60):
    """Prepare inputs for Jamify by running the pipeline steps"""
    import json
    import subprocess
    from pathlib import Path
    
    print("=== PREPARING JAMIFY INPUTS ===")
    print(f"Processing: {youtube_url} ({max_duration}s)")
    
    # Step 1: Extract visemes
    print("\n--- Step 1: Extract Visemes ---")
    visemes = extract_visemes.remote(youtube_url, max_duration)
    print(f"Extracted {len(visemes)} visemes")
    
    # Step 2: Generate lyrics chunks
    print("\n--- Step 2: Generate Lyrics ---") 
    lyrics_chunks = generate_lyrics_from_visemes.remote(visemes)
    print(f"Generated {len(lyrics_chunks)} lyric chunks")
    
    # Step 3: Convert to Jamify input format
    print("\n--- Step 3: Convert to Jamify Format ---")
    
    # Convert lyrics chunks to Jamify word format
    jamify_lyrics = []
    for chunk in lyrics_chunks:
        # Simple word distribution across chunk duration
        words = ["learning", "is", "fun", "and", "educational"] 
        chunk_duration = chunk['t1'] - chunk['t0']
        word_duration = chunk_duration / len(words)
        
        for i, word in enumerate(words):
            word_start = chunk['t0'] + (i * word_duration)
            word_end = min(word_start + word_duration, chunk['t1'])
            
            jamify_lyrics.append({
                "start": word_start,
                "end": word_end,
                "word": word
            })
    
    # Create Jamify input files
    jamify_root = Path("/root/jamify")
    inputs_dir = jamify_root / "inputs"
    
    # Input configuration
    input_config = {
        "id": "pipeline_generated",
        "duration": max_duration,
        "prompt": "Educational pop song with clear vocals"
    }
    
    # Create input.json
    input_data = [{
        "id": input_config["id"],
        "audio_path": f"inputs/{input_config['id']}.mp3",
        "lrc_path": f"inputs/{input_config['id']}.json", 
        "duration": input_config["duration"],
        "prompt_path": f"inputs/{input_config['id']}.txt"
    }]
    
    with open(inputs_dir / "input.json", "w") as f:
        json.dump(input_data, f, indent=2)
    
    # Create lyrics JSON
    with open(inputs_dir / f"{input_config['id']}.json", "w") as f:
        json.dump(jamify_lyrics, f, indent=2)
    
    # Create prompt file
    with open(inputs_dir / f"{input_config['id']}.txt", "w") as f:
        f.write(input_config["prompt"])
    
    # Create placeholder reference audio
    placeholder_audio = inputs_dir / f"{input_config['id']}.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={input_config['duration']}", 
        "-codec:a", "mp3", "-b:a", "192k", str(placeholder_audio)
    ], check=True)
    
    print(f"✅ Jamify inputs prepared: {len(jamify_lyrics)} words")
    return {
        "input_id": input_config["id"],
        "word_count": len(jamify_lyrics),
        "duration": input_config["duration"],
        "youtube_url": youtube_url
    }


@app.function(image=image, volumes=volumes, gpu="H100", timeout=600)
def run_jamify():
    """Run the Jamify inference on prepared inputs"""
    print("=== RUNNING JAMIFY INFERENCE ===")
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

    print("\n=== Extracting Generated Files ===")
    os.system("cp -r /root/jamify/outputs /mnt/outputs/")
    volumes["/mnt/outputs"].commit()
    print("✅ Files copied to my-output-volume")
    return True

@app.function(image=image, volumes=volumes, gpu="H100", timeout=1800)
def main(youtube_url: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ", max_duration: int = 60):
    """Main integrated pipeline function"""
    print("=== INTEGRATED PIPELINE START ===")
    
    try:
        # Phase 1: Prepare Jamify inputs from YouTube video
        print("\n=== PHASE 1: Preparing Jamify Inputs ===")
        prep_result = prepare_jamify_inputs.remote(youtube_url, max_duration)
        print(f"✅ Phase 1 Complete: {prep_result}")
        
        # Phase 2: Run Jamify inference
        print("\n=== PHASE 2: Running Jamify ===")
        jamify_result = run_jamify.remote()
        print("✅ Phase 2 Complete: Jamify finished")
        
        # Phase 3: Stitch video with generated audio
        print("\n=== PHASE 3: Stitching Video ===")
        video_result = stitch_audio_with_video.remote(youtube_url, "pipeline_generated.mp3", max_duration)
        print(f"✅ Phase 3 Complete: {video_result}")
        
        return {
            "success": True,
            "youtube_url": youtube_url,
            "duration": max_duration,
            "prep_result": prep_result,
            "jamify_result": jamify_result,
            "video_result": video_result,
            "message": "Complete pipeline finished successfully!"
        }
        
    except Exception as e:
        print(f"❌ Integrated pipeline failed: {e}")
        return {"success": False, "error": str(e)}

@app.local_entrypoint()
def local_main():
    main.remote()
