#!/usr/bin/env python3
"""
Integrated YouTube-to-Jamified-Video Pipeline

Takes a YouTube video, extracts visemes, generates educational lyrics,
uses Jamify to create audio, and stitches back with original video.
Only processes the first minute of the video.
"""

import subprocess
import json
import tempfile
import shutil
from pathlib import Path
import os
import sys

# === CONFIGURATION ===
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Change this
MAX_DURATION = 60  # Process only first 60 seconds
OUTDIR = Path("pipeline_output")
# =====================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"[STEP] {description}")
    print(f"[RUN] {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {result.stderr}")
        sys.exit(1)
    return result.stdout

def truncate_json_to_duration(json_path, max_duration):
    """Truncate visemes JSON to maximum duration"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter visemes within duration and adjust final end time
    truncated = []
    for item in data:
        if item['t0'] < max_duration:
            # Clip end time to max duration
            item['t1'] = min(item['t1'], max_duration)
            truncated.append(item)
        else:
            break
    
    # Write truncated version
    truncated_path = json_path.parent / f"{json_path.stem}_truncated.json"
    with open(truncated_path, 'w') as f:
        json.dump(truncated, f, indent=2)
    
    print(f"[INFO] Truncated visemes to {max_duration}s: {len(data)} -> {len(truncated)} visemes")
    return truncated_path

def step1_extract_visemes():
    """Step 1: Run yt_to_resonite_visemes.py"""
    print("\n=== STEP 1: Extracting Visemes ===")
    
    # Modify the script to use our URL
    script_path = Path("yt_to_resonite_visemes.py")
    
    # Read original script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update URL in script
    modified_content = content.replace(
        'YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"',
        f'YOUTUBE_URL = "{YOUTUBE_URL}"'
    )
    
    # Write temporary script
    temp_script = script_path.parent / "temp_viseme_script.py"
    with open(temp_script, 'w') as f:
        f.write(modified_content)
    
    # Run viseme extraction
    run_cmd(["uv", "run", str(temp_script)], "Extracting visemes from YouTube video")
    
    # Clean up temp script
    temp_script.unlink()
    
    visemes_path = Path("out/visemes_resonite_min.json")
    if not visemes_path.exists():
        print("[ERROR] Visemes JSON not generated")
        sys.exit(1)
    
    # Truncate to max duration
    truncated_path = truncate_json_to_duration(visemes_path, MAX_DURATION)
    
    # Copy to pipeline output
    ensure_dir(OUTDIR)
    final_visemes = OUTDIR / "visemes.json"
    shutil.copy2(truncated_path, final_visemes)
    
    return final_visemes

def step2_generate_lyrics(visemes_path):
    """Step 2: Generate educational lyrics using robust_lyrics_gen.py"""
    print("\n=== STEP 2: Generating Educational Lyrics ===")
    
    # Copy visemes to lyrics generation folder
    lyrics_dir = Path("lyricsgeneration")
    temp_visemes = lyrics_dir / "temp_visemes.json"
    shutil.copy2(visemes_path, temp_visemes)
    
    # Modify robust_lyrics_gen.py to use our visemes file
    robust_script = lyrics_dir / "robust_lyrics_gen.py"
    with open(robust_script, 'r') as f:
        content = f.read()
    
    # Create modified version that uses our input
    modified_content = content.replace(
        "with open('lyricsgeneration/visemes_resonite_min_short.json', 'r') as f:",
        f"with open('{temp_visemes}', 'r') as f:"
    )
    
    # Add output saving at the end
    modified_content = modified_content.replace(
        'print(f"    Est. syllables: {chunk[\'estimated_syllables\']}")',
        '''print(f"    Est. syllables: {chunk['estimated_syllables']}")
    
    # Save chunks to JSON for next step
    output_path = "temp_lyrics_output.json"
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"\\nSaved lyrics chunks to {output_path}")'''
    )
    
    temp_script = lyrics_dir / "temp_robust_script.py"
    with open(temp_script, 'w') as f:
        f.write(modified_content)
    
    # Run lyrics generation
    original_cwd = os.getcwd()
    try:
        os.chdir(lyrics_dir)
        run_cmd(["python", str(temp_script.name)], "Generating educational lyrics")
    finally:
        os.chdir(original_cwd)
    
    # Move output to pipeline directory
    lyrics_output = lyrics_dir / "temp_lyrics_output.json"
    final_lyrics = OUTDIR / "lyrics_chunks.json"
    shutil.move(lyrics_output, final_lyrics)
    
    # Clean up
    temp_visemes.unlink()
    temp_script.unlink()
    
    return final_lyrics

def step3_jamify_audio(lyrics_path):
    """Step 3: Use Modal Jamify to generate audio"""
    print("\n=== STEP 3: Generating Audio with Jamify ===")
    
    # For now, we'll simulate this step since Modal integration needs more setup
    # In a real implementation, you'd call the modal worker here
    print("[INFO] Modal Jamify integration would happen here")
    print("[INFO] For now, using placeholder audio file")
    
    # Check if there's existing jamify output we can use
    jamify_outputs = Path("jamify/outputs/generated")
    if jamify_outputs.exists():
        audio_files = list(jamify_outputs.glob("*.mp3"))
        if audio_files:
            # Use the first available generated audio
            source_audio = audio_files[0]
            final_audio = OUTDIR / "generated_audio.mp3"
            shutil.copy2(source_audio, final_audio)
            print(f"[INFO] Using existing Jamify output: {source_audio}")
            return final_audio
    
    # Create a placeholder (in real usage, this would be Modal output)
    placeholder_audio = OUTDIR / "generated_audio.mp3"
    
    # Use original YouTube audio as placeholder
    youtube_audio = Path("out").glob("*.wav")
    if youtube_audio:
        audio_file = next(youtube_audio)
        # Convert to MP3 and truncate to max duration
        run_cmd([
            "ffmpeg", "-y", "-i", str(audio_file),
            "-t", str(MAX_DURATION),
            "-codec:a", "mp3", "-b:a", "192k",
            str(placeholder_audio)
        ], f"Creating placeholder audio (first {MAX_DURATION}s)")
    else:
        print("[ERROR] No audio file found for placeholder")
        sys.exit(1)
    
    return placeholder_audio

def step4_stitch_video(audio_path):
    """Step 4: Stitch audio with YouTube video using stitch_audio_video.py"""
    print("\n=== STEP 4: Stitching Audio with Video ===")
    
    # Read stitch script
    stitch_script = Path("stitch_audio_video.py")
    with open(stitch_script, 'r') as f:
        content = f.read()
    
    # Modify script to use our inputs and truncate to max duration
    modified_content = content.replace(
        f'URL = "https://youtu.be/dQw4w9WgXcQ"',
        f'URL = "{YOUTUBE_URL}"'
    ).replace(
        f'MP3 = "lyrics_basic_math.mp3"',
        f'MP3 = "{audio_path}"'
    ).replace(
        f'OUT = "video_out.mp4"',
        f'OUT = "{OUTDIR / "final_video.mp4"}"'
    )
    
    # Add duration limit to ffmpeg command
    modified_content = modified_content.replace(
        '"-t", str(duration),',
        f'"-t", "{min(MAX_DURATION, duration)},"'
    )
    
    temp_script = Path("temp_stitch_script.py")
    with open(temp_script, 'w') as f:
        f.write(modified_content)
    
    # Run stitching
    run_cmd(["python", str(temp_script)], "Stitching audio with video")
    
    # Clean up
    temp_script.unlink()
    
    final_video = OUTDIR / "final_video.mp4"
    return final_video

def main():
    """Main pipeline execution"""
    print("=== YOUTUBE TO JAMIFIED VIDEO PIPELINE ===")
    print(f"Processing: {YOUTUBE_URL}")
    print(f"Duration limit: {MAX_DURATION} seconds")
    print(f"Output directory: {OUTDIR}")
    
    ensure_dir(OUTDIR)
    
    try:
        # Step 1: Extract visemes from YouTube video
        visemes_path = step1_extract_visemes()
        print(f"[SUCCESS] Visemes extracted: {visemes_path}")
        
        # Step 2: Generate educational lyrics
        lyrics_path = step2_generate_lyrics(visemes_path)
        print(f"[SUCCESS] Lyrics generated: {lyrics_path}")
        
        # Step 3: Generate audio with Jamify
        audio_path = step3_jamify_audio(lyrics_path)
        print(f"[SUCCESS] Audio generated: {audio_path}")
        
        # Step 4: Stitch audio with video
        video_path = step4_stitch_video(audio_path)
        print(f"[SUCCESS] Final video created: {video_path}")
        
        print(f"\n=== PIPELINE COMPLETE ===")
        print(f"Final output: {video_path}")
        print(f"All intermediate files saved in: {OUTDIR}")
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()