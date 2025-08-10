# Hardcoded YouTube URL + MP3 -> writes OUT (MP4 with H.264 + AAC)
# Requirements: pip install yt-dlp  |  install ffmpeg on PATH

import subprocess, tempfile, pathlib

# Modal integration
try:
    from modal_worker_integrated import app, image, volumes
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

# === EDIT THESE ===
URL = "https://youtu.be/dQw4w9WgXcQ"  # put your YouTube link here
MP3 = "lyrics_basic_math.mp3"                      # path to your local .mp3
OUT = "video_out.mp4"                    # output file name
# ===================

def main(url=None, mp3_path=None, output_path=None, max_duration=None):
    url = url or URL
    mp3_path = mp3_path or MP3
    output_path = output_path or OUT
    
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise SystemExit("Please `pip install yt-dlp` first.")

    with tempfile.TemporaryDirectory() as td:
        tdir = pathlib.Path(td)

        # Download best video-only (prefer mp4), no audio
        ydl_opts = {
            "format": "bestvideo[ext=mp4]/bestvideo",
            "outtmpl": str(tdir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        # Pick the most recently modified file as the video
        vids = [p for p in tdir.iterdir() if p.is_file()]
        if not vids:
            raise SystemExit("Failed to download video.")
        video = max(vids, key=lambda p: p.stat().st_mtime)

        # Get MP3 duration for precise timing
        probe_cmd = ["ffprobe", "-i", mp3_path, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        
        # Apply max_duration if specified
        if max_duration:
            duration = min(duration, max_duration)
        
        # Replace audio with MP3 (always re-encode video to be robust)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", mp3_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(duration),  # Explicitly set duration to match MP3
            "-avoid_negative_ts", "make_zero",  # Ensure proper sync
            "-movflags", "+faststart",
            output_path,
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
        print(f"Done: {output_path}")
        return output_path

# Modal function registration
if MODAL_AVAILABLE:
    @app.function(image=image, volumes=volumes, cpu=1, timeout=600)
    def stitch_audio_with_video(youtube_url: str, audio_filename: str, max_duration: int = 60):
        """Modal wrapper for main stitch function"""
        audio_path = f"/mnt/outputs/{audio_filename}"
        output_path = "/mnt/outputs/final_video.mp4"
        result = main(youtube_url, audio_path, output_path, max_duration)
        volumes["/mnt/outputs"].commit()
        return {"success": True, "output_path": output_path}

if __name__ == "__main__":
    main()
