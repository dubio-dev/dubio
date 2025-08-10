# Hardcoded YouTube URL + MP3 -> writes OUT (MP4 with H.264 + AAC)
# Requirements: pip install yt-dlp  |  install ffmpeg on PATH

import subprocess, tempfile, pathlib

# === EDIT THESE ===
URL = "https://youtu.be/dQw4w9WgXcQ"  # put your YouTube link here
MP3 = "lyrics_basic_math.mp3"                      # path to your local .mp3
OUT = "video_out.mp4"                    # output file name
# ===================

def main():
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
            ydl.extract_info(URL, download=True)

        # Pick the most recently modified file as the video
        vids = [p for p in tdir.iterdir() if p.is_file()]
        if not vids:
            raise SystemExit("Failed to download video.")
        video = max(vids, key=lambda p: p.stat().st_mtime)

        # Get MP3 duration for precise timing
        probe_cmd = ["ffprobe", "-i", MP3, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        
        # Replace audio with MP3 (always re-encode video to be robust)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", MP3,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(duration),  # Explicitly set duration to match MP3
            "-avoid_negative_ts", "make_zero",  # Ensure proper sync
            "-movflags", "+faststart",
            OUT,
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
        print(f"Done: {OUT}")

if __name__ == "__main__":
    main()
