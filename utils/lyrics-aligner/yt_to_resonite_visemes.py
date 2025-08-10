#!/usr/bin/env python3
"""
YouTube to Resonite Visemes Pipeline

- Downloads audio (wav) + English auto-captions (yt-dlp + ffmpeg)
- Converts captions to text
- Runs lyrics-aligner to get time-stamped phonemes
- Maps CMUdict ARPAbet phonemes to minimal Resonite viseme set:
  {AA, CH, DD, E, FF, IH, KK, NN, OH, OU, PP, RR, SS, TH, Silence}

Outputs:
- phonemes.json: Raw phoneme timing data
- visemes_resonite_min.json: Final viseme data with timing
"""

from pathlib import Path
import subprocess
import json
import re
import sys
import os
import argparse

# ---------------- Configuration ----------------
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
OUTDIR = Path("out")
LANGUAGE = "en"

# ---------------- CMUdict phones (39, stressless) ----------------
CMU_PHONES = [
    "AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW",
    "B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","P","R","S","SH",
    "T","TH","V","W","Y","Z","ZH"
]

# ---------------- Minimal Resonite Viseme Mapping ----------------
STRESS_RE = re.compile(r"[0-2]$")

ARPABET_TO_VISEME = {
    # Silence bucket
    "SIL": "Silence", "SP": "Silence", "NSN": "Silence", "BR": "Silence", "PAU": "Silence",
    # PP (lip closure)
    "P": "PP", "B": "PP", "M": "PP",
    # FF (lip-teeth contact)
    "F": "FF", "V": "FF",
    # TH (tongue-teeth)
    "TH": "TH", "DH": "TH",
    # DD (tongue tip)
    "T": "DD", "D": "DD",
    # SS (sibilants)
    "S": "SS", "Z": "SS",
    # CH (tongue shapes)
    "SH": "CH", "ZH": "CH", "CH": "CH", "JH": "CH",
    # KK (back tongue)
    "K": "KK", "G": "KK",
    # NN (nasal/lateral)
    "N": "NN", "NG": "NN", "L": "NN",
    # RR (r-sounds)
    "R": "RR",
    # Glides
    "W": "OU", "Y": "IH", "HH": "E",
    # Vowels
    "AA": "AA", "AE": "AA", "AH": "AA", "AY": "AA",
    "EH": "E", "ER": "E", "EY": "E",
    "IH": "IH", "IY": "IH",
    "AO": "OH", "OW": "OH", "OY": "OH",
    "UH": "OU", "UW": "OU",
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_cmu_dictionary():
    """Load CMU Pronouncing Dictionary from nltk or download if needed"""
    try:
        import nltk
        from nltk.corpus import cmudict
        
        # Set NLTK data path if environment variable is set
        nltk_data_path = os.getenv('NLTK_DATA')
        if nltk_data_path:
            nltk.data.path.insert(0, nltk_data_path)
        
        # Try to load CMU dictionary, download if needed
        try:
            cmudict.words()
        except LookupError:
            print("[INFO] Downloading CMU dictionary...")
            if nltk_data_path:
                nltk.download('cmudict', download_dir=nltk_data_path)
            else:
                nltk.download('cmudict')
        
        return cmudict.dict()
    except ImportError:
        print("[WARN] nltk not available, using fallback dictionary")
        return None

def extract_words_from_text(text_file):
    """Extract unique words from a text file"""
    words = set()
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Remove apostrophes then extract alphabetic words
            line_clean = line.lower().replace("'", "")
            line_words = re.findall(r'\b[a-zA-Z]+\b', line_clean)
            words.update(line_words)
    return sorted(list(words))

def setup_lyrics_aligner():
    """Basic setup - just ensure output directory exists"""
    ensure_dir(OUTDIR)
    print("[INFO] Setup complete - using built-in phoneme alignment")

def run(cmd):
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def download_audio_and_captions(url, outdir: Path, lang: str):
    ensure_dir(outdir)
    outtpl = str(outdir / "%(id)s.%(ext)s")
    run([
        "yt-dlp", "--no-playlist",
        "--extract-audio", "--audio-format", "wav", "--audio-quality", "0",
        "--write-auto-sub", "--sub-lang", lang, "--sub-format", "vtt/srt/best",
        "-o", outtpl, url
    ])
    audio = sorted(outdir.glob("*.wav"), key=lambda p: p.stat().st_mtime)[-1]
    subs = sorted(list(outdir.glob("*.vtt")) + list(outdir.glob("*.srt")), key=lambda p: p.stat().st_mtime)[-1]
    return audio, subs

def captions_to_text(sub_path: Path):
    """Extract clean text from VTT subtitle file"""
    lines = []
    
    with open(sub_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Split into lines and process
    for line in content.split('\n'):
        line = line.strip()
        
        # Skip header lines, timestamps, numbers, and empty lines
        if (not line or 
            line.upper() in ["WEBVTT", "KIND: CAPTIONS", "LANGUAGE: EN"] or
            re.match(r'\d{2}:\d{2}:\d{2}\.\d{3}\s+-->', line) or
            line.isdigit()):
            continue
        
        # Clean the text
        cleaned = line
        
        # Remove timing tags like <00:00:19.039><c> no</c>
        cleaned = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}><c>\s*', '', cleaned)
        cleaned = re.sub(r'</c>', '', cleaned)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Normalize spacing
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Only keep lines that look like proper sentences (have spaces and reasonable length)
        if (cleaned and 
            ' ' in cleaned and  # Has spaces (multiple words)
            len(cleaned) > 3 and  # Reasonable length
            not cleaned.startswith('[') and  # Not music markers
            cleaned != '[Music]'):
            
            # Only add if it's not a duplicate
            if not lines or cleaned != lines[-1]:
                lines.append(cleaned)
    
    return lines

def run_lyrics_aligner(audio_path: Path, lyrics_path: Path, out_json: Path):
    """Run phoneme alignment using built-in logic instead of external lyrics-aligner"""
    print("[INFO] Running built-in phoneme alignment...")
    
    # Read the lyrics
    with open(lyrics_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Get audio duration using ffprobe
    try:
        cmd = ["ffprobe", "-i", str(audio_path), "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
        duration = float(subprocess.check_output(cmd).decode().strip())
        print(f"[INFO] Audio duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"[WARN] Could not get audio duration: {e}")
        duration = 30.0  # Default fallback
    
    # Process text to phonemes
    phonemes = align_text_to_phonemes(text, duration)
    
    # Save phoneme data
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(phonemes, f, indent=2)
    
    print(f"[INFO] Generated {len(phonemes)} phoneme segments")

def align_text_to_phonemes(text: str, duration: float) -> list:
    """Convert text to timed phoneme segments using CMU dictionary"""
    # Clean and split text into words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return []
    
    # Load CMU dictionary for phoneme lookup
    cmu_dict = load_cmu_dictionary()
    
    # Convert words to phonemes
    all_phonemes = []
    for word in words:
        word_phonemes = get_word_phonemes(word, cmu_dict)
        all_phonemes.extend(word_phonemes)
    
    if not all_phonemes:
        return []
    
    # Distribute phonemes evenly across the audio duration
    phoneme_duration = duration / len(all_phonemes)
    
    # Create timed phoneme segments
    phoneme_segments = []
    current_time = 0.0
    
    for phoneme in all_phonemes:
        start_time = current_time
        end_time = current_time + phoneme_duration
        
        phoneme_segments.append({
            "t0": round(start_time, 3),
            "t1": round(end_time, 3),
            "phoneme": phoneme
        })
        
        current_time = end_time
    
    return phoneme_segments

def get_word_phonemes(word: str, cmu_dict=None) -> list:
    """Get phonemes for a word using CMU dictionary with simple fallback"""
    if cmu_dict and word in cmu_dict:
        # Get the first pronunciation and remove stress markers
        phonemes = cmu_dict[word][0]
        return [p.rstrip('012') for p in phonemes]
    
    # Fallback: simple phonetic approximation - just use schwa sound
    print(f"[WARN] Word '{word}' not found in CMU dictionary, using fallback")
    return ["AH"]  # Default to schwa sound

def normalize_phone(p: str) -> str:
    p = (p or "").strip().upper()
    p = STRESS_RE.sub("", p)        # strip 0/1/2
    p = re.sub(r"[^A-Z]", "", p)    # defensive
    return p

def to_minimal_visemes(phoneme_json: Path, out_json: Path):
    with open(phoneme_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Coverage sanity check (only warn once)
    missing = set()

    result = []
    last = None
    for item in data:
        start = float(item.get("t0", 0.0))
        end   = float(item.get("t1", start))
        ph    = normalize_phone(str(item.get("phoneme", "SIL")))

        if ph not in ARPABET_TO_VISEME:
            if ph not in {"SIL", "SP", "NSN", "BR", "PAU"} and ph not in CMU_PHONES:
                missing.add(ph)
            # map unknowns to neutral E
            vis = "E"
        else:
            vis = ARPABET_TO_VISEME[ph]

        row = {"t0": start, "t1": end, "phoneme": ph, "viseme": vis}

        if last and last["viseme"] == row["viseme"] and abs(row["t0"] - last["t1"]) < 0.02:
            last["t1"] = max(last["t1"], row["t1"])
        else:
            result.append(row)
            last = row

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if missing:
        print("[WARN] Unrecognized phones encountered (mapped to 'E'):", ", ".join(sorted(missing)), file=sys.stderr)

def process_youtube_to_visemes(youtube_url: str, max_duration: int = None) -> list:
    """
    Main processing function: YouTube URL -> Visemes
    
    Args:
        youtube_url: YouTube video URL
        max_duration: Optional max duration in seconds
        
    Returns:
        List of viseme dictionaries with timing
    """
    
    try:
        # Step 1: Download audio and captions
        print(f"[1/4] Downloading audio and captions from: {youtube_url}")
        audio, subs = download_audio_and_captions(youtube_url, OUTDIR, LANGUAGE)

        # Step 2: Extract and clean captions
        print("[2/4] Processing captions...")
        lyrics_txt = OUTDIR / "captions.txt"
        captions_lines = captions_to_text(subs)
        cleaned_captions = [line.replace("'", "") for line in captions_lines]
        lyrics_txt.write_text("\n".join(cleaned_captions), encoding="utf-8")
        print(f"[OK] Captions saved to: {lyrics_txt}")

        # Step 3: Set up lyrics aligner and run phoneme alignment
        print("[3/4] Running phoneme alignment...")
        setup_lyrics_aligner()
        
        phonemes_json = OUTDIR / "phonemes.json"
        run_lyrics_aligner(audio, lyrics_txt, phonemes_json)
        print(f"[OK] Phonemes saved to: {phonemes_json}")

        # Step 4: Convert phonemes to visemes
        print("[4/4] Converting phonemes to visemes...")
        visemes_json = OUTDIR / "visemes_resonite_min.json"
        to_minimal_visemes(phonemes_json, visemes_json)
        print(f"[DONE] Visemes saved to: {visemes_json}")
        
        # Load and return visemes
        with open(visemes_json, 'r') as f:
            visemes = json.load(f)
        
        # Truncate to max_duration if specified
        if max_duration:
            truncated = []
            for item in visemes:
                if item['t0'] < max_duration:
                    item['t1'] = min(item['t1'], max_duration)
                    truncated.append(item)
                else:
                    break
            visemes = truncated
            print(f"[INFO] Truncated to {max_duration}s: {len(truncated)} visemes")
        
        return visemes
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        raise

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract visemes from YouTube video")
    parser.add_argument("youtube_url", nargs='?', default=DEFAULT_YOUTUBE_URL, 
                       help="YouTube video URL")
    parser.add_argument("--max-duration", type=int, help="Maximum duration in seconds")
    parser.add_argument("--output", help="Output JSON file path (default: out/visemes_resonite_min.json)")
    
    args = parser.parse_args()
    
    try:
        # Process YouTube video to visemes
        visemes = process_youtube_to_visemes(args.youtube_url, args.max_duration)
        
        # Save to custom output file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(visemes, f, indent=2)
            print(f"Visemes also saved to: {args.output}")
        
        print(f"\n✅ Successfully extracted {len(visemes)} visemes")
        print(f"Duration: {visemes[-1]['t1']:.1f} seconds" if visemes else "No visemes generated")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
