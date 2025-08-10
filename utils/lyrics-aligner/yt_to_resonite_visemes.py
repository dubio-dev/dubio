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
    """Extract clean text from VTT subtitle file with improved robustness"""
    lines = []
    
    with open(sub_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Split into blocks and process each one
    blocks = re.split(r'\n\s*\n', content)
    
    for block in blocks:
        block_lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not block_lines:
            continue
            
        # Find text lines (skip timestamps, numbers, headers)
        text_lines = []
        for line in block_lines:
            # Skip VTT headers
            if line.upper() in ["WEBVTT", "KIND: CAPTIONS", "LANGUAGE: EN"]:
                continue
            # Skip timestamp lines
            if re.match(r'\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}', line):
                continue
            # Skip sequence numbers
            if line.isdigit():
                continue
            # Skip NOTE lines
            if line.upper().startswith('NOTE'):
                continue
                
            text_lines.append(line)
        
        # Process text lines
        for line in text_lines:
            cleaned = clean_vtt_text(line)
            if is_valid_caption_text(cleaned):
                # Only add if it's not a duplicate
                if not lines or cleaned != lines[-1]:
                    lines.append(cleaned)
    
    return lines

def clean_vtt_text(text: str) -> str:
    """Clean VTT text with comprehensive tag removal"""
    if not text:
        return ""
    
    # Remove all timing tags (more flexible pattern)
    text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', text)
    
    # Remove VTT markup tags
    text = re.sub(r'</?c[^>]*>', '', text)  # Color tags
    text = re.sub(r'</?v[^>]*>', '', text)  # Voice tags  
    text = re.sub(r'</?i[^>]*>', '', text)  # Italic tags
    text = re.sub(r'</?b[^>]*>', '', text)  # Bold tags
    text = re.sub(r'</?u[^>]*>', '', text)  # Underline tags
    text = re.sub(r'</?ruby[^>]*>', '', text)  # Ruby tags
    text = re.sub(r'</?rt[^>]*>', '', text)  # Ruby text tags
    
    # Remove any remaining HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove music/sound effect markers
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\([^\)]*music[^\)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\([^\)]*sound[^\)]*\)', '', text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def is_valid_caption_text(text: str) -> bool:
    """Check if text is valid caption content (more permissive)"""
    if not text:
        return False
    
    # Minimum length check (reduced from 3 to 1)
    if len(text) < 1:
        return False
    
    # Skip obvious non-content
    if text.lower() in ['music', 'applause', 'laughter', '♪', '♫']:
        return False
    
    # Allow single words and concatenated text (removed space requirement)
    # This handles cases like "happyyoucoulddie" from the terminal output
    
    # Check if it contains at least some alphabetic characters
    if not re.search(r'[a-zA-Z]', text):
        return False
    
    return True

def parse_vtt_with_timing(sub_path: Path):
    """Extract text segments with timing information from VTT subtitle file with improved robustness"""
    segments = []
    
    with open(sub_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Split into blocks separated by double newlines
    blocks = re.split(r'\n\s*\n', content)
    
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 2:
            continue
            
        # Look for timestamp line
        timestamp_line = None
        text_lines = []
        
        for line in lines:
            # More flexible timestamp matching
            if re.match(r'\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}', line):
                timestamp_line = line
            elif (line.upper() not in ["WEBVTT", "KIND: CAPTIONS", "LANGUAGE: EN"] and
                  not line.isdigit() and
                  not line.upper().startswith('NOTE')):
                text_lines.append(line)
        
        if not timestamp_line or not text_lines:
            continue
            
        # Parse timestamp with more flexible regex
        match = re.match(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})', timestamp_line)
        if not match:
            continue
            
        # Convert to seconds
        start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
        end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
        
        start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
        end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0
        
        # Skip invalid time ranges
        if end_time <= start_time:
            continue
        
        # Clean text using improved function
        text = ' '.join(text_lines)
        text = clean_vtt_text(text)
        
        # Use improved validation (more permissive)
        if is_valid_caption_text(text):
            # Check for overlapping segments and merge if very close
            if (segments and 
                abs(start_time - segments[-1]['end']) < 0.1 and
                segments[-1]['text'] == text):
                # Extend previous segment instead of adding duplicate
                segments[-1]['end'] = max(segments[-1]['end'], end_time)
            else:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })
    
    # Sort segments by start time and remove duplicates
    segments.sort(key=lambda x: x['start'])
    
    # Remove exact duplicates
    unique_segments = []
    for segment in segments:
        if not unique_segments or segment != unique_segments[-1]:
            unique_segments.append(segment)
    
    return unique_segments

def run_lyrics_aligner(audio_path: Path, lyrics_path: Path, out_json: Path, vtt_path: Path = None):
    """Run phoneme alignment using built-in logic with optional VTT timing"""
    print("[INFO] Running built-in phoneme alignment...")
    
    # Get audio duration using ffprobe
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())
    
    print(f"[INFO] Audio duration: {duration:.1f} seconds")
    
    # Try to use VTT timing if available
    if vtt_path and vtt_path.exists():
        print(f"[INFO] Using VTT timing from: {vtt_path}")
        segments = parse_vtt_with_timing(vtt_path)
        if segments:
            print(f"[INFO] Found {len(segments)} timed segments")
            phoneme_segments = align_segments_to_phonemes(segments)
        else:
            print("[WARN] No valid segments found in VTT, falling back to text-only alignment")
            # Fallback to text-only alignment
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            phoneme_segments = align_text_to_phonemes(text, duration)
    else:
        print("[INFO] No VTT file provided, using text-only alignment")
        # Read lyrics text
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Convert text to phonemes with timing
        phoneme_segments = align_text_to_phonemes(text, duration)
    
    # Save phoneme data
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(phoneme_segments, f, indent=2)
    
    print(f"[INFO] Generated {len(phoneme_segments)} phoneme segments")

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

def align_segments_to_phonemes(segments: list) -> list:
    """Convert VTT segments with timing to timed phoneme segments using CMU dictionary"""
    if not segments:
        return []
    
    # Load CMU dictionary for phoneme lookup
    cmu_dict = load_cmu_dictionary()
    
    all_phoneme_segments = []
    
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        segment_duration = end_time - start_time
        
        # Clean and split text into words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            continue
        
        # Convert words to phonemes
        segment_phonemes = []
        for word in words:
            word_phonemes = get_word_phonemes(word, cmu_dict)
            segment_phonemes.extend(word_phonemes)
        
        if not segment_phonemes:
            continue
        
        # Distribute phonemes evenly within this segment's time range
        phoneme_duration = segment_duration / len(segment_phonemes)
        current_time = start_time
        
        for phoneme in segment_phonemes:
            phoneme_start = current_time
            phoneme_end = current_time + phoneme_duration
            
            all_phoneme_segments.append({
                "t0": round(phoneme_start, 3),
                "t1": round(phoneme_end, 3),
                "phoneme": phoneme
            })
            
            current_time = phoneme_end
    
    return all_phoneme_segments

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
        run_lyrics_aligner(audio, lyrics_txt, phonemes_json, subs)
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
