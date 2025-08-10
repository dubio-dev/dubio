#!/usr/bin/env python3
"""
yt_to_resonite_visemes_minimal_cmudict.py

- Downloads audio (wav) + English auto-captions (yt-dlp + ffmpeg)
- Converts captions to text
- Runs local lyrics-aligner (python -m lyrics_aligner) to get time-stamped phonemes
- STRICTLY maps **CMUdict 39-phone ARPAbet** (stress stripped) to the minimal Resonite viseme set:
  {AA, CH, DD, E, FF, IH, KK, NN, OH, OU, PP, RR, SS, TH, Silence}

Outputs:
- out/phonemes.json
- out/visemes_resonite_min.json
"""

from pathlib import Path
import subprocess
import json
import re
import sys
import shutil
import os

# Modal integration
try:
    from modal_worker_integrated import app, image, volumes
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

# ---------------- Configuration ----------------
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # change this
OUTDIR = Path("out")
LANGUAGE = "en"
ALIGNER_CMD = ["python", "lyrics-aligner/align.py"]

# ---------------- CMUdict phones (39, stressless) ----------------
# Reference set (no stress digits)
CMU_PHONES = [
    "AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW",
    "B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","P","R","S","SH",
    "T","TH","V","W","Y","Z","ZH"
]

# ---------------- Minimal Resonite Viseme Mapping ----------------
# Target visemes: AA, CH, DD, E, FF, IH, KK, NN, OH, OU, PP, RR, SS, TH, Silence
# Grouping rationale:
# - Stops and closures: PP (P,B,M), DD (T,D), KK (K,G)
# - Fricatives/affricates: FF (F,V), TH (TH,DH), SS (S,Z), CH (SH,ZH,CH,JH)
# - Sonorants: NN (N,NG,L), RR (R), glides W→OU (rounded), Y→IH (high-front)
# - Vowels clustered by mouth shape: AA={AA,AE,AH}, E={EH,ER,EY}, IH={IH,IY},
#   OH={AO,OW,OY}, OU={UH,UW}, diphthongs AY→AA, AW→OU.
STRESS_RE = re.compile(r"[0-2]$")

ARPABET_TO_VISEME = {
    # Silence bucket (aligner-specific tokens may show; CMUdict itself doesn't include them)
    "SIL": "Silence", "SP": "Silence", "NSN": "Silence", "BR": "Silence", "PAU": "Silence",
    # PP
    "P": "PP", "B": "PP", "M": "PP",
    # FF
    "F": "FF", "V": "FF",
    # TH
    "TH": "TH", "DH": "TH",
    # DD
    "T": "DD", "D": "DD",
    # SS
    "S": "SS", "Z": "SS",
    # CH
    "SH": "CH", "ZH": "CH", "CH": "CH", "JH": "CH",
    # KK
    "K": "KK", "G": "KK",
    # NN
    "N": "NN", "NG": "NN", "L": "NN",
    # RR
    "R": "RR",
    # Glides
    "W": "OU",
    "Y": "IH",
    "HH": "E",
    # Vowels
    "AA": "AA", "AE": "AA", "AH": "AA",
    "AY": "AA",          # starts with open 'a' shape
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
        
        # Download CMU dictionary if not already downloaded
        try:
            cmudict.words()
        except LookupError:
            nltk.download('cmudict')
        
        return cmudict.dict()
    except ImportError:
        print("[WARN] nltk not available, using fallback dictionary")
        return None

def create_word_to_phonemes_dict(words, cmu_dict=None):
    """Create word-to-phonemes mapping for given words using CMU dictionary"""
    if cmu_dict is None:
        cmu_dict = load_cmu_dictionary()
    
    word2phonemes = {}
    
    for word in words:
        word_lower = word.lower()
        
        if cmu_dict and word_lower in cmu_dict:
            # Get the first pronunciation (most common)
            phonemes = cmu_dict[word_lower][0]
            # Remove stress markers (0, 1, 2)
            phonemes = [p.rstrip('012') for p in phonemes]
            # Convert to space-separated string as expected by lyrics-aligner
            word2phonemes[word_lower] = ' '.join(phonemes)
        else:
            # Fallback: simple phonetic approximation
            # This is a very basic fallback - for production use, you'd want a better solution
            print(f"[WARN] Word '{word}' not found in CMU dictionary, using fallback")
            word2phonemes[word_lower] = "AH"  # Default to schwa sound
    
    return word2phonemes

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
    """Set up the required word-to-phoneme dictionary for lyrics-aligner"""
    import pickle
    
    # Check if the required file already exists
    word2phonemes_file = Path("lyrics-aligner/files/youtube_word2phonemes.pickle")
    if word2phonemes_file.exists():
        return
    
    # Extract words from the captions file
    captions_file = OUTDIR / "captions.txt"
    if not captions_file.exists():
        print("[WARN] Captions file not found, creating basic dictionary")
        # Create a minimal dictionary with common words
        words = ["the", "a", "and", "to", "of", "in", "it", "that", "with", "for", "not", "on", "at", "this", "but", "they", "have", "from", "or", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "into", "him", "time", "two", "more", "go", "no", "way", "could", "my", "than", "first", "been", "call", "who", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part"]
    else:
        words = extract_words_from_text(captions_file)
    
    # Create word-to-phoneme mapping
    word2phonemes = create_word_to_phonemes_dict(words)
    
    # Save the dictionary
    with open(word2phonemes_file, 'wb') as f:
        pickle.dump(word2phonemes, f)
    
    print(f"[SETUP] Created word-to-phoneme dictionary with {len(word2phonemes)} words: {word2phonemes_file}")
    
    # Debug: Show some word-to-phoneme mappings
    print("\n[DEBUG] Sample word-to-phoneme mappings:")
    sample_words = list(word2phonemes.keys())[:10]  # Show first 10 words
    for word in sample_words:
        phonemes = word2phonemes[word]
        print(f"  '{word}' -> {phonemes}")
    
    # Debug: Show how captions would be converted
    print("\n[DEBUG] Sample caption conversion:")
    with open(captions_file, 'r', encoding='utf-8') as f:
        caption_lines = f.readlines()[:5]  # Show first 5 lines
    
    for line in caption_lines:
        line = line.strip()
        if line:
            words_in_line = line.lower().split()
            phonemes_in_line = []
            for word in words_in_line:
                if word in word2phonemes:
                    phonemes_in_line.extend(word2phonemes[word])
                else:
                    phonemes_in_line.append("AH")  # Fallback
            print(f"  '{line}' -> {phonemes_in_line}")

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
    # Create temporary directories for lyrics-aligner
    temp_audio_dir = OUTDIR / "temp_audio"
    temp_lyrics_dir = OUTDIR / "temp_lyrics"
    ensure_dir(temp_audio_dir)
    ensure_dir(temp_lyrics_dir)
    
    # Copy audio file to temp directory with expected name
    audio_name = audio_path.stem
    temp_audio_file = temp_audio_dir / f"{audio_name}.wav"
    shutil.copy2(audio_path, temp_audio_file)
    
    # Copy lyrics file to temp directory with expected name
    temp_lyrics_file = temp_lyrics_dir / f"{audio_name}.txt"
    shutil.copy2(lyrics_path, temp_lyrics_file)
    
    # Resolve absolute input directories (since we chdir below)
    abs_audio_dir = str(temp_audio_dir.resolve())
    abs_lyrics_dir = str(temp_lyrics_dir.resolve())
    print(f"[DEBUG] Aligner inputs:\n  audio_dir={abs_audio_dir}\n  lyrics_dir={abs_lyrics_dir}")
    
    # Run lyrics-aligner with directory paths
    cmd = ["python", "align.py", abs_audio_dir, abs_lyrics_dir, "--lyrics-format", "w", "--onsets", "p", "--dataset-name", "youtube"]
    
    # Change to lyrics-aligner directory to run the script
    original_cwd = Path.cwd()
    try:
        os.chdir("lyrics-aligner")
        run(cmd)
    finally:
        os.chdir(original_cwd)
    
    # Convert the output text file to JSON format
    output_txt = Path("lyrics-aligner/outputs/youtube/phoneme_onsets") / f"{audio_name}.txt"
    if output_txt.exists():
        convert_phoneme_txt_to_json(output_txt, out_json)
    else:
        raise FileNotFoundError(f"Expected output file {output_txt} not found")
    
    # Clean up temp directories
    shutil.rmtree(temp_audio_dir)
    shutil.rmtree(temp_lyrics_dir)

def convert_phoneme_txt_to_json(txt_path: Path, json_path: Path):
    """Convert lyrics-aligner text output to JSON format"""
    result = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            phoneme = parts[0]
            start_time = float(parts[1])
            
            # Calculate end time (use next phoneme's start time, or add 0.1s if last)
            if i + 1 < len(lines):
                next_parts = lines[i + 1].strip().split('\t')
                if len(next_parts) >= 2:
                    end_time = float(next_parts[1])
                else:
                    end_time = start_time + 0.1
            else:
                end_time = start_time + 0.1
            
            result.append({
                "t0": start_time,
                "t1": end_time,
                "phoneme": phoneme
            })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

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
        start = float(item.get("start", 0.0))
        end   = float(item.get("end", start))
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

def main(youtube_url=None, max_duration=None):
    url = youtube_url or YOUTUBE_URL
    
    # Temporarily modify the global URL for this run
    global YOUTUBE_URL
    original_url = YOUTUBE_URL
    YOUTUBE_URL = url
    
    try:
        audio, subs = download_audio_and_captions(url, OUTDIR, LANGUAGE)

        lyrics_txt = OUTDIR / "captions.txt"
        captions_lines = captions_to_text(subs)
        # Clean captions text by removing apostrophes to handle contractions
        cleaned_captions = [line.replace("'", "") for line in captions_lines]
        lyrics_txt.write_text("\n".join(cleaned_captions), encoding="utf-8")
        print("[OK] Captions ->", lyrics_txt)

        # Set up lyrics-aligner after captions are created so we can extract words from them
        setup_lyrics_aligner()

        phonemes_json = OUTDIR / "phonemes.json"
        run_lyrics_aligner(audio, lyrics_txt, phonemes_json)
        print("[OK] Aligned phonemes ->", phonemes_json)

        visemes_json = OUTDIR / "visemes_resonite_min.json"
        to_minimal_visemes(phonemes_json, visemes_json)
        print("[DONE] Minimal visemes ->", visemes_json)
        
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
        
    finally:
        YOUTUBE_URL = original_url

# Modal function registration
if MODAL_AVAILABLE:
    @app.function(image=image, cpu=2, timeout=900)
    def extract_visemes(youtube_url: str, max_duration: int = 60):
        """Modal function to extract visemes from YouTube video"""
        print(f"=== EXTRACTING VISEMES: {youtube_url} (max {max_duration}s) ===")
        return main(youtube_url, max_duration)

if __name__ == "__main__":
    main()
