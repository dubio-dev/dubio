"""
Robust educational lyrics generation with viseme matching.
Uses candidate generation + scoring approach for better lip-sync matching.
Works with actual viseme names from previous pipeline stage.
"""

import re
from typing import List, Dict
import os
import json
import argparse

# Viseme importance weights for scoring (based on visibility)
VISEME_WEIGHTS = {
    'PP': 3.0,      # p, b, m - very visible lip closure
    'FF': 2.5,      # f, v - teeth/lip contact, very visible
    'TH': 2.5,      # th - tongue/teeth, visible
    'DD': 2.0,      # d, t - tongue tip, moderately visible
    'KK': 1.5,      # g, k - back tongue, less visible
    'CH': 2.0,      # tS, dZ, S - tongue/teeth shapes
    'SS': 2.0,      # s, z - sibilants, visible tongue position
    'NN': 1.5,      # n, l - tongue tip/ridge
    'RR': 1.0,      # r - less visible tongue position
    'AA': 2.0,      # A - wide open mouth, very visible
    'E': 1.8,       # e - mid-open mouth
    'IH': 1.5,      # ih - narrow mouth opening
    'OH': 2.5,      # oh, o - rounded lips, very visible
    'OU': 2.5,      # ou, u - rounded lips, very visible
    'Silence': 0.5  # silence - least important for matching
}

def chunk_for_lyrics(visemes, target_duration=3.0):
    """Chunk visemes into natural phrase boundaries."""
    if not visemes:
        return []
    
    break_visemes = {'Silence', 'silence', 'sil', 'pause', 'rest'}
    chunks = []
    current_chunk = []
    chunk_start = None
    
    for viseme in visemes:
        if chunk_start is None:
            chunk_start = viseme['t0']
        
        # Handle silence visemes
        if viseme['viseme'] in break_visemes:
            if current_chunk:
                chunk_duration = viseme['t0'] - chunk_start
                if chunk_duration >= target_duration * 0.7:
                    chunks.append(create_chunk(current_chunk, chunk_start, viseme['t0']))
                    current_chunk = []
                    chunk_start = None
            continue
        
        current_chunk.append(viseme)
        chunk_duration = viseme['t1'] - chunk_start
        
        if chunk_duration >= target_duration * 1.5:
            chunks.append(create_chunk(current_chunk, chunk_start, viseme['t1']))
            current_chunk = []
            chunk_start = None
        elif chunk_duration >= target_duration and len(current_chunk) >= 3:
            chunks.append(create_chunk(current_chunk, chunk_start, viseme['t1']))
            current_chunk = []
            chunk_start = None
    
    if current_chunk:
        chunks.append(create_chunk(current_chunk, chunk_start, current_chunk[-1]['t1']))
    
    return chunks

def create_chunk(visemes, t0, t1):
    """Create chunk with target viseme pattern."""
    if not visemes:
        return None
    
    viseme_pattern = [v['viseme'] for v in visemes]
    durations = [v['t1'] - v['t0'] for v in visemes]
    
    # Estimate syllables based on vowel-like visemes
    vowel_visemes = {'AA', 'E', 'IH', 'OH', 'OU'}
    estimated_syllables = max(2, len([v for v in visemes if v['viseme'] in vowel_visemes]))
    if estimated_syllables == 0:  # fallback if no vowels detected
        estimated_syllables = max(2, len(visemes) // 3)
    
    return {
        't0': round(t0, 2),
        't1': round(t1, 2),
        'duration': round(t1 - t0, 2),
        'visemes': visemes,
        'target_viseme_pattern': viseme_pattern,
        'target_durations': durations,
        'estimated_syllables': estimated_syllables,
        'total_visemes': len(viseme_pattern)
    }

def compute_viseme_distance(candidate_visemes: List[str], target_visemes: List[str]) -> float:
    """Compute weighted edit distance between viseme sequences."""
    
    def substitution_cost(a, b):
        if a == b:
            return 0
        # Cost based on visual importance - more visible visemes have higher penalties
        weight_a = VISEME_WEIGHTS.get(a, 1.0)
        weight_b = VISEME_WEIGHTS.get(b, 1.0)
        return max(weight_a, weight_b)
    
    # Weighted edit distance algorithm
    m, n = len(candidate_visemes), len(target_visemes)
    if m == 0:
        return n
    if n == 0:
        return m
        
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = substitution_cost(candidate_visemes[i-1], target_visemes[j-1])
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion  
                dp[i-1][j-1] + cost  # substitution
            )
    
    return dp[m][n] / max(m, n, 1)  # Normalize by length

def compute_duration_penalty(candidate_text: str, target_duration: float) -> float:
    """Estimate duration mismatch penalty."""
    words = candidate_text.split()
    # Estimate ~0.4 seconds per word for natural speech
    estimated_duration = len(words) * 0.4
    
    duration_diff = abs(estimated_duration - target_duration)
    return duration_diff / target_duration if target_duration > 0 else 0

class RobustLyricGenerator:
    def __init__(self, api_key: str, topic: str = "science"):
        self.api_key = api_key
        self.topic = topic
    
    def generate_lyrics_for_chunks(self, chunks: List[Dict], topic: str = None) -> List[Dict]:
        topic = topic or self.topic
        all_word_entries = []
        
        for chunk in chunks:
            lyrics = self._generate_best_matching_lyrics(chunk, topic)
            word_entries = self._format_with_timestamps(lyrics, chunk)
            all_word_entries.extend(word_entries)
        
        return all_word_entries
    
    def _generate_best_matching_lyrics(self, chunk: Dict, topic: str) -> str:
        """Generate multiple candidates and pick the best viseme match."""
        
        # Step 1: Generate multiple candidate lyrics
        candidates = self._generate_candidate_lyrics(chunk, topic)
        
        # Step 2: Score each candidate (no phoneme conversion needed)
        best_candidate = None
        best_score = float('inf')
        
        for candidate in candidates:
            score = self._score_candidate_simple(candidate, chunk)
            if score < best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate if best_candidate else "learn today"
    
    def _generate_candidate_lyrics(self, chunk: Dict, topic: str) -> List[str]:
        """Generate multiple candidate lyrics with same syllable count."""
        
        target_syllables = chunk['estimated_syllables']
        viseme_info = ', '.join(chunk['target_viseme_pattern'])
        
        prompt = f"""
Generate 5 different phrases about {topic}, each with exactly {target_syllables} syllables.

The phrases will be matched to these lip movements: {viseme_info}

Requirements:
- Each phrase must have exactly {target_syllables} syllables
- All about {topic}
- Clear
- Different word choices for each
- Natural sounding phrases

Format: Return only the phrases, one per line.

Example for 4 syllables about math:
- Add two plus three
- Math is so fun  
- Learn to subtract
- Count up to ten
- Numbers are cool

Generate 5 phrases with {target_syllables} syllables about {topic}:
"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You generate educational phrases with exact syllable counts for lip-sync matching."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.8
            )
            
            content = response.choices[0].message.content.strip()
            candidates = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Clean up candidates
            cleaned = []
            for candidate in candidates:
                candidate = re.sub(r'^[-•*]\s*', '', candidate)
                candidate = self._clean_lyrics(candidate)
                if candidate:
                    cleaned.append(candidate)
            
            return cleaned[:5] if cleaned else []
            
        except Exception as e:
            print(f"Error generating candidates: {e}")
            return []
        
    def _score_candidate_simple(self, candidate: str, chunk: Dict) -> float:
        """Score candidate based on duration and length matching (simplified)."""
        
        # Duration penalty
        duration_penalty = compute_duration_penalty(candidate, chunk['duration'])
        
        # Word count penalty (prefer candidates with appropriate word count)
        words = candidate.split()
        target_word_count = max(2, chunk['estimated_syllables'] // 1.3)  # rough estimate
        word_count_penalty = abs(len(words) - target_word_count) / max(target_word_count, 1)
        
        # Syllable penalty (try to estimate syllables roughly)
        estimated_syllables = sum(max(1, len([c for c in word if c.lower() in 'aeiou'])) for word in words)
        syllable_penalty = abs(estimated_syllables - chunk['estimated_syllables']) / max(chunk['estimated_syllables'], 1)
        
        # Combine scores
        total_score = (
            2.0 * duration_penalty +     # Duration matching
            1.0 * word_count_penalty +   # Word count matching  
            1.5 * syllable_penalty       # Syllable count matching
        )
        
        return total_score
    
    def _clean_lyrics(self, lyrics: str) -> str:
        lyrics = re.sub(r'^["\']|["\']$', '', lyrics)
        lyrics = re.sub(r'[.!?]+$', '', lyrics)
        return lyrics.strip()
    
    def _format_with_timestamps(self, lyrics: str, chunk: Dict) -> List[Dict]:
        """Format lyrics as list of word dictionaries with timestamps."""
        words = lyrics.split()
        if not words:
            return []
        
        total_duration = chunk['t1'] - chunk['t0']
        time_per_word = total_duration / len(words)
        
        word_entries = []
        current_time = chunk['t0']
        
        for word in words:
            start_time = round(current_time, 3)
            end_time = round(current_time + time_per_word, 3)
            
            word_entries.append({
                "start": start_time,
                "end": end_time,
                "word": word
            })
            current_time = end_time
        
        return word_entries

def create_robust_lipsync_lyrics(visemes: List[Dict], topic: str, openai_api_key: str, output_file: str = None) -> List[Dict]:
    """
    Robust pipeline: visemes -> candidates -> scoring -> best match selection.
    
    Args:
        visemes: List of viseme events with 'viseme', 't0', 't1'
        topic: Educational topic
        openai_api_key: OpenAI API key
        output_file: Optional filename to save JSON output
        
    Returns:
        List of word dictionaries with start/end times
    """
    chunks = chunk_for_lyrics(visemes)
    generator = RobustLyricGenerator(openai_api_key, topic)
    word_entries = generator.generate_lyrics_for_chunks(chunks)
    
    # Save to JSON file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(word_entries, f, indent=2)
        print(f"Lyrics saved to {output_file}")
    
    return word_entries

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate educational lyrics from visemes")
    parser.add_argument("input_file", help="Path to input JSON file with visemes")
    parser.add_argument("output_file", help="Path to output JSON file for lyrics")
    parser.add_argument("--topic", default="basic math", help="Educational topic for lyrics")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No OpenAI API key provided. Using fallback mode.")
        api_key = "fake-key"
    
    try:
        # Load input visemes
        with open(args.input_file, 'r') as f:
            visemes = json.load(f)
        print(f"Loaded {len(visemes)} visemes from {args.input_file}")
        
        # Generate lyrics
        word_entries = create_robust_lipsync_lyrics(
            visemes=visemes,
            topic=args.topic,
            openai_api_key=api_key,
            output_file=args.output_file
        )
        
        print(f"Generated {len(word_entries)} words for topic: {args.topic}")
        print(f"Output saved to: {args.output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
