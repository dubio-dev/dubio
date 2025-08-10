"""
Robust educational lyrics generation with viseme matching.
Uses candidate generation + scoring approach for better lip-sync matching.
Works with actual viseme names from previous pipeline stage.
"""

import re
from typing import List, Dict, Tuple
import os

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

def chunk_for_educational_lyrics(visemes, target_duration=3.0):
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
                    chunks.append(create_educational_chunk(current_chunk, chunk_start, viseme['t0']))
                    current_chunk = []
                    chunk_start = None
            continue
        
        current_chunk.append(viseme)
        chunk_duration = viseme['t1'] - chunk_start
        
        if chunk_duration >= target_duration * 1.5:
            chunks.append(create_educational_chunk(current_chunk, chunk_start, viseme['t1']))
            current_chunk = []
            chunk_start = None
        elif chunk_duration >= target_duration and len(current_chunk) >= 3:
            chunks.append(create_educational_chunk(current_chunk, chunk_start, viseme['t1']))
            current_chunk = []
            chunk_start = None
    
    if current_chunk:
        chunks.append(create_educational_chunk(current_chunk, chunk_start, current_chunk[-1]['t1']))
    
    return chunks

def create_educational_chunk(visemes, t0, t1):
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

class RobustEducationalLyricGenerator:
    def __init__(self, api_key: str, educational_topic: str = "science"):
        self.api_key = api_key
        self.educational_topic = educational_topic
    
    def generate_lyrics_for_chunks(self, chunks: List[Dict], topic: str = None) -> str:
        topic = topic or self.educational_topic
        lyric_lines = []
        
        for chunk in chunks:
            lyrics = self._generate_best_matching_lyrics(chunk, topic)
            formatted = self._format_with_timestamps(lyrics, chunk)
            lyric_lines.append(formatted)
        
        return " ".join(lyric_lines)
    
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
Generate 5 different educational phrases about {topic}, each with exactly {target_syllables} syllables.

The phrases will be matched to these lip movements: {viseme_info}

Requirements:
- Each phrase must have exactly {target_syllables} syllables
- All about {topic}
- Educational and clear
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
            
            return cleaned[:5] if cleaned else self._fallback_candidates(chunk, topic)
            
        except Exception as e:
            print(f"Error generating candidates: {e}")
            return self._fallback_candidates(chunk, topic)
    
    def _fallback_candidates(self, chunk: Dict, topic: str) -> List[str]:
        """Generate simple fallback candidates."""
        syllables = chunk['estimated_syllables']
        
        if 'math' in topic.lower():
            base_phrases = ["add numbers up", "learn math today", "count to ten", "plus and minus", "solve the problem"]
        elif 'science' in topic.lower():
            base_phrases = ["study nature well", "plants grow tall", "water flows down", "learn science facts", "observe the world"]
        else:
            base_phrases = ["learn something new", "study very hard", "read good books", "think really clear", "explore the world"]
        
        return base_phrases[:3]
    
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
    
    def _format_with_timestamps(self, lyrics: str, chunk: Dict) -> str:
        words = lyrics.split()
        if not words:
            return ""
        
        total_duration = chunk['t1'] - chunk['t0']
        time_per_word = total_duration / len(words)
        
        formatted_words = []
        current_time = chunk['t0']
        
        for word in words:
            start_time = current_time
            end_time = current_time + time_per_word
            formatted_words.append(f"{word}[{start_time:.2f}:{end_time:.2f}]")
            current_time = end_time
        
        return " ".join(formatted_words)

def create_robust_educational_lipsync_lyrics(visemes: List[Dict], topic: str, openai_api_key: str) -> str:
    """
    Robust pipeline: visemes -> candidates -> scoring -> best match selection.
    
    Args:
        visemes: List of viseme events with 'viseme', 't0', 't1' (using actual viseme names like AA, PP, etc.)
        topic: Educational topic
        openai_api_key: OpenAI API key
        
    Returns:
        Formatted lyrics string with timestamps like: word[start:end]
    """
    chunks = chunk_for_educational_lyrics(visemes)
    generator = RobustEducationalLyricGenerator(openai_api_key, topic)
    lyrics_with_timestamps = generator.generate_lyrics_for_chunks(chunks)
    return lyrics_with_timestamps

# Example usage and testing
if __name__ == "__main__":
    # Sample with actual viseme names from your table
    sample_visemes = [
        {"viseme":"PP","t0":0.0,"t1":0.1},       # p, b, m sounds
        {"viseme":"AA","t0":0.1,"t1":0.3},       # A sound  
        {"viseme":"DD","t0":0.3,"t1":0.4},       # d, t sounds
        {"viseme":"Silence","t0":0.4,"t1":0.6},  # pause
        {"viseme":"IH","t0":0.6,"t1":0.8},       # ih sound
        {"viseme":"SS","t0":0.8,"t1":0.9},       # s, z sounds
    ]
    
    chunks = chunk_for_educational_lyrics(sample_visemes)
    print("Generated chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk['target_viseme_pattern']}")
        print(f"    Syllables: {chunk['estimated_syllables']}")
        print(f"    Duration: {chunk['duration']}s")
    
    print("\n" + "="*50)
    print("Testing candidate generation and scoring...")
    
    # Use the real API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, using fallback mode")
        generator = RobustEducationalLyricGenerator("fake-key", "math")
    else:
        print("✅ Using real OpenAI API key")
        generator = RobustEducationalLyricGenerator(api_key, "math")
    
    for chunk in chunks:
        candidates = generator._fallback_candidates(chunk, "math")
        print(f"\nCandidates for chunk: {chunk['target_viseme_pattern']}")
        for candidate in candidates:
            score = generator._score_candidate_simple(candidate, chunk)
            print(f"  '{candidate}' -> score: {score:.2f}")
    
    print("\n" + "="*50)
    print("Testing full pipeline...")
    
    # Test with the real API
    result = generator.generate_lyrics_for_chunks(chunks, "mathematics")
    print(f"Generated lyrics: {result}")
    
    # Also test the main function
    print("\n" + "="*50)
    print("Testing main function...")
    
    if api_key:
        full_result = create_robust_educational_lipsync_lyrics(
            visemes=sample_visemes,
            topic="basic math",
            openai_api_key=api_key
        )
        print(f"Full pipeline result: {full_result}")
    else:
        print("Skipping main function test - no API key")
