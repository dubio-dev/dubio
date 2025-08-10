"""
Robust educational lyrics generation with viseme matching.
Uses candidate generation + scoring approach for better lip-sync matching.
Works with actual viseme names from previous pipeline stage.
"""

import re
from typing import List, Dict, Tuple
import os
import json

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
    
    def generate_lyrics_for_chunks(self, chunks: List[Dict], topic: str = None) -> List[Dict]:
        topic = topic or self.educational_topic
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
            start_time = round(current_time, 1)
            end_time = round(current_time + time_per_word, 1)
            
            word_entries.append({
                "start": start_time,
                "end": end_time,
                "word": word
            })
            current_time = end_time
        
        return word_entries

def create_robust_educational_lipsync_lyrics(visemes: List[Dict], topic: str, openai_api_key: str, output_file: str = None) -> List[Dict]:
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
    chunks = chunk_for_educational_lyrics(visemes)
    generator = RobustEducationalLyricGenerator(openai_api_key, topic)
    word_entries = generator.generate_lyrics_for_chunks(chunks)
    
    # Save to JSON file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(word_entries, f, indent=2)
        print(f"Lyrics saved to {output_file}")
    
    return word_entries

# Replace the __main__ section with this more realistic test:

if __name__ == "__main__":
    # More realistic viseme sequence with varied timing (like actual speech)
    sample_visemes = [
        # "Plants need sunlight to grow well"
        {"viseme":"PP","t0":0.0,"t1":0.08},      # P in "Plants"
        {"viseme":"NN","t0":0.08,"t1":0.15},     # l in "Plants" 
        {"viseme":"AA","t0":0.15,"t1":0.28},     # a in "Plants"
        {"viseme":"NN","t0":0.28,"t1":0.35},     # n in "Plants"
        {"viseme":"DD","t0":0.35,"t1":0.42},     # t in "Plants"
        {"viseme":"SS","t0":0.42,"t1":0.48},     # s in "Plants"
        
        {"viseme":"Silence","t0":0.48,"t1":0.65}, # brief pause
        
        {"viseme":"NN","t0":0.65,"t1":0.72},     # n in "need"
        {"viseme":"E","t0":0.72,"t1":0.88},      # ee in "need"
        {"viseme":"DD","t0":0.88,"t1":0.95},     # d in "need"
        
        {"viseme":"Silence","t0":0.95,"t1":1.1}, # pause
        
        {"viseme":"SS","t0":1.1,"t1":1.18},      # s in "sunlight"
        {"viseme":"AA","t0":1.18,"t1":1.32},     # u in "sunlight"
        {"viseme":"NN","t0":1.32,"t1":1.38},     # n in "sunlight"
        {"viseme":"NN","t0":1.38,"t1":1.45},     # l in "sunlight"
        {"viseme":"AA","t0":1.45,"t1":1.58},     # i in "sunlight"
        {"viseme":"DD","t0":1.58,"t1":1.65},     # t in "sunlight"
        
        {"viseme":"Silence","t0":1.65,"t1":1.85}, # longer pause
        
        {"viseme":"DD","t0":1.85,"t1":1.92},     # t in "to"
        {"viseme":"OU","t0":1.92,"t1":2.08},     # o in "to"
        
        {"viseme":"Silence","t0":2.08,"t1":2.18}, # brief pause
        
        {"viseme":"KK","t0":2.18,"t1":2.25},     # g in "grow"
        {"viseme":"RR","t0":2.25,"t1":2.32},     # r in "grow"
        {"viseme":"OH","t0":2.32,"t1":2.48},     # ow in "grow"
        
        {"viseme":"Silence","t0":2.48,"t1":2.62}, # pause
        
        {"viseme":"OU","t0":2.62,"t1":2.75},     # w in "well"
        {"viseme":"E","t0":2.75,"t1":2.88},      # e in "well"
        {"viseme":"NN","t0":2.88,"t1":2.95},     # l in "well"
        
        {"viseme":"Silence","t0":2.95,"t1":3.2}, # final pause
        
        # Second phrase: "Water flows through roots"
        {"viseme":"OU","t0":3.2,"t1":3.28},      # w in "water"
        {"viseme":"AA","t0":3.28,"t1":3.42},     # a in "water"
        {"viseme":"DD","t0":3.42,"t1":3.48},     # t in "water"
        {"viseme":"RR","t0":3.48,"t1":3.58},     # r in "water"
        
        {"viseme":"Silence","t0":3.58,"t1":3.75}, # pause
        
        {"viseme":"FF","t0":3.75,"t1":3.82},     # f in "flows"
        {"viseme":"NN","t0":3.82,"t1":3.88},     # l in "flows"
        {"viseme":"OH","t0":3.88,"t1":4.05},     # ow in "flows"
        {"viseme":"SS","t0":4.05,"t1":4.15},     # s in "flows"
        
        {"viseme":"Silence","t0":4.15,"t1":4.32}, # pause
        
        {"viseme":"TH","t0":4.32,"t1":4.38},     # th in "through"
        {"viseme":"RR","t0":4.38,"t1":4.45},     # r in "through"
        {"viseme":"OU","t0":4.45,"t1":4.62},     # ough in "through"
        
        {"viseme":"Silence","t0":4.62,"t1":4.78}, # pause
        
        {"viseme":"RR","t0":4.78,"t1":4.85},     # r in "roots"
        {"viseme":"OU","t0":4.85,"t1":4.98},     # oo in "roots"
        {"viseme":"DD","t0":4.98,"t1":5.05},     # t in "roots"
        {"viseme":"SS","t0":5.05,"t1":5.15},     # s in "roots"
        
        {"viseme":"Silence","t0":5.15,"t1":5.5}, # final silence
    ]
    
    print("=== ROBUST LYRICS GENERATOR TEST ===")
    print(f"Total test duration: {sample_visemes[-1]['t1']:.1f} seconds")
    print(f"Total visemes: {len(sample_visemes)}")
    
    chunks = chunk_for_educational_lyrics(sample_visemes, target_duration=2.5)
    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        viseme_summary = ' -> '.join(chunk['target_viseme_pattern'][:8])  # Show first 8
        if len(chunk['target_viseme_pattern']) > 8:
            viseme_summary += "..."
        print(f"  Chunk {i+1}: {chunk['t0']:.2f}-{chunk['t1']:.2f}s ({chunk['duration']:.1f}s)")
        print(f"    Visemes: {viseme_summary}")
        print(f"    Est. syllables: {chunk['estimated_syllables']}")
    
    print("\n" + "="*60)
    print("TESTING CANDIDATE GENERATION...")
    
    # Use the real API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, using fallback mode")
        generator = RobustEducationalLyricGenerator("fake-key", "plant biology")
    else:
        print("✅ Using real OpenAI API key")
        generator = RobustEducationalLyricGenerator(api_key, "plant biology")
    
    # Test each chunk individually
    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i+1} ANALYSIS ---")
        print(f"Target: {chunk['target_viseme_pattern']}")
        print(f"Duration: {chunk['duration']:.1f}s, Syllables: {chunk['estimated_syllables']}")
        
        # Generate candidates
        if api_key and api_key != "fake-key":
            try:
                candidates = generator._generate_candidate_lyrics(chunk, "plant biology")
                print(f"AI Generated {len(candidates)} candidates:")
                for j, candidate in enumerate(candidates):
                    score = generator._score_candidate_simple(candidate, chunk)
                    print(f"  {j+1}. '{candidate}' (score: {score:.2f})")
            except Exception as e:
                print(f"AI generation failed: {e}")
                candidates = generator._fallback_candidates(chunk, "plant biology")
                print(f"Fallback candidates:")
                for j, candidate in enumerate(candidates):
                    score = generator._score_candidate_simple(candidate, chunk)
                    print(f"  {j+1}. '{candidate}' (score: {score:.2f})")
        else:
            candidates = generator._fallback_candidates(chunk, "plant biology")
            print(f"Fallback candidates:")
            for j, candidate in enumerate(candidates):
                score = generator._score_candidate_simple(candidate, chunk)
                print(f"  {j+1}. '{candidate}' (score: {score:.2f})")
        
        # Show best match
        best_lyrics = generator._generate_best_matching_lyrics(chunk, "plant biology")
        print(f"  → BEST: '{best_lyrics}'")
    
    print("\n" + "="*60)
    print("TESTING FULL PIPELINE...")
    
    # Test the complete pipeline
    topics_to_test = ["plant biology", "basic math", "water cycle"]
    
    for topic in topics_to_test:
        print(f"\n--- TOPIC: {topic.upper()} ---")
        if api_key and api_key != "fake-key":
            try:
                full_result = create_robust_educational_lipsync_lyrics(
                    visemes=sample_visemes,
                    topic=topic,
                    openai_api_key=api_key
                )
                print(f"Result: {full_result}")
                
                # Analyze the result
                words_with_times = re.findall(r'(\w+)\[([0-9.]+):([0-9.]+)\]', full_result)
                print(f"Generated {len(words_with_times)} words:")
                for word, start, end in words_with_times[:5]:  # Show first 5
                    print(f"  '{word}' at {start}-{end}s")
                if len(words_with_times) > 5:
                    print(f"  ... and {len(words_with_times) - 5} more")
                    
            except Exception as e:
                print(f"Full pipeline failed: {e}")
        else:
            print("Skipping AI generation - no API key")
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE!")

    print("\n" + "="*60)
    print("TESTING JSON OUTPUT...")
    
    # Test the complete pipeline with JSON output
    topics_to_test = ["plant biology", "basic math"]
    
    for i, topic in enumerate(topics_to_test):
        print(f"\n--- TOPIC: {topic.upper()} ---")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = "fake-key"
        
        try:
            # Generate lyrics and save to JSON
            output_filename = f"lyrics_{topic.replace(' ', '_')}.json"
            word_entries = create_robust_educational_lipsync_lyrics(
                visemes=sample_visemes,
                topic=topic,
                openai_api_key=api_key,
                output_file=output_filename
            )
            
            print(f"Generated {len(word_entries)} words")
            print("Sample output:")
            for j, entry in enumerate(word_entries[:5]):  # Show first 5
                print(f"  {j+1}. {entry}")
            if len(word_entries) > 5:
                print(f"  ... and {len(word_entries) - 5} more words")
            
            # Also print the JSON structure
            print(f"\nJSON structure preview:")
            print(json.dumps(word_entries[:3], indent=2))
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
    
    print(f"\n{'='*60}")
    print("JSON OUTPUT TEST COMPLETE!")
    print("Check the generated .json files in the current directory")
