"""
DO NOT USE
This script takes in input of visemes and time stamps.
It will generate educational lyrics that match the lip movements/shapes.
It will then output words and timestamps to be sent as input into the JAM model.
"""

import re
from typing import List, Dict
import os

# Viseme to lip shape mapping - multiple sounds can share the same lip shape
VISEME_WORDS = {
    # Lip closure visemes (P, B, M sounds)
    'p': ['put', 'plus', 'paper', 'ople', 'map', 'stop', 'up'],
    'b': ['be', 'big', 'book', 'about', 'number', 'web', 'lab'],
    'm': ['me', 'my', 'make', 'math', 'time', 'sum', 'arm'],
    
    # Lip rounding visemes (vowels like O, U)
    'o': ['go', 'no', 'so', 'flow', 'grow', 'show', 'know'],
    'u': ['you', 'do', 'to', 'new', 'blue', 'true', 'two'],
    
    # Wide lip visemes (vowels like A, E)
    'a': ['add', 'and', 'at', 'cat', 'math', 'path', 'class'],
    'e': ['see', 'be', 'me', 'tree', 'green', 'mean', 'read'],
    
    # Neutral/mid visemes (vowels like I)
    'i': ['it', 'is', 'this', 'with', 'big', 'six', 'fit'],
    
    # Teeth visemes (F, V sounds)
    'f': ['for', 'from', 'if', 'five', 'fun', 'leaf', 'half'],
    'v': ['very', 'have', 'give', 'value', 'view', 'love'],
    
    # Tongue tip visemes (T, D, N, L sounds)
    't': ['to', 'time', 'ten', 'two', 'three', 'eight', 'sit'],
    'd': ['do', 'day', 'add', 'read', 'need', 'good', 'end'],
    'n': ['no', 'not', 'new', 'nine', 'on', 'learn', 'ten'],
    'l': ['let', 'like', 'learn', 'line', 'all', 'will', 'cool'],
    
    # Back tongue visemes (K, G sounds)  
    'k': ['can', 'come', 'make', 'take', 'like', 'back', 'book'],
    'g': ['go', 'good', 'big', 'again', 'group', 'grow', 'bag'],
    
    # Sibilant visemes (S, Z, SH sounds)
    's': ['see', 'so', 'six', 'sum', 'this', 'use', 'yes'],
    'z': ['is', 'his', 'has', 'zero', 'size', 'quiz', 'buzz'],
    'sh': ['she', 'show', 'shape', 'share', 'shine', 'fish'],
    
    # Silence/rest
    'sil': ['pause', 'rest', 'stop', 'wait', 'breath']
}

def chunk_for_educational_lyrics(visemes, target_duration=3.0):
    """
    Chunk visemes into natural phrase boundaries for educational lyric generation.
    
    Args:
        visemes: List of {"viseme": str, "t0": float, "t1": float}
        target_duration: Target phrase length in seconds
        
    Returns:
        List of chunks suitable for educational lyric generation
    """
    if not visemes:
        return []
    
    # Visemes that indicate natural breaks
    break_visemes = {'sil', 'pause', 'rest'}
    
    chunks = []
    current_chunk = []
    chunk_start = None
    
    for viseme in visemes:
        if chunk_start is None:
            chunk_start = viseme['t0']
        
        # Handle silence/pause visemes
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
        
        # Force chunk if getting too long
        if chunk_duration >= target_duration * 1.5:
            chunks.append(create_educational_chunk(current_chunk, chunk_start, viseme['t1']))
            current_chunk = []
            chunk_start = None
        
        # Natural chunk at good duration
        elif chunk_duration >= target_duration and len(current_chunk) >= 3:
            chunks.append(create_educational_chunk(current_chunk, chunk_start, viseme['t1']))
            current_chunk = []
            chunk_start = None
    
    if current_chunk:
        chunks.append(create_educational_chunk(current_chunk, chunk_start, current_chunk[-1]['t1']))
    
    return chunks

def create_educational_chunk(visemes, t0, t1):
    """Create a chunk with viseme-to-word matching."""
    if not visemes:
        return None
    
    viseme_pattern = ' '.join([v['viseme'] for v in visemes])
    estimated_words = max(1, min(5, len(visemes) // 2))  # Rough estimate
    
    # Find words that match the viseme sequence
    matching_words = find_viseme_matching_words(visemes)
    
    return {
        't0': round(t0, 2),
        't1': round(t1, 2),
        'duration': round(t1 - t0, 2),
        'visemes': visemes,
        'viseme_pattern': viseme_pattern,
        'estimated_words': estimated_words,
        'matching_words': matching_words,
        'prompt_context': f"Educational lyrics with {estimated_words} words, duration {t1-t0:.1f}s, lip shapes: {viseme_pattern}"
    }

def find_viseme_matching_words(visemes):
    """Find educational words that could match this viseme sequence."""
    matching_words = []
    
    # Educational word categories
    math_words = ['add', 'sum', 'plus', 'math', 'ten', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'one', 'zero']
    science_words = ['see', 'grow', 'show', 'know', 'flow', 'light', 'plant', 'water', 'earth', 'space', 'time']
    general_words = ['learn', 'read', 'study', 'think', 'know', 'understand', 'explore', 'discover']
    
    all_educational = math_words + science_words + general_words
    
    # For each viseme, find words that could start with that lip shape
    for viseme in visemes[:3]:  # Look at first few visemes
        viseme_key = viseme['viseme'].lower()
        if viseme_key in VISEME_WORDS:
            # Prefer educational words
            candidates = VISEME_WORDS[viseme_key]
            educational_matches = [w for w in candidates if w in all_educational]
            
            if educational_matches:
                matching_words.extend(educational_matches[:2])
            else:
                matching_words.extend(candidates[:2])
    
    return list(set(matching_words))  # Remove duplicates

class EducationalLyricGenerator:
    def __init__(self, api_key: str, educational_topic: str = "science"):
        self.api_key = api_key
        self.educational_topic = educational_topic
    
    def generate_lyrics_for_chunks(self, chunks: List[Dict], topic: str = None) -> str:
        topic = topic or self.educational_topic
        lyric_lines = []
        
        for chunk in chunks:
            lyrics = self._generate_chunk_lyrics(chunk, topic)
            formatted = self._format_with_timestamps(lyrics, chunk)
            lyric_lines.append(formatted)
        
        return " ".join(lyric_lines)
    
    def _generate_chunk_lyrics(self, chunk: Dict, topic: str) -> str:
        """Generate lyrics that match visemes for lip-sync."""
        
        # Try to use matching words directly
        if chunk['matching_words']:
            available_words = chunk['matching_words']
            target_count = chunk['estimated_words']
            
            # Filter for topic-relevant words
            if 'math' in topic.lower():
                topic_priority = ['add', 'sum', 'plus', 'ten', 'two', 'three', 'math', 'number']
            elif 'science' in topic.lower():
                topic_priority = ['see', 'grow', 'know', 'show', 'light', 'plant', 'water', 'space']
            else:
                topic_priority = ['learn', 'study', 'read', 'know', 'think', 'explore']
            
            # Prioritize topic words that are in our matching set
            prioritized = [w for w in topic_priority if w in available_words]
            remaining = [w for w in available_words if w not in prioritized]
            
            final_words = (prioritized + remaining)[:target_count]
            
            if len(final_words) >= target_count:
                return " ".join(final_words)
        
        # Fallback: Use OpenAI with viseme-aware prompt
        prompt = f"""
Create educational lyrics about {topic} that match these lip movements/visemes: {chunk['viseme_pattern']}

The words should:
1. Be educational about {topic}
2. Use {chunk['estimated_words']} words
3. Create lip movements that roughly match the viseme sequence
4. Form a coherent educational phrase

Suggested words that could work: {', '.join(chunk['matching_words']) if chunk['matching_words'] else 'any educational words'}

Focus on meaning and education rather than exact lip matching.

Return only the lyrics:
"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You create educational lyrics that roughly match lip movement patterns for educational videos."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=40,
                temperature=0.5
            )
            
            lyrics = response.choices[0].message.content.strip()
            return self._clean_lyrics(lyrics)
            
        except Exception as e:
            print(f"Error generating lyrics: {e}")
            # Better fallback using educational content
            if chunk['matching_words']:
                return " ".join(chunk['matching_words'][:chunk['estimated_words']])
            else:
                return "learn math"
    
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

def create_educational_lipsync_lyrics(visemes: List[Dict], topic: str, openai_api_key: str) -> str:
    """
    Complete pipeline: visemes -> chunks -> educational lyrics with timestamps.
    
    Args:
        visemes: List of viseme events with 'viseme', 't0', 't1'
        topic: Educational topic
        openai_api_key: OpenAI API key
        
    Returns:
        Formatted lyrics string with timestamps like: word[start:end]
    """
    chunks = chunk_for_educational_lyrics(visemes)
    generator = EducationalLyricGenerator(openai_api_key, topic)
    lyrics_with_timestamps = generator.generate_lyrics_for_chunks(chunks)
    return lyrics_with_timestamps

# Example usage
if __name__ == "__main__":
    # Sample viseme data (lip shapes over time)
    sample_visemes = [
        {"viseme":"p","t0":0.43,"t1":0.50},   # lip closure
        {"viseme":"a","t0":0.50,"t1":0.65},   # wide mouth
        {"viseme":"t","t0":0.65,"t1":0.75},   # tongue tip
        {"viseme":"sil","t0":0.75,"t1":1.00}, # pause
        {"viseme":"m","t0":1.00,"t1":1.10},   # lip closure
        {"viseme":"a","t0":1.10,"t1":1.25},   # wide mouth
        {"viseme":"t","t0":1.25,"t1":1.35},   # tongue tip
        {"viseme":"s","t0":1.35,"t1":1.50},   # sibilant
    ]
    
    chunks = chunk_for_educational_lyrics(sample_visemes)
    print("Viseme chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk['viseme_pattern']}")
        print(f"    Matching words: {chunk['matching_words']}")
        print(f"    Duration: {chunk['t0']}-{chunk['t1']}s")
    
    print("\n" + "="*50)
    
    # Test with educational content
    generator = EducationalLyricGenerator("fake-key", "math")
    result = generator.generate_lyrics_for_chunks(chunks, "math")
    print(f"Educational lyrics: {result}")
