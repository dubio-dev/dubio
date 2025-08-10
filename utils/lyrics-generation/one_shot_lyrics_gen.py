import openai
import os
import argparse
import json

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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

        print(visemes[:10])
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0
