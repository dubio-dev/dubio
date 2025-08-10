import openai
import os
import argparse
import json

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

cmu_to_ipa = {
    "AA": "ɑ",   "AE": "æ",   "AH": "ʌ",   "AO": "ɔ",   "AW": "aʊ",
    "AY": "aɪ",  "B": "b",    "CH": "tʃ",  "D": "d",    "DH": "ð",
    "EH": "ɛ",   "ER": "ɝ",   "EY": "eɪ",  "F": "f",    "G": "ɡ",
    "HH": "h",   "IH": "ɪ",   "IY": "i",   "JH": "dʒ",  "K": "k",
    "L": "l",    "M": "m",    "N": "n",    "NG": "ŋ",   "OW": "oʊ",
    "OY": "ɔɪ",  "P": "p",    "R": "ɹ",    "S": "s",    "SH": "ʃ",
    "T": "t",    "TH": "θ",   "UH": "ʊ",   "UW": "u",   "V": "v",
    "W": "w",    "Y": "j",    "Z": "z",    "ZH": "ʒ"
}


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

        print(" ".join([v["viseme"] for v in visemes[:50]]))

        with open("./out/lyrics.json", "w") as fout:
            res = "[\n"
            for v in visemes:
                phoneme = cmu_to_ipa[v["phoneme"]]
                res += "  " + json.dumps({ "start": v["t0"], "end": v["t1"], "phoneme": phoneme }, ensure_ascii=False) + ",\n"
            res = res[:-2] + "\n]"
            fout.write(res)
        
        import shutil
        shutil.copy("./out/lyrics.json", "/home/willi/coding/ai/dubio/utils/song-gen/lyrics.json")
            
        
    except Exception as e:
        raise e
    
    return 0


if __name__ == "__main__":
    main()
