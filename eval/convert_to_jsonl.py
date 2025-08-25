#!/usr/bin/env python3
"""
Convert reference.txt (space-separated token IDs) to JSONL format
Uses o200k_harmony tokenizer like in eval.py
"""

import hashlib
import json
import tiktoken

def convert_reference_to_jsonl(input_file="reference.txt", prompts_file="input.txt", output_file="converted_reference.jsonl"):
    """
    Convert reference.txt with token IDs to JSONL format matching ref_test.jsonl structure
    """
    # Get the o200k_harmony tokenizer
    enc = tiktoken.get_encoding("o200k_harmony")
    
    # Read prompts and token IDs
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    with open(input_file, "r", encoding="utf-8") as f:
        token_lines = [line.strip() for line in f if line.strip()]
    
    # Ensure we have matching number of prompts and completions
    if len(prompts) != len(token_lines):
        print(f"Warning: Number of prompts ({len(prompts)}) doesn't match number of token lines ({len(token_lines)})")
        # Use the minimum to avoid index errors
        num_entries = min(len(prompts), len(token_lines))
    else:
        num_entries = len(prompts)
    
    # Convert to JSONL format
    with open(output_file, "w", encoding="utf-8") as out_f:
        for i in range(num_entries):
            prompt = prompts[i]
            token_ids_str = token_lines[i]
            
            # Parse token IDs
            try:
                token_ids = [int(x) for x in token_ids_str.split()]
                # Decode tokens to text
                completion = enc.decode(token_ids)
                
                # Create entry matching ref_test.jsonl format
                entry = {
                    "id": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt": prompt,
                    "completion": completion
                }
                
                # Write as JSONL (one JSON object per line)
                json.dump(entry, out_f, ensure_ascii=False)
                out_f.write("\n")
                
            except Exception as e:
                print(f"Error processing line {i+1}: {e}")
                continue
    
    print(f"Successfully converted {num_entries} entries to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert reference.txt to JSONL format")
    parser.add_argument("-i", "--input", default="reference.txt", help="Input file with token IDs")
    parser.add_argument("-p", "--prompts", default="input.txt", help="Prompts file")
    parser.add_argument("-o", "--output", default="converted_reference.jsonl", help="Output JSONL file")
    
    args = parser.parse_args()
    
    convert_reference_to_jsonl(args.input, args.prompts, args.output)