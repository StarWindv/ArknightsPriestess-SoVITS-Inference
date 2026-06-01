#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS CLI Interface
A command-line interface for GPT-SoVITS text-to-speech synthesis.
"""

import argparse
import os
import sys
import glob
import re
from datetime import datetime

# Set up workspace paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(WORKSPACE_DIR, "lib")
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
REFS_DIR = os.path.join(WORKSPACE_DIR, "refs")

# Add lib directories to Python path
sys.path.insert(0, LIB_DIR)
sys.path.insert(0, os.path.join(LIB_DIR, "eres2net"))

# Set environment variables before importing torch
os.environ["version"] = "v2ProPlus"
os.environ["is_half"] = "True"


import numpy as np
import logging

# Suppress unnecessary logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)



# Language mapping from CLI args to internal codes
LANG_MAP = {
    "JP": "all_ja",
    "CN": "all_zh",
    "EN": "en",
    "KOR": "all_ko",
}

# Cut method mapping
CUT_METHOD = "cut5"  # 按标点符号切分


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPT-SoVITS CLI: Text-to-Speech Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interface.py "text to synthesize"
  python interface.py "Hello World" -l EN -s 32
  python interface.py "text" -l CN -tk 10 -tp 0.9
  python interface.py "text" -l KOR -o output.wav
  python interface.py -f input.txt
  python interface.py -d texts_dir
        """
    )
    
    # Required arguments (word is optional when -f or -d is used)
    parser.add_argument(
        "word",
        type=str,
        nargs="?",
        default=None,
        help="Target text to synthesize (optional if -f or -d is used)"
    )
    
    # File/directory input arguments
    parser.add_argument(
        "-f", "--words-file-path",
        type=str,
        default=None,
        help="Input file path for batch synthesis (saves to {output_dir}/{filename}.wav)"
    )
    
    parser.add_argument(
        "-d", "--words-file-dir",
        type=str,
        default=None,
        help="Input directory path for batch synthesis (saves to {output_dir}/{dir_name}/{filename}.wav)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-s", "--sampling",
        type=int,
        default=64,
        choices=[4, 8, 16, 32, 64, 128],
        help="Sampling steps (default: 64)"
    )
    
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        default=None,
        help="Output audio file path (default: workspace/output/{yy-MM-dd}/{word}.wav)"
    )
    
    parser.add_argument(
        "-tk", "--topk",
        type=int,
        default=5,
        help="Top-k sampling (default: 5, range: 1-100)"
    )
    
    parser.add_argument(
        "-tp", "--topp",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0, range: 0-1)"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0, range: 0-1)"
    )
    
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=20,
        help="Batch size for inference (default: 20, range: 1-200)"
    )
    
    parser.add_argument(
        "-p", "--paragraph-separation",
        type=float,
        default=0.3,
        help="Paragraph separation interval in seconds (default: 0.3, range: 0.01-1)"
    )
    
    parser.add_argument(
        "-rp", "--repetition-punishment",
        type=float,
        default=1.35,
        help="Repetition punishment (default: 1.35, range: 0-2)"
    )
    
    parser.add_argument(
        "-r", "--speech-rate",
        type=float,
        default=1.0,
        help="Speech rate (default: 1.0, range: 0.6-1.65)"
    )
    
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="JP",
        choices=["EN", "CN", "JP", "KOR"],
        help="Language of the text (default: JP)"
    )
    
    parser.add_argument(
        "-rl", "--refs-lang",
        type=str,
        default="JP",
        choices=["EN", "CN", "JP", "KOR"],
        help="Language of the reference audio (default: JP, must match the reference audio filename language)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=-1,
        help="Random seed (default: -1 for random)"
    )
    
    args = parser.parse_args()
    
    # Validate that at least one input source is provided
    if args.word is None and args.words_file_path is None and args.words_file_dir is None:
        parser.error("Must provide either 'word', -f/--words-file-path, or -d/--words-file-dir")
    
    # Validate arguments
    if not 1 <= args.topk <= 100:
        parser.error("topk must be between 1 and 100")
    if not 0 <= args.topp <= 1:
        parser.error("topp must be between 0 and 1")
    if not 0 <= args.temperature <= 1:
        parser.error("temperature must be between 0 and 1")
    if not 1 <= args.batch_size <= 200:
        parser.error("batch-size must be between 1 and 200")
    if not 0.01 <= args.paragraph_separation <= 1:
        parser.error("paragraph-separation must be between 0.01 and 1")
    if not 0 <= args.repetition_punishment <= 2:
        parser.error("repetition-punishment must be between 0 and 2")
    if not 0.6 <= args.speech_rate <= 1.65:
        parser.error("speech-rate must be between 0.6 and 1.65")
    
    return args


def get_ref_audio_info(prompt_lang="all_ja"):
    """
    Get reference audio information from the refs directory.
    
    Args:
        prompt_lang: Language code for the prompt text (default: "all_ja")
    
    Returns:
        tuple: (main_ref_path, sub_ref_paths, prompt_text, prompt_lang)
    """
    main_dir = os.path.join(REFS_DIR, "main")
    sub_dir = os.path.join(REFS_DIR, "sub")
    
    # Get main reference audio (should be exactly one file)
    main_files = glob.glob(os.path.join(main_dir, "*.wav"))
    if not main_files:
        raise FileNotFoundError(f"No .wav files found in {main_dir}")
    main_ref_path = main_files[0]
    
    # Get prompt text from filename (without extension)
    prompt_text = os.path.splitext(os.path.basename(main_ref_path))[0]
    
    # Get sub reference audio files
    sub_files = glob.glob(os.path.join(sub_dir, "*.wav"))
    
    return main_ref_path, sub_files, prompt_text, prompt_lang


def initialize_tts():
    """
    Initialize the TTS pipeline.
    
    Returns:
        TTS: Initialized TTS pipeline
    """
    config_path = os.path.join(WORKSPACE_DIR, "configs", "tts_infer.yaml")
    
    print(f"Loading TTS configuration from {config_path}...")
    tts_config = TTS_Config(config_path)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_config.device = device
    tts_config.is_half = device == "cuda"
    
    print(f"Device: {device}")
    print(f"Half precision: {tts_config.is_half}")
    print(f"Version: {tts_config.version}")
    
    print("Initializing TTS pipeline...")
    tts_pipeline = TTS(tts_config)
    
    return tts_pipeline


def get_output_path(args):
    """
    Get the output file path for direct text input.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        str: Output file path
    """
    if args.output_path:
        return args.output_path
    
    date_str = datetime.now().strftime("%y-%m-%d")
    clean_word = re.sub(r'[\\/:*?"<>|]', '_', args.word)
    if len(clean_word) > 50:
        clean_word = clean_word[:50]
    
    output_dir = os.path.join(WORKSPACE_DIR, "output", date_str)
    return os.path.join(output_dir, f"{clean_word}.wav")


def get_file_output_path(file_path, output_base_dir=None):
    """
    Get output path for file-based synthesis.
    
    Args:
        file_path: Path to the input text file
        output_base_dir: Base output directory (default: workspace/output/{yy-MM-dd})
    
    Returns:
        str: Output file path
    """
    date_str = datetime.now().strftime("%y-%m-%d")
    
    if output_base_dir is None:
        output_base_dir = os.path.join(WORKSPACE_DIR, "output", date_str)
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    clean_filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
    
    return os.path.join(output_base_dir, f"{clean_filename}.wav")


def get_dir_output_path(file_path, dir_path, output_base_dir=None):
    """
    Get output path for directory-based synthesis.
    
    Args:
        file_path: Path to the input text file
        dir_path: Path to the input directory
        output_base_dir: Base output directory (default: workspace/output/{yy-MM-dd})
    
    Returns:
        str: Output file path
    """
    date_str = datetime.now().strftime("%y-%m-%d")
    
    if output_base_dir is None:
        output_base_dir = os.path.join(WORKSPACE_DIR, "output", date_str)
    
    dir_name = os.path.basename(os.path.abspath(dir_path))
    filename = os.path.splitext(os.path.basename(file_path))[0]
    clean_dir_name = re.sub(r'[\\/:*?"<>|]', '_', dir_name)
    clean_filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
    
    return os.path.join(output_base_dir, clean_dir_name, f"{clean_filename}.wav")


def save_audio(audio, sr, output_path):
    """
    Save audio to file.
    
    Args:
        audio: Audio data as numpy array
        sr: Sampling rate
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.max() > 1.0 or audio_tensor.min() < -1.0:
            audio_tensor = audio_tensor / 32768.0
    else:
        audio_tensor = audio
    
    torchaudio.save(output_path, audio_tensor.unsqueeze(0), sr)
    print(f"Audio saved to: {output_path}")


def read_text_file(file_path):
    """
    Read text content from a file.
    If the first line is exactly a language code (EN/CN/JP/KOR), use that language.
    Otherwise, treat the entire content as text with no language override.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        tuple: (text_content, language_code_or_None)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        return "", None
    
    # Check if first line is exactly a language code
    first_line = lines[0].strip()
    if first_line in LANG_MAP:
        # First line is a language code, rest is text
        text = "".join(lines[1:]).strip()
        return text, first_line
    else:
        # First line is not a language code, entire content is text
        text = "".join(lines).strip()
        return text, None


def run_inference_with_text(tts_pipeline, text, args, main_ref_path, sub_ref_paths, prompt_text, prompt_lang="all_ja", language=None):
    """
    Run TTS inference with given text.
    
    Args:
        tts_pipeline: Initialized TTS pipeline
        text: Text to synthesize
        args: Parsed command line arguments
        main_ref_path: Path to main reference audio
        sub_ref_paths: List of paths to sub reference audio files
        prompt_text: Text of the reference audio
        prompt_lang: Language code for the prompt text (default: "all_ja")
        language: Language code override (e.g., "JP", "CN", "EN", "KOR") or None to use args.language
    
    Returns:
        tuple: (sampling_rate, audio_data)
    """
    # Use provided language or fall back to args.language
    lang = language if language is not None else args.language
    
    # Prepare inputs
    inputs = {
        "text": text,
        "text_lang": LANG_MAP[lang],
        "ref_audio_path": main_ref_path,
        "aux_ref_audio_paths": sub_ref_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": args.topk,
        "top_p": args.topp,
        "temperature": args.temperature,
        "text_split_method": CUT_METHOD,
        "batch_size": args.batch_size,
        "speed_factor": args.speech_rate,
        "split_bucket": True,
        "return_fragment": False,
        "fragment_interval": args.paragraph_separation,
        "seed": args.random_seed,
        "parallel_infer": True,
        "repetition_penalty": args.repetition_punishment,
        "sample_steps": args.sampling,
        "super_sampling": True,  # Enabled by default (will only work for v3)
    }
    
    print(f"\nSynthesizing text: {text}")
    print(f"Language: {lang}")
    print(f"Reference audio: {main_ref_path}")
    print(f"Sub references: {len(sub_ref_paths)} files")
    print(f"Sampling steps: {args.sampling}")
    print(f"Speech rate: {args.speech_rate}")
    print(f"Random seed: {'Random' if args.random_seed == -1 else args.random_seed}")
    
    # Run inference
    print("Starting inference...")
    sr, audio = None, None
    for sr, audio in tts_pipeline.run(inputs):
        print(f"Inference chunk received: sr={sr}, audio_shape={audio.shape if hasattr(audio, 'shape') else 'N/A'}")
    
    if sr is not None and audio is not None:
        print(f"Inference completed: sr={sr}, audio_length={len(audio)}")
    
    return sr, audio


def main():
    """Main entry point for the CLI."""
    # Parse command line arguments
    args = parse_args()
    logging.info("Loading TTS")
    from TTS_infer_pack.TTS import TTS, TTS_Config
    global TTS, TTS_Config
    from TTS_infer_pack.text_segmentation_method import get_method
    global get_method
    logging.info("Loading Torch")
    import torch # 延迟导入 | 为了 argparse 的效率考虑
    global torch
    logging.info("Loading TorchAudio")
    import torchaudio
    global torchaudio
    
    try:
        # Get reference audio information
        print("Loading reference audio information...")
        prompt_lang = LANG_MAP[args.refs_lang]
        main_ref_path, sub_ref_paths, prompt_text, prompt_lang = get_ref_audio_info(prompt_lang)
        
        # Initialize TTS pipeline
        tts_pipeline = initialize_tts()
        
        # Handle different input modes
        if args.words_file_dir is not None:
            # Directory mode: process all .txt files in directory
            dir_path = args.words_file_dir
            if not os.path.isdir(dir_path):
                print(f"Error: Directory not found: {dir_path}")
                sys.exit(1)
            
            txt_files = sorted(glob.glob(os.path.join(dir_path, "*.txt")))
            if not txt_files:
                print(f"Error: No .txt files found in {dir_path}")
                sys.exit(1)
            
            print(f"\nProcessing {len(txt_files)} files from directory: {dir_path}")
            
            for i, file_path in enumerate(txt_files, 1):
                print(f"\n[{i}/{len(txt_files)}] Processing: {os.path.basename(file_path)}")
                text, file_lang = read_text_file(file_path)
                if not text:
                    print(f"  Skipping empty file: {file_path}")
                    continue
                
                # Use file's language if specified, otherwise use args.language
                effective_lang = file_lang if file_lang is not None else args.language
                if file_lang is not None:
                    print(f"  File language: {file_lang} (overrides -l {args.language})")
                
                sr, audio = run_inference_with_text(
                    tts_pipeline, text, args, main_ref_path, sub_ref_paths, prompt_text, prompt_lang=prompt_lang, language=effective_lang
                )
                
                if audio is not None:
                    output_path = get_dir_output_path(file_path, dir_path, args.output_path)
                    save_audio(audio, sr, output_path)
                else:
                    print(f"  Failed to synthesize: {file_path}")
            
            print(f"\nBatch synthesis completed! Processed {len(txt_files)} files.")
            
        elif args.words_file_path is not None:
            # File mode: process single file
            file_path = args.words_file_path
            if not os.path.isfile(file_path):
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
            
            print(f"\nProcessing file: {file_path}")
            text, file_lang = read_text_file(file_path)
            if not text:
                print("Error: File is empty")
                sys.exit(1)
            
            # Use file's language if specified, otherwise use args.language
            effective_lang = file_lang if file_lang is not None else args.language
            if file_lang is not None:
                print(f"File language: {file_lang} (overrides -l {args.language})")
            
            sr, audio = run_inference_with_text(
                tts_pipeline, text, args, main_ref_path, sub_ref_paths, prompt_text, prompt_lang=prompt_lang, language=effective_lang
            )
            
            if audio is not None:
                output_path = get_file_output_path(file_path, args.output_path)
                save_audio(audio, sr, output_path)
                print("\nSynthesis completed successfully!")
            else:
                print("\nSynthesis failed: No audio output generated.")
                sys.exit(1)
                
        else:
            # Direct text mode
            sr, audio = run_inference_with_text(
                tts_pipeline, args.word, args, main_ref_path, sub_ref_paths, prompt_text, prompt_lang=prompt_lang
            )
            
            if audio is not None:
                output_path = get_output_path(args)
                save_audio(audio, sr, output_path)
                print("\nSynthesis completed successfully!")
            else:
                print("\nSynthesis failed: No audio output generated.")
                sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nSynthesis interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during synthesis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print(__file__)
    
    main()
