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
        """
    )
    
    # Required arguments
    parser.add_argument(
        "word",
        type=str,
        help="Target text to synthesize"
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
        "--random-seed",
        type=int,
        default=-1,
        help="Random seed (default: -1 for random)"
    )
    
    args = parser.parse_args()
    
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


def get_ref_audio_info():
    """
    Get reference audio information from the refs directory.
    
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
    
    return main_ref_path, sub_files, prompt_text, "all_ja"


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


def run_inference(tts_pipeline, args, main_ref_path, sub_ref_paths, prompt_text):
    """
    Run TTS inference.
    
    Args:
        tts_pipeline: Initialized TTS pipeline
        args: Parsed command line arguments
        main_ref_path: Path to main reference audio
        sub_ref_paths: List of paths to sub reference audio files
        prompt_text: Text of the reference audio
    
    Returns:
        tuple: (sampling_rate, audio_data)
    """
    # Prepare inputs
    inputs = {
        "text": args.word,
        "text_lang": LANG_MAP[args.language],
        "ref_audio_path": main_ref_path,
        "aux_ref_audio_paths": sub_ref_paths,
        "prompt_text": prompt_text,
        "prompt_lang": "all_ja",  # Reference language is always JP
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
    
    print(f"\nSynthesizing text: {args.word}")
    print(f"Language: {args.language}")
    print(f"Reference audio: {main_ref_path}")
    print(f"Sub references: {len(sub_ref_paths)} files")
    print(f"Sampling steps: {args.sampling}")
    print(f"Speech rate: {args.speech_rate}")
    print(f"Random seed: {'Random' if args.random_seed == -1 else args.random_seed}")
    
    # Run inference
    for sr, audio in tts_pipeline.run(inputs):
        return sr, audio
    
    return None, None


def save_audio(audio, sr, output_path):
    """
    Save audio to file.
    
    Args:
        audio: Audio data as numpy array
        sr: Sampling rate
        output_path: Output file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to torch tensor and save
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float()
        # Normalize if needed
        if audio_tensor.max() > 1.0 or audio_tensor.min() < -1.0:
            audio_tensor = audio_tensor / 32768.0
    else:
        audio_tensor = audio
    
    # Save using torchaudio
    torchaudio.save(output_path, audio_tensor.unsqueeze(0), sr)
    print(f"\nAudio saved to: {output_path}")


def get_output_path(args):
    """
    Get the output file path.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        str: Output file path
    """
    if args.output_path:
        return args.output_path
    
    # Generate default output path
    date_str = datetime.now().strftime("%y-%m-%d")
    # Clean the word for filename (remove special characters)
    clean_word = re.sub(r'[\\/:*?"<>|]', '_', args.word)
    # Truncate if too long
    if len(clean_word) > 50:
        clean_word = clean_word[:50]
    
    output_dir = os.path.join(WORKSPACE_DIR, "output", date_str)
    output_path = os.path.join(output_dir, f"{clean_word}.wav")
    
    return output_path


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
        main_ref_path, sub_ref_paths, prompt_text, prompt_lang = get_ref_audio_info()
        
        # Initialize TTS pipeline
        tts_pipeline = initialize_tts()
        
        # Run inference
        sr, audio = run_inference(tts_pipeline, args, main_ref_path, sub_ref_paths, prompt_text)
        
        if audio is not None:
            # Get output path
            output_path = get_output_path(args)
            
            # Save audio
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
