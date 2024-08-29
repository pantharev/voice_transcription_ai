import sys
import os
import argparse
import torch
from faster_whisper import WhisperModel

# Set environment variable to handle OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def transcribe_audio(file_path, model_size="base", use_gpu=True):
    # Check if CUDA is available
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    print(f"Using device: {device}")

    # Load the Whisper model
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Transcribe the audio file
    segments, info = model.transcribe(file_path)

    # Print the detected language
    print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")

    # Print the transcription
    print("\nTranscription:")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

def main():

    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description="Transcribe MP3 audio files using faster-whisper")
    parser.add_argument("file_path", help="Path to the MP3 file")
    parser.add_argument("--model_size", default="base", choices=["tiny", "base", "small", "medium", "large"], help="Size of the Whisper model to use")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    args = parser.parse_args()

    try:
        transcribe_audio(args.file_path, args.model_size, not args.cpu)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()