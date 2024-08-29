import sys
import os
import argparse
from faster_whisper import WhisperModel
import torch

# Set environment variable to handle OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def transcribe_audio(file_path, model_size="base", use_gpu=True):
    # Load the Whisper model
    if use_gpu and torch.cuda.is_available():
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    else:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Transcribe the audio file
    segments, info = model.transcribe(file_path)

    # Prepare the output
    output = f"Detected language: {info.language} with probability {info.language_probability:.2f}\n\n"
    output += "Transcription:\n"
    for segment in segments:
        output += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"

    return output

def save_transcription(transcription, input_file):
    # Generate output file name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_transcription.txt"

    # Save transcription to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription)

    print(f"Transcription saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3 audio files using faster-whisper")
    parser.add_argument("file_path", help="Path to the MP3 file")
    parser.add_argument("--model_size", default="base", choices=["tiny", "base", "small", "medium", "large"], help="Size of the Whisper model to use")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    try:
        transcription = transcribe_audio(args.file_path, args.model_size, not args.cpu)
        print(transcription)  # Print to console
        save_transcription(transcription, args.file_path)  # Save to file
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()