#!/usr/bin/env python3

import os
import re
import subprocess # For running ffmpeg
import shutil # For removing chunk folder
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import make_chunks

# --- Configuration ---
TEMP_AUDIO_FOLDER = "temp_extracted_audio_from_videos" # For main extracted audio and chunk subfolders
TRANSCRIPTION_FOLDER = "video_transcriptions"
SUPPORTED_VIDEO_EXTENSIONS = (
    '.mp4', '.mkv', '.avi', '.mov', '.webm', 
    '.flv', '.mpeg', '.mpg', '.wmv', '.ts', '.m2ts'
)
# Whisper API limits and safety margins
WHISPER_API_FILE_SIZE_LIMIT = 25 * 1024 * 1024  # 25 MB
# Target size for chunks, slightly less than the API limit for safety
SAFE_CHUNK_SIZE_BYTES = 24 * 1024 * 1024       # 24 MB
# Assumed bitrate for MP3 files extracted by ffmpeg (must match ffmpeg settings)
MP3_BITRATE_KBPS = 192 
# --- End Configuration ---

def sanitize_filename(name):
    """
    Sanitizes a string to be used as a valid filename.
    Removes path and extension if present.
    """
    if name is None:
        name = "untitled"
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_ ')
    if not name:
        name = "untitled_video_file"
    return name

def extract_audio_from_video(video_file_path):
    """
    Extracts audio from a video file using ffmpeg and saves it as an MP3.
    """
    if not os.path.exists(TEMP_AUDIO_FOLDER):
        os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)

    base_filename = sanitize_filename(video_file_path)
    output_audio_path = os.path.join(TEMP_AUDIO_FOLDER, f"{base_filename}.mp3")

    if os.path.exists(output_audio_path):
        print(f"Audio file {output_audio_path} already exists. Using existing file.")
        return output_audio_path

    command = [
        'ffmpeg',
        '-i', video_file_path,
        '-vn', # Disable video
        '-acodec', 'libmp3lame', # Audio codec MP3
        '-ab', f'{MP3_BITRATE_KBPS}k', # Audio bitrate
        '-ar', '44100', # Audio sampling rate
        '-y', # Overwrite output file
        output_audio_path
    ]

    print(f"Extracting audio from: {video_file_path} to {output_audio_path}...")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("Audio extraction successful.")
            return output_audio_path
        else:
            print(f"Error during audio extraction for {video_file_path}.")
            print(f"ffmpeg stdout: {result.stdout}")
            print(f"ffmpeg stderr: {result.stderr}")
            if os.path.exists(output_audio_path): os.remove(output_audio_path)
            return None
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during audio extraction: {e}")
        if os.path.exists(output_audio_path):
            try: os.remove(output_audio_path)
            except OSError: pass
        return None

def _transcribe_single_audio_file(audio_chunk_path, openai_api_key):
    """
    Transcribes a single audio file (or chunk) using OpenAI Whisper.
    """
    client = OpenAI(api_key=openai_api_key)
    try:
        with open(audio_chunk_path, "rb") as audio_file:
            chunk_size_mb = os.path.getsize(audio_chunk_path) / (1024 * 1024)
            print(f"Sending chunk {os.path.basename(audio_chunk_path)} ({chunk_size_mb:.2f} MB) to Whisper API...")
            
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcript
    except Exception as e:
        print(f"Error during OpenAI Whisper transcription for chunk {audio_chunk_path}: {e}")
        return None

def _split_and_transcribe_audio(full_audio_path, openai_api_key):
    """
    Splits a large audio file into manageable chunks and transcribes each.
    """
    # Create a unique subfolder for this large file's chunks
    base_full_audio_name = sanitize_filename(full_audio_path)
    chunks_subfolder = os.path.join(TEMP_AUDIO_FOLDER, f"{base_full_audio_name}_chunks")
    os.makedirs(chunks_subfolder, exist_ok=True)

    print(f"Splitting large audio file: {full_audio_path}")
    try:
        audio = AudioSegment.from_mp3(full_audio_path)
    except Exception as e:
        print(f"Could not load audio file {full_audio_path} with pydub: {e}")
        if os.path.exists(chunks_subfolder): shutil.rmtree(chunks_subfolder)
        return None

    # Calculate chunk length in milliseconds to stay under SAFE_CHUNK_SIZE_BYTES
    bytes_per_second_audio = (MP3_BITRATE_KBPS * 1000) / 8.0
    if bytes_per_second_audio <=0:
        print("Error: Audio bytes per second is zero or negative. Check MP3_BITRATE_KBPS.")
        if os.path.exists(chunks_subfolder): shutil.rmtree(chunks_subfolder)
        return None
        
    # Max duration in seconds, with a slight safety margin (e.g., 95% of calculated)
    max_duration_seconds_per_chunk = (SAFE_CHUNK_SIZE_BYTES / bytes_per_second_audio) * 0.95
    chunk_length_ms = int(max_duration_seconds_per_chunk * 1000)

    if chunk_length_ms <= 1000: # Ensure chunk length is reasonable (e.g., at least 1 second)
        print(f"Error: Calculated chunk length ({chunk_length_ms}ms) is too small. "
              f"This might be due to a very high effective bitrate or small SAFE_CHUNK_SIZE_BYTES. "
              f"Minimum chunk duration should be >0. Current value: {max_duration_seconds_per_chunk:.2f}s.")
        if os.path.exists(chunks_subfolder): shutil.rmtree(chunks_subfolder)
        return None
        
    print(f"Targeting audio chunks of up to {chunk_length_ms / 1000.0:.2f} seconds.")
    
    audio_chunks = make_chunks(audio, chunk_length_ms)
    full_transcription_parts = []
    all_chunks_processed_successfully = True

    for i, audio_segment_chunk in enumerate(audio_chunks):
        chunk_filename = os.path.join(chunks_subfolder, f"chunk_{base_full_audio_name}_{i}.mp3")
        print(f"Exporting audio chunk {i+1}/{len(audio_chunks)}: {chunk_filename}")
        try:
            audio_segment_chunk.export(chunk_filename, format="mp3", bitrate=f"{MP3_BITRATE_KBPS}k")
        except Exception as e:
            print(f"Error exporting audio chunk {chunk_filename}: {e}")
            all_chunks_processed_successfully = False
            break # Stop processing if a chunk cannot be exported

        # Verify chunk size before uploading (optional, but good safeguard)
        if os.path.getsize(chunk_filename) >= WHISPER_API_FILE_SIZE_LIMIT:
            print(f"Critical Error: Exported chunk {chunk_filename} is too large ({os.path.getsize(chunk_filename)/(1024*1024):.2f}MB). "
                  "Aborting for this file.")
            try: os.remove(chunk_filename)
            except OSError: pass
            all_chunks_processed_successfully = False
            break
        
        transcription_part = _transcribe_single_audio_file(chunk_filename, openai_api_key)
        
        try:
            os.remove(chunk_filename) # Clean up individual chunk file
        except OSError as e:
            print(f"Warning: Could not delete audio chunk file {chunk_filename}: {e}")

        if transcription_part is None:
            print(f"Failed to transcribe audio chunk {i+1}/{len(audio_chunks)}. Aborting transcription for this video.")
            all_chunks_processed_successfully = False
            break
        
        full_transcription_parts.append(transcription_part)

    # Clean up the subfolder for chunks
    if os.path.exists(chunks_subfolder):
        try:
            shutil.rmtree(chunks_subfolder)
            print(f"Cleaned up chunk folder: {chunks_subfolder}")
        except OSError as e:
            print(f"Warning: Could not remove chunk folder {chunks_subfolder}: {e}")

    if not all_chunks_processed_successfully:
        print(f"Transcription for {full_audio_path} was incomplete due to errors.")
        return None

    return " ".join(full_transcription_parts) if full_transcription_parts else None


def transcribe_audio_manager(audio_file_path, openai_api_key):
    """
    Manages transcription, deciding whether to split the audio file or transcribe directly.
    """
    if not openai_api_key:
        print("OpenAI API key not provided. Skipping transcription.")
        return None
    if not audio_file_path or not os.path.exists(audio_file_path):
        print(f"Audio file not found at {audio_file_path}. Skipping transcription.")
        return None

    file_size = os.path.getsize(audio_file_path)
    # Using 98% of limit as threshold for direct transcription
    if file_size < (WHISPER_API_FILE_SIZE_LIMIT * 0.98):
        print(f"Audio file {os.path.basename(audio_file_path)} ({file_size/(1024*1024):.2f} MB) is within size limit. Transcribing directly.")
        return _transcribe_single_audio_file(audio_file_path, openai_api_key)
    else:
        print(f"Audio file {os.path.basename(audio_file_path)} ({file_size/(1024*1024):.2f} MB) exceeds threshold. Attempting to split.")
        return _split_and_transcribe_audio(audio_file_path, openai_api_key)


def save_transcription(text_content, original_video_filename_base):
    """
    Saves the transcribed text to a .txt file.
    """
    if not os.path.exists(TRANSCRIPTION_FOLDER):
        os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
    
    filepath = os.path.join(TRANSCRIPTION_FOLDER, f"{original_video_filename_base}.txt")
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"Transcription saved to: {filepath}")
    except Exception as e:
        print(f"Error saving transcription to {filepath}: {e}")

def process_video_folder(folder_path, current_openai_api_key):
    """
    Main processing loop: scans folder, extracts audio, transcribes, and saves.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    video_files_found = []
    for item in os.listdir(folder_path):
        full_item_path = os.path.join(folder_path, item)
        if os.path.isfile(full_item_path) and item.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
            video_files_found.append(full_item_path)

    if not video_files_found:
        print(f"No supported video files found in '{folder_path}'.")
        print(f"Supported extensions: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}")
        return

    print(f"Found {len(video_files_found)} video file(s) to process in '{folder_path}'.")

    for i, video_path in enumerate(video_files_found):
        video_basename = os.path.basename(video_path)
        print(f"\n--- Processing video {i+1}/{len(video_files_found)}: {video_basename} ---")
        
        extracted_audio_path = extract_audio_from_video(video_path)
        
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            print(f"Full audio extracted to: {extracted_audio_path}")
            
            # Use the new transcription manager that handles splitting
            transcription_text = transcribe_audio_manager(extracted_audio_path, current_openai_api_key)
            
            video_filename_base = sanitize_filename(video_basename)
            if transcription_text is not None: # Check for None (failure) vs empty string (actual empty transcription)
                save_transcription(transcription_text, video_filename_base)
            else:
                print(f"Failed to get transcription for '{video_basename}'.")
            
            # Clean up the main extracted audio file
            try:
                os.remove(extracted_audio_path)
                print(f"Cleaned up main extracted audio file: {extracted_audio_path}")
            except OSError as e:
                print(f"Error deleting main audio file {extracted_audio_path}: {e}")
        else:
            print(f"Failed to extract audio for '{video_basename}'. Skipping transcription.")

def main():
    load_dotenv()
    print("Local Video File Transcriber (with audio chunking for large files)")
    print("=" * 70)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\nOpenAI API key not found in environment variables or .env file.")
        print("Please create a .env file with OPENAI_API_KEY='your_key' or enter it manually.")
        while not openai_api_key: # Loop until a key is provided or user explicitly skips
            openai_api_key = input("Enter your OpenAI API key (or press Enter to skip all transcriptions): ").strip()
            if not openai_api_key:
                print("No API key entered. Transcription will be skipped for all videos.")
                break 
    
    if openai_api_key:
         print("OpenAI API key loaded.")
    # else: (message handled in loop)

    while True:
        input_folder_path = input("\nEnter the path to the folder containing your video files: ").strip()
        if os.path.isdir(input_folder_path):
            break
        else:
            print(f"Folder not found: '{input_folder_path}'. Please enter a valid folder path.")
            
    os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True) # Ensure main temp audio folder exists
    os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
        
    process_video_folder(input_folder_path, openai_api_key)
    
    print("\n--- All video processing complete. ---")
    # Optional: Clean up the main temp audio folder if it's empty (chunks subfolders should be gone)
    try:
        if os.path.exists(TEMP_AUDIO_FOLDER) and not os.listdir(TEMP_AUDIO_FOLDER):
            os.rmdir(TEMP_AUDIO_FOLDER)
            print(f"Cleaned up empty temporary audio folder: {TEMP_AUDIO_FOLDER}")
    except OSError as e:
        print(f"Could not remove main temporary audio folder {TEMP_AUDIO_FOLDER} (it might not be empty or other issues): {e}")

if __name__ == "__main__":
    main()
