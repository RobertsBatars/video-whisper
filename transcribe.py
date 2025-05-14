#!/usr/bin/env python3

import os
import re
import subprocess # For running ffmpeg
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
TEMP_AUDIO_FOLDER = "temp_extracted_audio_from_videos"
TRANSCRIPTION_FOLDER = "video_transcriptions"
SUPPORTED_VIDEO_EXTENSIONS = (
    '.mp4', '.mkv', '.avi', '.mov', '.webm', 
    '.flv', '.mpeg', '.mpg', '.wmv', '.ts', '.m2ts'
)
# --- End Configuration ---

def sanitize_filename(name):
    """
    Sanitizes a string to be used as a valid filename.
    Removes or replaces characters that are not allowed in filenames.
    """
    if name is None:
        name = "untitled"
    # Remove path and extension if present
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

    Args:
        video_file_path (str): The full path to the video file.

    Returns:
        str: Path to the extracted MP3 audio file, or None if extraction fails.
    """
    if not os.path.exists(TEMP_AUDIO_FOLDER):
        os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)

    base_filename = sanitize_filename(video_file_path)
    output_audio_path = os.path.join(TEMP_AUDIO_FOLDER, f"{base_filename}.mp3")

    # Check if audio file already exists (e.g., from a previous partial run)
    if os.path.exists(output_audio_path):
        print(f"Audio file {output_audio_path} already exists. Using existing file.")
        return output_audio_path

    # ffmpeg command:
    # -i <input_video>
    # -vn (disable video recording)
    # -acodec libmp3lame (audio codec MP3)
    # -ab 192k (audio bitrate 192kbps)
    # -ar 44100 (audio sampling rate 44.1kHz)
    # -y (overwrite output file if it exists - though we check above)
    # <output_audio.mp3>
    command = [
        'ffmpeg',
        '-i', video_file_path,
        '-vn',
        '-acodec', 'libmp3lame',
        '-ab', '192k',
        '-ar', '44100', # Whisper works well with 16kHz, but 44.1kHz is standard for mp3
        '-y',
        output_audio_path
    ]

    print(f"Extracting audio from: {video_file_path} to {output_audio_path}...")
    try:
        # Using subprocess.run
        # capture_output=True to get stdout/stderr, text=True to decode them as strings
        # check=True to raise CalledProcessError if ffmpeg returns a non-zero exit code
        result = subprocess.run(command, capture_output=True, text=True, check=False) # check=False to handle errors manually

        if result.returncode == 0:
            print("Audio extraction successful.")
            return output_audio_path
        else:
            print(f"Error during audio extraction for {video_file_path}.")
            print(f"ffmpeg stdout: {result.stdout}")
            print(f"ffmpeg stderr: {result.stderr}")
            # Attempt to clean up partially created file if extraction failed
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path)
            return None
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during audio extraction: {e}")
        # Attempt to clean up partially created file if extraction failed
        if os.path.exists(output_audio_path):
            try:
                os.remove(output_audio_path)
            except OSError:
                pass # Ignore if removal fails
        return None


def transcribe_audio_openai(audio_file_path, openai_api_key):
    """
    Transcribes the given audio file using OpenAI's Whisper API.
    (This function is similar to the one in the YouTube script)
    """
    if not openai_api_key:
        print("OpenAI API key not provided. Skipping transcription.")
        return None
    if not audio_file_path or not os.path.exists(audio_file_path):
        print(f"Audio file not found at {audio_file_path}. Skipping transcription.")
        return None

    client = OpenAI(api_key=openai_api_key)
    print(f"Transcribing audio file: {audio_file_path}...")

    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        print("Transcription successful.")
        return transcript
    except Exception as e:
        print(f"Error during OpenAI Whisper transcription for {audio_file_path}: {e}")
        return None

def save_transcription(text_content, original_video_filename_base):
    """
    Saves the transcribed text to a .txt file.
    The filename is derived from the original video filename.
    """
    if not os.path.exists(TRANSCRIPTION_FOLDER):
        os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
    
    # original_video_filename_base should already be sanitized
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
        print(f"\n--- Processing video {i+1}/{len(video_files_found)}: {os.path.basename(video_path)} ---")
        
        # Extract audio. Returns path_to_audio_file or None
        extracted_audio_path = extract_audio_from_video(video_path)
        
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            print(f"Audio extracted to: {extracted_audio_path}")
            
            transcription_text = transcribe_audio_openai(extracted_audio_path, current_openai_api_key)
            
            video_filename_base = sanitize_filename(os.path.basename(video_path)) # Get sanitized name for .txt file
            if transcription_text is not None:
                save_transcription(transcription_text, video_filename_base)
            else:
                print(f"Failed to transcribe audio for '{os.path.basename(video_path)}'.")
            
            # Clean up (delete) the temporary extracted audio file
            try:
                os.remove(extracted_audio_path)
                print(f"Cleaned up temporary audio file: {extracted_audio_path}")
            except OSError as e:
                print(f"Error deleting temporary audio file {extracted_audio_path}: {e}")
        else:
            print(f"Failed to extract audio for '{os.path.basename(video_path)}'. Skipping transcription.")

def main():
    """
    Main function to run the script.
    """
    load_dotenv() # Load environment variables from .env file

    print("Local Video File Transcriber")
    print("=" * 40)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("\nOpenAI API key not found in environment variables or .env file.")
        print("Please create a .env file with OPENAI_API_KEY='your_key' or enter it manually.")
        while not openai_api_key:
            openai_api_key = input("Enter your OpenAI API key (or press Enter to skip transcription for all videos): ").strip()
            if not openai_api_key:
                print("Transcription will be skipped as no API key was provided.")
                break # Exit loop, openai_api_key remains None or empty
    
    if openai_api_key:
         print("OpenAI API key loaded.")
    else:
        print("Proceeding without OpenAI API key. Transcription step will be skipped.")


    while True:
        input_folder_path = input("\nEnter the path to the folder containing your video files: ").strip()
        if os.path.isdir(input_folder_path):
            break
        else:
            print(f"Folder not found: '{input_folder_path}'. Please enter a valid folder path.")
            
    # Create necessary directories if they don't exist
    os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
        
    process_video_folder(input_folder_path, openai_api_key)
    
    print("\n--- All video processing complete. ---")
    # Optional: Clean up the temp audio folder if it's empty
    try:
        if os.path.exists(TEMP_AUDIO_FOLDER) and not os.listdir(TEMP_AUDIO_FOLDER):
            os.rmdir(TEMP_AUDIO_FOLDER)
            print(f"Cleaned up empty temporary audio folder: {TEMP_AUDIO_FOLDER}")
    except OSError as e:
        print(f"Could not remove temporary audio folder {TEMP_AUDIO_FOLDER}: {e}")


if __name__ == "__main__":
    main()
