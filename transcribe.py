import whisper
import os

def transcribe_video(video_path):
    """
    Transcribes a video file and saves the output to a text file 
    with timestamps for easy viral segment identification.
    """
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} not found.")
        return

    print("Loading AI Model (Whisper)... Please wait.")
    # 'base' is faster than 'small' or 'medium' and accurate enough for viral clips
    model = whisper.load_model("base")

    print(f"Transcribing {video_path}... This may take a few minutes.")
    # verbose=False keeps the terminal clean; timestamps=True is critical
    result = model.transcribe(video_path, verbose=False)

    # Prepare the output file
    output_file = "transcript.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("--- VIDEO TRANSCRIPT WITH TIMESTAMPS ---\n\n")
        
        for segment in result['segments']:
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            
            # Convert seconds to MM:SS format
            start_min = int(start // 60)
            start_sec = int(start % 60)
            end_min = int(end // 60)
            end_sec = int(end % 60)
            
            timestamp = f"[{start_min:02d}:{start_sec:02d} -> {end_min:02d}:{end_sec:02d}]"
            f.write(f"{timestamp} {text}\n")

    print(f"\nTranscription complete! Saved to: {output_file}")
    print("You can now copy the text from transcript.txt into GPT-4 to find viral timestamps.")

if __name__ == "__main__":
    print("\n--- AI Video Transcriber ---")
    video_file = input("Enter the path to your video file (e.g., WW3.mp4): ").replace('"', '')
    transcribe_video(video_file)
