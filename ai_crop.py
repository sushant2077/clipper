import cv2
import mediapipe as mp
import json
import os
import subprocess
import numpy as np

# --- CONFIGURATION ---
TARGET_WIDTH = 608  # 9:16 ratio relative to 1080 height
TARGET_HEIGHT = 1080
SMOOTHING_FACTOR = 0.1  # Lower = smoother, Higher = faster tracking
DARK_LUXURY_ALPHA = 1.3 # Contrast
DARK_LUXURY_BETA = -20   # Brightness

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def convert_to_seconds(time_val):
    """Converts HH:MM:SS, MM:SS, or string numbers to float seconds"""
    if isinstance(time_val, (int, float)):
        return float(time_val)
    
    if isinstance(time_val, str):
        # Handle formats like 02:45 or 1:02:45
        parts = time_val.split(':')
        try:
            if len(parts) == 2: # MM:SS
                return float(int(parts[0]) * 60 + float(parts[1]))
            elif len(parts) == 3: # HH:MM:SS
                return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
            else:
                return float(time_val) # Try to convert "165" (string) to 165.0
        except ValueError:
            print(f"Warning: Could not parse timestamp {time_val}. Defaulting to 0.")
            return 0.0
    return 0.0

def apply_dark_luxury_cc(frame):
    """Applies Dark Luxury Color Grading: High Contrast, Low Brightness, Blue Boost"""
    frame = cv2.convertScaleAbs(frame, alpha=DARK_LUXURY_ALPHA, beta=DARK_LUXURY_BETA)
    b, g, r = cv2.split(frame)
    b = cv2.add(b, 20) 
    r = cv2.subtract(r, 10) 
    return cv2.merge([b, g, r])

def get_face_center(frame):
    """Detects face and returns the center X coordinate"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        center_x = int((bbox.xmin + bbox.width / 2) * frame.shape[1])
        return center_x
    return None

def process_clip(video_path, start_time, end_time, output_name):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"temp_{output_name}", fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))
    
    current_frame = start_frame
    last_center_x = width // 2
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        face_x = get_face_center(frame)
        if face_x is not None:
            last_center_x = int((last_center_x * (1 - SMOOTHING_FACTOR)) + (face_x * SMOOTHING_FACTOR))
        
        left = last_center_x - (TARGET_WIDTH // 2)
        if left < 0: left = 0
        if left + TARGET_WIDTH > width: left = width - TARGET_WIDTH
        
        clip_duration = end_time - start_time
        beat_duration = clip_duration / 4
        current_time = (current_frame - start_frame) / fps
        beat_index = int(current_time // beat_duration) % 2 
        
        zoom_factor = 1.1 if beat_index == 0 else 1.0
        
        crop = frame[0:height, left:left + TARGET_WIDTH]
        if zoom_factor != 1.0:
            h, w = crop.shape[:2]
            z_w = int(w / zoom_factor)
            z_h = int(h / zoom_factor)
            off_x = (w - z_w) // 2
            off_y = (h - z_h) // 2
            crop = crop[off_y:off_y+z_h, off_x:off_x+z_w]
            crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
        else:
            crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

        crop = apply_dark_luxury_cc(crop)
        out.write(crop)
        current_frame += 1

    cap.release()
    out.release()

    final_output = output_name
    cmd = [
        'ffmpeg', '-y', 
        '-i', f"temp_{output_name}", 
        '-i', video_path, 
        '-ss', str(start_time), 
        '-t', str(end_time - start_time), 
        '-map', '0:v:0', 
        '-map', '1:a:0', 
        '-c:v', 'libx264', 
        '-crf', '18', 
        '-c:a', 'aac', 
        '-shortest', 
        final_output
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    if os.path.exists(f"temp_{output_name}"):
        os.remove(f"temp_{output_name}")

def main():
    print("\n--- AI AUTOMATED BATCH CLIPPER (Fixed Timestamps) ---")
    video_path = input("Path to source video: ").strip('"')
    
    print("\nChoose mode: [1] Manual [2] JSON")
    mode = input("Mode: ")
    
    if mode == '1':
        start = convert_to_seconds(input("Start time (sec or MM:SS): "))
        end = convert_to_seconds(input("End time (sec or MM:SS): "))
        name = input("Output filename (e.g., clip1.mp4): ")
        process_clip(video_path, start, end, name)
    
    elif mode == '2':
        json_path = input("Path to JSON file: ").strip('"')
        with open(json_path, 'r') as f:
            clips = json.load(f)
        
        for i, clip in enumerate(clips):
            # CONVERT TIMESTAMPS TO SECONDS HERE
            start = convert_to_seconds(clip.get('start', 0))
            end = convert_to_seconds(clip.get('end', 10))
            name = clip.get('filename', f"clip_{i+1}.mp4")
            
            print(f"Processing {name} ({start}s -> {end}s)...")
            process_clip(video_path, start, end, name)
            print(f"Saved: {name}")

    print("\n--- ALL BATCH CLIPS COMPLETED ---")

if __name__ == "__main__":
    main()