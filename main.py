import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
from pytube import YouTube
import datetime
import json
import shutil
import glob

def download_youtube_video(url, output_path='videos/'):
    """Download a YouTube video to the specified path."""
    os.makedirs(output_path, exist_ok=True)
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        filename = stream.download(output_path)
        print(f"Downloaded: {filename}")
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def extract_landmarks(video_path):
    """Extract pose landmarks from a video."""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            frame_count += 1
            # Process every 3rd frame for efficiency
            if frame_count % 3 != 0:
                continue
                
            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frames_landmarks.append(frame_landmarks)
                
                # Draw landmarks for visualization
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(
                    annotated_image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Resize for display
                scale_percent = 50
                width = int(annotated_image.shape[1] * scale_percent / 100)
                height = int(annotated_image.shape[0] * scale_percent / 100)
                resized = cv2.resize(annotated_image, (width, height))
                
                cv2.imshow('Pose Detection', resized)
                
            if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {len(frames_landmarks)} frames from {video_path}")
    return frames_landmarks

def calculate_elbow_angle(landmarks, frame_idx):
    """Calculate the angle at the elbow joint."""
    mp_pose = mp.solutions.pose
    
    # Get shoulder, elbow, and wrist coordinates
    shoulder = landmarks[frame_idx][mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    elbow = landmarks[frame_idx][mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    wrist = landmarks[frame_idx][mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    # Calculate vectors
    shoulder_to_elbow = np.array([elbow['x'] - shoulder['x'], elbow['y'] - shoulder['y']])
    elbow_to_wrist = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
    
    # Calculate angle using dot product
    cosine_angle = np.dot(shoulder_to_elbow, elbow_to_wrist) / (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_knee_angle(landmarks, frame_idx):
    """Calculate the angle at the knee joint."""
    mp_pose = mp.solutions.pose
    
    # Get hip, knee, and ankle coordinates
    hip = landmarks[frame_idx][mp_pose.PoseLandmark.RIGHT_HIP.value]
    knee = landmarks[frame_idx][mp_pose.PoseLandmark.RIGHT_KNEE.value]
    ankle = landmarks[frame_idx][mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    
    # Calculate vectors
    hip_to_knee = np.array([knee['x'] - hip['x'], knee['y'] - hip['y']])
    knee_to_ankle = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
    
    # Calculate angle
    cosine_angle = np.dot(hip_to_knee, knee_to_ankle) / (np.linalg.norm(hip_to_knee) * np.linalg.norm(knee_to_ankle))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def extract_shooting_metrics(landmarks_sequence):
    """Extract basketball shooting metrics from landmark sequence."""
    mp_pose = mp.solutions.pose
    
    metrics = {
        'elbow_angles': [],
        'knee_angles': [],
        'release_height': [],
        'shoulder_alignment': []
    }
    
    for frame_idx in range(len(landmarks_sequence)):
        # Only process frames with good visibility
        try:
            # Calculate metrics
            metrics['elbow_angles'].append(calculate_elbow_angle(landmarks_sequence, frame_idx))
            metrics['knee_angles'].append(calculate_knee_angle(landmarks_sequence, frame_idx))
            
            # Release height (wrist y-coordinate at release frame)
            wrist = landmarks_sequence[frame_idx][mp_pose.PoseLandmark.RIGHT_WRIST.value]
            metrics['release_height'].append(wrist['y'])
            
            # Shoulder alignment (horizontal distance between shoulders)
            left_shoulder = landmarks_sequence[frame_idx][mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks_sequence[frame_idx][mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            metrics['shoulder_alignment'].append(abs(left_shoulder['y'] - right_shoulder['y']))
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue
            
    return metrics

def analyze_shooting_form(normal_metrics, fatigued_metrics):
    """Analyze differences between normal and fatigued shooting forms."""
    results = {}
    
    # Calculate statistics for normal shooting form
    normal_stats = {
        'elbow_angle_mean': np.mean(normal_metrics['elbow_angles']),
        'elbow_angle_std': np.std(normal_metrics['elbow_angles']),
        'knee_angle_mean': np.mean(normal_metrics['knee_angles']),
        'knee_angle_std': np.std(normal_metrics['knee_angles']),
        'release_height_mean': np.mean(normal_metrics['release_height']),
        'release_height_std': np.std(normal_metrics['release_height']),
        'shoulder_alignment_mean': np.mean(normal_metrics['shoulder_alignment']),
        'shoulder_alignment_std': np.std(normal_metrics['shoulder_alignment'])
    }
    
    # Calculate statistics for fatigued shooting form
    fatigued_stats = {
        'elbow_angle_mean': np.mean(fatigued_metrics['elbow_angles']),
        'elbow_angle_std': np.std(fatigued_metrics['elbow_angles']),
        'knee_angle_mean': np.mean(fatigued_metrics['knee_angles']),
        'knee_angle_std': np.std(fatigued_metrics['knee_angles']),
        'release_height_mean': np.mean(fatigued_metrics['release_height']),
        'release_height_std': np.std(fatigued_metrics['release_height']),
        'shoulder_alignment_mean': np.mean(fatigued_metrics['shoulder_alignment']),
        'shoulder_alignment_std': np.std(fatigued_metrics['shoulder_alignment'])
    }
    
    # Calculate Z-scores to detect deviations
    z_scores = {
        'elbow_angle': (fatigued_stats['elbow_angle_mean'] - normal_stats['elbow_angle_mean']) / normal_stats['elbow_angle_std'],
        'knee_angle': (fatigued_stats['knee_angle_mean'] - normal_stats['knee_angle_mean']) / normal_stats['knee_angle_std'],
        'release_height': (fatigued_stats['release_height_mean'] - normal_stats['release_height_mean']) / normal_stats['release_height_std'],
        'shoulder_alignment': (fatigued_stats['shoulder_alignment_mean'] - normal_stats['shoulder_alignment_mean']) / normal_stats['shoulder_alignment_std']
    }
    
    # Detect significant deviations (Z-score > 1.96 indicates 95% confidence)
    deviations = {}
    for metric, z_score in z_scores.items():
        if abs(z_score) > 1.96:
            normal_value = normal_stats[f'{metric}_mean']
            fatigued_value = fatigued_stats[f'{metric}_mean']
            deviations[metric] = {
                'z_score': z_score,
                'normal_value': normal_value,
                'fatigued_value': fatigued_value,
                'percent_change': ((fatigued_value - normal_value) / normal_value) * 100
            }
    
    return {
        'normal_stats': normal_stats,
        'fatigued_stats': fatigued_stats,
        'z_scores': z_scores,
        'significant_deviations': deviations
    }

def visualize_results(analysis_results, results_dir):
    """Visualize the analysis results with charts and tables."""
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Basketball Shooting Form Analysis: Normal vs. Fatigued', fontsize=16)
    
    # Define colors
    normal_color = '#3498db'  # Blue
    fatigued_color = '#e74c3c'  # Red
    
    # Plot elbow angle
    axes[0, 0].bar(['Normal', 'Fatigued'], 
                  [analysis_results['normal_stats']['elbow_angle_mean'], 
                   analysis_results['fatigued_stats']['elbow_angle_mean']],
                  color=[normal_color, fatigued_color])
    axes[0, 0].set_title('Elbow Angle (degrees)')
    axes[0, 0].set_ylabel('Angle (degrees)')
    
    # Add error bars
    axes[0, 0].errorbar(
        x=['Normal', 'Fatigued'],
        y=[analysis_results['normal_stats']['elbow_angle_mean'], 
           analysis_results['fatigued_stats']['elbow_angle_mean']],
        yerr=[analysis_results['normal_stats']['elbow_angle_std'], 
              analysis_results['fatigued_stats']['elbow_angle_std']],
        fmt='none', ecolor='black', capsize=5
    )
    
    # Plot knee angle
    axes[0, 1].bar(['Normal', 'Fatigued'], 
                  [analysis_results['normal_stats']['knee_angle_mean'], 
                   analysis_results['fatigued_stats']['knee_angle_mean']],
                  color=[normal_color, fatigued_color])
    axes[0, 1].set_title('Knee Angle (degrees)')
    axes[0, 1].set_ylabel('Angle (degrees)')
    
    # Add error bars
    axes[0, 1].errorbar(
        x=['Normal', 'Fatigued'],
        y=[analysis_results['normal_stats']['knee_angle_mean'], 
           analysis_results['fatigued_stats']['knee_angle_mean']],
        yerr=[analysis_results['normal_stats']['knee_angle_std'], 
              analysis_results['fatigued_stats']['knee_angle_std']],
        fmt='none', ecolor='black', capsize=5
    )
    
    # Plot release height
    axes[1, 0].bar(['Normal', 'Fatigued'], 
                  [analysis_results['normal_stats']['release_height_mean'], 
                   analysis_results['fatigued_stats']['release_height_mean']],
                  color=[normal_color, fatigued_color])
    axes[1, 0].set_title('Release Height (normalized)')
    
    # Add error bars
    axes[1, 0].errorbar(
        x=['Normal', 'Fatigued'],
        y=[analysis_results['normal_stats']['release_height_mean'], 
           analysis_results['fatigued_stats']['release_height_mean']],
        yerr=[analysis_results['normal_stats']['release_height_std'], 
              analysis_results['fatigued_stats']['release_height_std']],
        fmt='none', ecolor='black', capsize=5
    )
    
    # Plot shoulder alignment
    axes[1, 1].bar(['Normal', 'Fatigued'], 
                  [analysis_results['normal_stats']['shoulder_alignment_mean'], 
                   analysis_results['fatigued_stats']['shoulder_alignment_mean']],
                  color=[normal_color, fatigued_color])
    axes[1, 1].set_title('Shoulder Alignment (normalized)')
    
    # Add error bars
    axes[1, 1].errorbar(
        x=['Normal', 'Fatigued'],
        y=[analysis_results['normal_stats']['shoulder_alignment_mean'], 
           analysis_results['fatigued_stats']['shoulder_alignment_mean']],
        yerr=[analysis_results['normal_stats']['shoulder_alignment_std'], 
              analysis_results['fatigued_stats']['shoulder_alignment_std']],
        fmt='none', ecolor='black', capsize=5
    )
    
    plt.tight_layout()
    chart_path = os.path.join(results_dir, 'shooting_form_analysis.png')
    plt.savefig(chart_path)
    plt.show()
    
    # Generate analysis report text
    report_text = "=== STATISTICAL ANALYSIS RESULTS ===\n"
    report_text += "\nZ-scores (values > 1.96 or < -1.96 indicate significant changes):\n"
    for metric, value in analysis_results['z_scores'].items():
        significance = "SIGNIFICANT" if abs(value) > 1.96 else "not significant"
        report_text += f"• {metric.replace('_', ' ').title()}: {value:.2f} ({significance})\n"
    
    report_text += "\nSignificant Deviations Detected:\n"
    if analysis_results['significant_deviations']:
        for metric, data in analysis_results['significant_deviations'].items():
            report_text += f"\n• {metric.replace('_', ' ').title()}:\n"
            report_text += f"  - Normal: {data['normal_value']:.2f}\n"
            report_text += f"  - Fatigued: {data['fatigued_value']:.2f}\n"
            report_text += f"  - Change: {data['percent_change']:.2f}%\n"
            report_text += f"  - Z-score: {data['z_score']:.2f}\n"
            
            # Provide interpretation
            direction = "increased" if data['percent_change'] > 0 else "decreased"
            report_text += f"  - Interpretation: {metric.replace('_', ' ').title()} {direction} by {abs(data['percent_change']):.1f}% when fatigued\n"
            
            # Risk assessment
            if metric == 'knee_angle' and data['percent_change'] < -10:
                report_text += "    WARNING: Significant reduction in knee angle when fatigued may indicate increased landing stress\n"
            elif metric == 'elbow_angle' and abs(data['percent_change']) > 15:
                report_text += "    WARNING: Large change in elbow angle may affect shooting accuracy and increase injury risk\n"
            elif metric == 'shoulder_alignment' and data['percent_change'] > 20:
                report_text += "    WARNING: Increased shoulder misalignment may indicate compensatory mechanics\n"
    else:
        report_text += "No significant deviations detected\n"
    
    # Write report to file
    report_path = os.path.join(results_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    # Also save raw data as CSV and JSON
    normal_df = pd.DataFrame(analysis_results['normal_stats'], index=[0])
    fatigued_df = pd.DataFrame(analysis_results['fatigued_stats'], index=[0])
    
    # Save as CSV
    normal_df.to_csv(os.path.join(results_dir, 'normal_stats.csv'))
    fatigued_df.to_csv(os.path.join(results_dir, 'fatigued_stats.csv'))
    
    # Save complete analysis results as JSON
    with open(os.path.join(results_dir, 'full_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    # Print to console as well
    print(report_text)

def main():
    # Define video URLs
    print("Basketball Movement Analysis Proof of Concept")
    print("============================================")
    
    # Analysis data containers
    normal_metrics = None
    fatigued_metrics = None
    
    # Get user input on data source
    print("\nSelect data source:")
    print("1. Use example data")
    print("2. Use local videos (from 'videos' folder)")
    print("3. Download videos from YouTube")
    
    data_source = input("Enter choice (1-3): ")
    
    if data_source == "1":
        print("Using example data...")
        # Create synthetic data for demonstration
        normal_metrics = {
            'elbow_angles': [145.2, 146.8, 147.3, 144.9, 145.6, 146.1, 145.8, 146.2, 145.5, 146.9],
            'knee_angles': [132.5, 133.8, 131.9, 134.2, 133.1, 132.8, 134.0, 133.5, 132.2, 133.7],
            'release_height': [0.85, 0.86, 0.84, 0.85, 0.87, 0.86, 0.85, 0.86, 0.85, 0.84],
            'shoulder_alignment': [0.02, 0.03, 0.01, 0.02, 0.02, 0.03, 0.01, 0.02, 0.03, 0.02]
        }
        
        fatigued_metrics = {
            'elbow_angles': [139.8, 138.5, 140.2, 137.9, 139.1, 138.3, 139.5, 138.7, 137.6, 139.4],
            'knee_angles': [121.3, 119.8, 122.5, 120.7, 118.9, 121.5, 119.6, 122.1, 120.3, 118.5],
            'release_height': [0.82, 0.81, 0.80, 0.82, 0.79, 0.80, 0.81, 0.80, 0.79, 0.78],
            'shoulder_alignment': [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.08]
        }
        
        data_source_info = "Analysis performed using example data"
        
    elif data_source == "2":
        print("\nUsing local videos from 'videos' folder")
        
        # Check for videos in the videos directory
        videos_dir = "videos"
        if not os.path.exists(videos_dir):
            print(f"Error: 'videos' directory not found. Please create a 'videos' folder in the same directory as this script.")
            return
        
        # List all videos in the directory
        video_files = glob.glob(os.path.join(videos_dir, "*.mp4")) + glob.glob(os.path.join(videos_dir, "*.avi")) + glob.glob(os.path.join(videos_dir, "*.mov"))
        
        if not video_files:
            print("No video files found in the 'videos' directory. Please add .mp4, .avi, or .mov files.")
            return
            
        # Display available videos
        print("\nAvailable video files:")
        for i, video_path in enumerate(video_files):
            print(f"{i+1}. {os.path.basename(video_path)}")
            
        # Get user selection for normal form videos
        print("\nSelect videos for normal shooting form (comma-separated numbers, e.g., 1,3,5):")
        normal_selections = input("> ")
        normal_indices = [int(idx.strip())-1 for idx in normal_selections.split(",") if idx.strip().isdigit()]
        
        if not normal_indices or any(idx >= len(video_files) or idx < 0 for idx in normal_indices):
            print("Invalid selection for normal form videos.")
            return
            
        normal_videos = [video_files[idx] for idx in normal_indices]
        
        # Get user selection for fatigued form videos
        print("\nSelect videos for fatigued shooting form (comma-separated numbers, e.g., 2,4,6):")
        fatigued_selections = input("> ")
        fatigued_indices = [int(idx.strip())-1 for idx in fatigued_selections.split(",") if idx.strip().isdigit()]
        
        if not fatigued_indices or any(idx >= len(video_files) or idx < 0 for idx in fatigued_indices):
            print("Invalid selection for fatigued form videos.")
            return
            
        fatigued_videos = [video_files[idx] for idx in fatigued_indices]
        
        data_source_info = f"Analysis performed using local videos:\nNormal form: {', '.join([os.path.basename(v) for v in normal_videos])}\nFatigued form: {', '.join([os.path.basename(v) for v in fatigued_videos])}"
        
        # Process normal shooting videos
        print("\nProcessing normal form videos...")
        normal_metrics_list = []
        landmarks_data = {}
        
        for video_path in normal_videos:
            print(f"Extracting landmarks from {video_path}...")
            landmarks = extract_landmarks(video_path)
            if landmarks:
                print("Calculating metrics...")
                metrics = extract_shooting_metrics(landmarks)
                normal_metrics_list.append(metrics)
                landmarks_data[os.path.basename(video_path)] = landmarks
            else:
                print(f"Warning: Failed to extract landmarks from {video_path}")
        
        if not normal_metrics_list:
            print("Error: Could not extract any data from normal form videos.")
            return
            
        # Combine metrics from all normal videos
        normal_metrics = {
            'elbow_angles': [],
            'knee_angles': [],
            'release_height': [],
            'shoulder_alignment': []
        }
        
        for metrics in normal_metrics_list:
            for key in normal_metrics:
                normal_metrics[key].extend(metrics[key])
        
        # Process fatigued shooting videos
        print("\nProcessing fatigued form videos...")
        fatigued_metrics_list = []
        
        for video_path in fatigued_videos:
            print(f"Extracting landmarks from {video_path}...")
            landmarks = extract_landmarks(video_path)
            if landmarks:
                print("Calculating metrics...")
                metrics = extract_shooting_metrics(landmarks)
                fatigued_metrics_list.append(metrics)
                landmarks_data[os.path.basename(video_path)] = landmarks
            else:
                print(f"Warning: Failed to extract landmarks from {video_path}")
        
        if not fatigued_metrics_list:
            print("Error: Could not extract any data from fatigued form videos.")
            return
            
        # Combine metrics from all fatigued videos
        fatigued_metrics = {
            'elbow_angles': [],
            'knee_angles': [],
            'release_height': [],
            'shoulder_alignment': []
        }
        
        for metrics in fatigued_metrics_list:
            for key in fatigued_metrics:
                fatigued_metrics[key].extend(metrics[key])
    
    elif data_source == "3":
        print("\nEnter YouTube URLs of basketball shooting form videos")
        print("For proper analysis, videos should clearly show the complete shooting motion")
        
        normal_form_urls = []
        fatigued_form_urls = []
        
        # Get URLs for normal form videos
        num_normal = int(input("\nHow many normal form videos to analyze? "))
        for i in range(num_normal):
            url = input(f"Enter URL for normal form video {i+1}: ")
            normal_form_urls.append(url)
        
        # Get URLs for fatigued form videos
        num_fatigued = int(input("\nHow many fatigued form videos to analyze? "))
        for i in range(num_fatigued):
            url = input(f"Enter URL for fatigued form video {i+1}: ")
            fatigued_form_urls.append(url)
            
        data_source_info = f"Analysis performed using YouTube videos:\nNormal form URLs: {', '.join(normal_form_urls)}\nFatigued form URLs: {', '.join(fatigued_form_urls)}"
        
        # Create temporary video directories
        temp_videos_dir = "temp_videos"
        os.makedirs(temp_videos_dir, exist_ok=True)
        normal_videos_dir = os.path.join(temp_videos_dir, "normal")
        fatigued_videos_dir = os.path.join(temp_videos_dir, "fatigued")
        os.makedirs(normal_videos_dir, exist_ok=True)
        os.makedirs(fatigued_videos_dir, exist_ok=True)
        
        # Download and process videos
        print("\nDownloading and processing videos...")
        
        # Process normal shooting videos
        normal_videos = [download_youtube_video(url, normal_videos_dir + '/') for url in normal_form_urls]
        if not any(normal_videos):
            print("Error: Failed to download any normal form videos.")
            # Clean up temporary directories
            shutil.rmtree(temp_videos_dir, ignore_errors=True)
            return
            
        normal_metrics_list = []
        landmarks_data = {}
        
        for video_path in normal_videos:
            if video_path:  # Check if download was successful
                print(f"Extracting landmarks from {video_path}...")
                landmarks = extract_landmarks(video_path)
                if landmarks:
                    print("Calculating metrics...")
                    metrics = extract_shooting_metrics(landmarks)
                    normal_metrics_list.append(metrics)
                    landmarks_data[os.path.basename(video_path)] = landmarks
        
        if not normal_metrics_list:
            print("Error: Could not extract any data from normal form videos.")
            # Clean up temporary directories
            shutil.rmtree(temp_videos_dir, ignore_errors=True)
            return
            
        # Combine metrics from all normal videos
        normal_metrics = {
            'elbow_angles': [],
            'knee_angles': [],
            'release_height': [],
            'shoulder_alignment': []
        }
        
        for metrics in normal_metrics_list:
            for key in normal_metrics:
                normal_metrics[key].extend(metrics[key])
        
        # Process fatigued shooting videos
        fatigued_videos = [download_youtube_video(url, fatigued_videos_dir + '/') for url in fatigued_form_urls]
        if not any(fatigued_videos):
            print("Error: Failed to download any fatigued form videos.")
            # Clean up temporary directories
            shutil.rmtree(temp_videos_dir, ignore_errors=True)
            return
            
        fatigued_metrics_list = []
        
        for video_path in fatigued_videos:
            if video_path:  # Check if download was successful
                print(f"Extracting landmarks from {video_path}...")
                landmarks = extract_landmarks(video_path)
                if landmarks:
                    print("Calculating metrics...")
                    metrics = extract_shooting_metrics(landmarks)
                    fatigued_metrics_list.append(metrics)
                    landmarks_data[os.path.basename(video_path)] = landmarks
        
        if not fatigued_metrics_list:
            print("Error: Could not extract any data from fatigued form videos.")
            # Clean up temporary directories
            shutil.rmtree(temp_videos_dir, ignore_errors=True)
            return
            
        # Combine metrics from all fatigued videos
        fatigued_metrics = {
            'elbow_angles': [],
            'knee_angles': [],
            'release_height': [],
            'shoulder_alignment': []
        }
        
        for metrics in fatigued_metrics_list:
            for key in fatigued_metrics:
                fatigued_metrics[key].extend(metrics[key])
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Only create results directory if we have valid data
    if normal_metrics is None or fatigued_metrics is None:
        print("Error: No valid data for analysis. Exiting.")
        return
    
    # Create results directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_parent_dir = "results"
    os.makedirs(results_parent_dir, exist_ok=True)
    results_dir = os.path.join(results_parent_dir, f"analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create log file to record analysis session
    log_path = os.path.join(results_dir, "session_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write(f"Basketball Shooting Form Analysis\n")
        log_file.write(f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write(data_source_info + "\n\n")
    
    # Analyze and visualize the results
    print("\nAnalyzing shooting form differences...")
    with open(log_path, "a") as log_file:
        log_file.write("Performing analysis of shooting form differences\n")
    
    analysis_results = analyze_shooting_form(normal_metrics, fatigued_metrics)
    
    # Save raw analysis results
    with open(os.path.join(results_dir, 'analysis_results_raw.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    # Save any collected landmark data
    if data_source in ["2", "3"] and landmarks_data:
        landmarks_dir = os.path.join(results_dir, "landmarks")
        os.makedirs(landmarks_dir, exist_ok=True)
        for video_name, landmarks in landmarks_data.items():
            landmarks_file = os.path.join(landmarks_dir, f"landmarks_{video_name}.json")
            with open(landmarks_file, "w") as f:
                json.dump(landmarks, f)
    
    # Save the raw metrics data
    with open(os.path.join(results_dir, 'normal_metrics.json'), 'w') as f:
        json.dump(normal_metrics, f, indent=4)
    
    with open(os.path.join(results_dir, 'fatigued_metrics.json'), 'w') as f:
        json.dump(fatigued_metrics, f, indent=4)
    
    visualize_results(analysis_results, results_dir)
    
    # Create a README file for the results folder
    readme_path = os.path.join(results_dir, "README.txt")
    with open(readme_path, "w") as readme:
        readme.write(f"Basketball Shooting Form Analysis Results\n")
        readme.write(f"======================================\n")
        readme.write(f"Analysis performed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        readme.write(f"{data_source_info}\n\n")
        readme.write(f"This folder contains:\n")
        readme.write(f"- shooting_form_analysis.png: Visualization of key metrics comparing normal and fatigued shooting form\n")
        readme.write(f"- analysis_report.txt: Detailed textual analysis of shooting form differences\n")
        readme.write(f"- full_analysis.json: Complete analysis results in JSON format\n")
        readme.write(f"- normal_stats.csv/fatigued_stats.csv: Statistics for both conditions in CSV format\n")
        readme.write(f"- session_log.txt: Log of the analysis session\n")
        if data_source in ["2", "3"]:
            readme.write(f"- landmarks/: Raw pose landmark data extracted from each video\n")
    
    # Clean up temporary files if downloaded from YouTube
    if data_source == "3":
        shutil.rmtree(temp_videos_dir, ignore_errors=True)
    
    print(f"\nAnalysis complete! All results saved to: {results_dir}")
    print(f"- Chart: {os.path.join(results_dir, 'shooting_form_analysis.png')}")
    print(f"- Report: {os.path.join(results_dir, 'analysis_report.txt')}")
    print(f"- Data: Multiple JSON and CSV files with detailed metrics and analysis\n")
    print("This analysis demonstrates how simple video analysis can detect")
    print("biomechanical changes that may indicate increased injury risk when fatigued.")

if __name__ == "__main__":
    main()