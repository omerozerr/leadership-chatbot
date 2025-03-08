import os
import math
from pytubefix import YouTube, Playlist
import whisper
import torch

def format_interval(group_index, interval):
    start_sec = group_index * interval
    end_sec = (group_index + 1) * interval
    # Format seconds into MM:SS
    start_min, start_sec = divmod(start_sec, 60)
    end_min, end_sec = divmod(end_sec, 60)
    return f"[{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}]"

def process_video(video_url, interval=30):
    yt = YouTube(video_url)
    video_id = video_url.split("v=")[-1]
    print(f"Processing video: {video_url}")

    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_filename = f"{video_id}.mp3"
    audio_stream.download(filename=audio_filename)

    # Try to use GPU (if available) and fall back to CPU if needed.
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("turbo").to(device)
        result = model.transcribe(audio_filename, language="tr")
    except Exception as e:
        print(f"Error on GPU for video {video_id}: {e}\nFalling back to CPU...")
        device = "cpu"
        model = whisper.load_model("turbo").to(device)
        result = model.transcribe(audio_filename, language="tr")

    segments = result.get("segments", [])
    transcript_by_interval = {}
    for seg in segments:
        group_index = int(seg["start"] // interval)
        transcript_by_interval.setdefault(group_index, []).append(seg["text"].strip())

    transcript_lines = [f"Video URL: {video_url}\n"]
    for group_index in sorted(transcript_by_interval.keys()):
        time_label = format_interval(group_index, interval)
        transcript_lines.append(time_label + ":")
        transcript_lines.append(" ".join(transcript_by_interval[group_index]))
        transcript_lines.append("")

    output_filename = f"{video_id}_transcript.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_lines))

    print(f"Transcript for video {video_id} saved to {output_filename}")

def process_playlist(playlist_url, interval=30):
    playlist = Playlist(playlist_url)
    print(f"Found {len(playlist.video_urls)} videos in the playlist.")
    for video_url in playlist.video_urls:
        try:
            process_video(video_url, interval=interval)
        except Exception as e:
            print(f"Error processing {video_url}: {e}")

# process a YouTube playlist.
playlist_url = "https://www.youtube.com/playlist?list=PLCi3Q_-uGtdlCsFXHLDDHBSLyq4BkQ6gZ"
process_playlist(playlist_url, interval=30)
