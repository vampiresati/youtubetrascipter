from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


def get_video_id(url: str) -> str:
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.strip("/")

    raise ValueError("Invalid YouTube URL")


def get_transcript(youtube_url: str):
    video_id = get_video_id(youtube_url)

    if not video_id:
        raise ValueError("Could not extract YouTube video ID")

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    text = "\n".join(
        snippet.text
        for snippet in transcript
    )

    return text

def get_transcript_from_url(url="https://www.youtube.com/watch?v=FJHyCAi4GcY",where_to_save="transcript.txt"):
    transcript=''
    try:
        transcript = get_transcript(url)
#         print("\n--- TRANSCRIPT ---\n")
#         print(transcript)

        with open(where_to_save, "w", encoding="utf-8") as file:
            file.write(transcript)
            print("\nTranscript saved to transcript.txt")
    except Exception as e:
        print("Error:", e)
    return transcript

if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    get_transcript_from_url(url="https://www.youtube.com/watch?v=FJHyCAi4GcY",where_to_save="transcript.txt")

