from youtubetranscripter import get_transcript_from_url
from llm_transcripter_to_code import build_graph


if __name__ == "__main__":
    transcript=get_transcript_from_url(url="https://www.youtube.com/watch?v=FJHyCAi4GcY",where_to_save="transcript.txt")
    app = build_graph()
    result = run_graph(
        app,
        transcript,
        "/home/satvir/Downloads/TestCreation"
    )

    # --------------------------------------------------------
    # Print generated files
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("📁 GENERATED PROJECT")
    print("=" * 70)

    for file_path in result.get("files",{}):
        print(f"   📄 {file_path}")
