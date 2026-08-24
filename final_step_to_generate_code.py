from typing import TypedDict
from pathlib import Path
from datetime import datetime
import json
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from IPython.display import Image,display
from youtubetranscripter import get_transcript_from_url

OLLAMA_MODEL = "qwen2.5-coder:7b"
OUTPUT_PATH = Path("/home/satvir/Downloads/TestCreation")
STEPS_PATH = OUTPUT_PATH / "steps"
GRAPH_IMAGE_PATH = Path("/home/satvir/youtubetrascipter/langgraph.png")
where_to_save_transcript_markup="/home/satvir/youtubetrascipter/transcript_analysis.md"
OUTPUT_FILE = Path(where_to_save_transcript_markup)

llm = ChatOllama(model=OLLAMA_MODEL,temperature=0)

class State(BaseModel):
    transcript: str
    transcription_md_file_str:str

def analyze_transcript(state: State):
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert technical content analyst and senior python developer.
    Your task is to read a video transcript and convert it into
    a useful, structured Markdown technical document.

    Do NOT simply summarize the transcript.

    Extract the actual technical information and organize it.

    The Markdown should contain:

    # Title

    ## 1. Overview

    Explain what the video teaches.

    ## 2. Main Concepts

    List and explain the important concepts.

    ## 3. Step-by-Step Workflow

    Convert the process described in the transcript into
    a clear workflow.

    Use Mermaid diagrams when they help explain architecture
    or workflows.

    ## 4. Important Components

    Explain technologies, libraries, frameworks, agents,
    tools, functions, data structures, etc.

    ## 5. Architecture

    Create an ASCII or Mermaid architecture diagram when
    appropriate.

    ## 6. Code / Implementation Concepts

    Extract important functions, classes, tools, agents,
    parameters, and implementation ideas.

    If the transcript does not contain actual source code,
    do NOT invent complete code. Instead describe the
    implementation conceptually.

    ## 7. Patterns / Design Patterns

    If the transcript contains multiple patterns, number them
    and explain each one.

    For every pattern include:

    - Purpose
    - Input
    - Processing
    - Output
    - Components
    - Workflow
    - When to use it

    ## 8. Examples

    Extract the examples actually mentioned in the transcript.

    ## 9. Key Takeaways

    Summarize the most important technical lessons.

    ## 10. Implementation Roadmap

    Give a practical order in which someone could implement
    the concepts from the transcript.

    IMPORTANT:

    - Use only information supported by the transcript.
    - Do not invent facts.
    - Remove duplicated speech.
    - Correct obvious speech-to-text errors when the meaning
      is clear.
    - Preserve important technical names.
    - Make the result useful for a developer.
    - Use Markdown headings.
    - Use bullet lists.
    - Use tables where useful.
    - Use code blocks for code.
    - Use Mermaid diagrams for workflows where useful.
    - Return ONLY Markdown.
            """
        ),
        (
            "human",
            """
    Here is the video transcript:

    ---------------- TRANSCRIPT ----------------

    {transcript}

    ---------------- END TRANSCRIPT -------------

    Convert this transcript into the requested technical
    Markdown document.
    """
        )
    ])


    # ============================================================
    # READ TRANSCRIPT
    # ============================================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Transcript file not found: {INPUT_FILE}"
        )

    transcript = INPUT_FILE.read_text(
        encoding="utf-8"
    )


    # ============================================================
    # CALL LLM
    # ============================================================

    print("==============================================")
    print("READING TRANSCRIPT")
    print("==============================================")

    print(f"Input file : {INPUT_FILE}")
    print(f"Characters : {len(transcript)}")


    messages = prompt.format_messages(
        transcript=transcript
    )


    print()
    print("==============================================")
    print("RUNNING LANGCHAIN LLM")
    print("==============================================")

    response = llm.invoke(messages)

    # ============================================================
    # EXTRACT RESULT
    # ============================================================

    markdown = response.content


    # Remove accidental markdown fences around the whole document
    if markdown.startswith("```markdown"):
        markdown = markdown[len("```markdown"):]

    if markdown.endswith("```"):
        markdown = markdown[:-3]

    markdown = markdown.strip()


    # ============================================================
    # SAVE MARKDOWN
    # ============================================================

    OUTPUT_FILE.write_text(
        markdown,
        encoding="utf-8"
    )


    # ============================================================
    # RESULT
    # ============================================================

    print()
    print("==============================================")
    print("MARKDOWN GENERATED")
    print("==============================================")

    print(f"Output file : {OUTPUT_FILE}")
    print(f"Characters  : {len(markdown)}")

    print()
    print("DONE")
    state['transcription_md_file_str']=markdown
    return state



















