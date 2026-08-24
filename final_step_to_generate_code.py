from pathlib import Path

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from youtubetranscripter import get_transcript_from_url



OLLAMA_MODEL = "qwen2.5-coder:7b"
OUTPUT_PATH = Path("/home/satvir/Downloads/MarkdownSatvir")
STEPS_PATH = OUTPUT_PATH / "steps"
INPUT_FILE = Path("/home/satvir/youtubetrascipter/transcript.txt")
OUTPUT_FILE = Path("/home/satvir/youtubetrascipter/transcript_analysis.md")
GRAPH_IMAGE_PATH = Path("/home/satvir/youtubetrascipter/langgraph.png")
OUTPUT_PATH.mkdir(parents=True,exist_ok=True)
STEPS_PATH.mkdir(parents=True,exist_ok=True)



llm = ChatOllama(model=OLLAMA_MODEL,temperature=0)


class State(BaseModel):
    video_url:str="https://www.youtube.com/watch?v=FJHyCAi4GcY"
    where_to_save_transcript:str="transcript.txt"
    transcript: str = ""
    transcription_md_file_str: str = ""
    python_code_transcript:str=""
    output_file: str = ""


prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",

            """
You are an expert technical content analyst and senior Python developer.

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
do NOT invent complete code.

Instead describe the implementation conceptually.

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

    ]
)


prompt2= ChatPromptTemplate.from_messages(
    [
        (
            "system",

            """
You are an expert technical senior Python developer read markdown down .
Your task is to read a video transcript and read transcript markdown and converting this coding markdown
IMPORTANT:
- Use only information supported by the transcript.
- use markup to generate code
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


---------------- MARKDOWN ------------------
{transcript_markdown}

_______________  END OF MARKDOWN _____________
Convert this transcript nto the requested technical
Markdown document.
"""
        )

    ]
)


# ============================================================
# NODE 1
# READ TRANSCRIPT
# ============================================================
def get_youtube_transcript(state:State):
    transcript=get_transcript_from_url(url=state.video_url,where_to_save=state.where_to_save_transcript)
    state.transcript = transcript
    return state

# ============================================================
# NODE 2
# ANALYZE TRANSCRIPT
# ============================================================

def analyze_transcript(state: State):
    messages = prompt.format_messages(transcript=state.transcript)
    response = llm.invoke(messages)
    markdown = response.content
    markdown = markdown.strip()
    if markdown.startswith("```markdown"):
        markdown = markdown[
            len("```markdown"):
        ]
    elif markdown.startswith("```"):
        markdown = markdown[
            len("```"):
        ]
    if markdown.endswith("```"):
        markdown = markdown[
            :-len("```")
        ]
    markdown = markdown.strip()
    state.transcription_md_file_str = markdown
    print(
        f"Generated Markdown characters: "
        f"{len(markdown)}"
    )

    return state

def save_markdown(state: State):
    print()
    print("=" * 60)
    print("STEP 3: SAVING MARKDOWN")
    print("=" * 60)

    OUTPUT_FILE.write_text(
        state.transcription_md_file_str,
        encoding="utf-8"
    )

    state.output_file = str(
        OUTPUT_FILE
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print(
        f"Characters: "
        f"{len(state.transcription_md_file_str)}"
    )

    return state

def save_markdown_python_code(state: State):

    OUTPUT_FILE.write_text(
        "output_python_code.md",
        encoding="utf-8"
    )
    state.output_file = str(
        OUTPUT_FILE
    )

    print(
        f"Characters: "
        f"{len(state.python_code_transcript)}"
    )

    return state


def python_code_md_file_generate(state:State):
    messages = prompt2.format_messages(transcript=state.transcript,transcript_markdown= state.transcription_md_file_str)
    response = llm.invoke(messages)
    markdown = response.content
        markdown = markdown.strip()
        if markdown.startswith("```markdown"):
            markdown = markdown[
                len("```markdown"):
            ]
        elif markdown.startswith("```"):
            markdown = markdown[
                len("```"):
            ]
        if markdown.endswith("```"):
            markdown = markdown[
                :-len("```")
            ]
        markdown = markdown.strip()
        state.python_code_transcript = markdown
        print(
            f"Generated Markdown characters: "
            f"{len(markdown)}"
        )
        return state


# ============================================================
# BUILD LANGGRAPH
# ============================================================
def build_graph_langchain():
    builder = StateGraph(State)
    builder.add_node("get_youtube_transcript",get_youtube_transcript)
    builder.add_node("analyze_transcript",analyze_transcript)
    builder.add_node("save_markdown",save_markdown)
    builder.add_node("save_markdown_python_code",save_markdown_python_code)
    builder.add_node("python_code_md_file_generate",python_code_md_file_generate)

    builder.add_edge(START,"get_youtube_transcript")
    builder.add_edge("get_youtube_transcript","analyze_transcript")
    builder.add_edge("analyze_transcript","save_markdown")
    builder.add_edge("save_markdown","python_code_md_file_generate")
    builder.add_edge("python_code_md_file_generate","save_markdown_python_code")
    builder.add_edge("save_markdown_python_code",END)

    graph = builder.compile()
    return graph

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("STARTING LANGGRAPH")
    print("=" * 60)

    initial_state = State(video_url="https://www.youtube.com/watch?v=FJHyCAi4GcY",where_to_save_transcript="transcript.txt")

    graph=build_graph_langchain()
    final_state = graph.invoke(initial_state)

    print()
    print("=" * 60)
    print("WORKFLOW COMPLETED")
    print("=" * 60)

    print(
        f"Markdown file: "
        f"{final_state['output_file']}"
    )

    print(
        f"Markdown characters: "
        f"{len(final_state['transcription_md_file_str'])}"
    )
    print('------python code markdown script -------')
    print(final_state['save_markdown_python_code'])
    print()
    print("DONE")
