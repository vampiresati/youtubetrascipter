from pathlib import Path

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from youtubetranscripter import get_transcript_from_url


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = "qwen2.5-coder:7b"

OUTPUT_PATH = Path(
    "/home/satvir/Downloads/MarkdownSatvir"
)

STEPS_PATH = OUTPUT_PATH / "steps"

OUTPUT_FILE = OUTPUT_PATH / "transcript_analysis.md"

PYTHON_OUTPUT_FILE = OUTPUT_PATH / "output_python_code.md"

GRAPH_IMAGE_PATH = OUTPUT_PATH / "langgraph.png"


OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

STEPS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OLLAMA
# ============================================================

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)


# ============================================================
# STATE
# ============================================================

class State(BaseModel):

    video_url: str = (
        "https://www.youtube.com/watch?v=FJHyCAi4GcY"
    )

    where_to_save_transcript: str = "transcript.txt"

    transcript: str = ""

    transcription_md_file_str: str = ""

    python_code_transcript: str = ""

    output_file: str = ""

    python_output_file: str = ""


# ============================================================
# PROMPT 1
# TRANSCRIPT -> TECHNICAL MARKDOWN
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",

            """
You are an expert technical content analyst and
senior Python developer.

Your task is to read a video transcript and convert it
into a useful, structured Markdown technical document.

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


# ============================================================
# PROMPT 2
# TRANSCRIPT + MARKDOWN -> IMPLEMENTATION MARKDOWN
# ============================================================

prompt2 = ChatPromptTemplate.from_messages(
    [

        (
            "system",

            """
You are a senior Python developer and AI engineer.

You are given:

1. The original YouTube transcript.
2. A technical Markdown analysis generated from that transcript.

Your task is to create a second Markdown document focused
on HOW TO IMPLEMENT the technical concepts described in
the video.

The document should be useful to a Python developer who
wants to reproduce the implementation.

Include sections such as:

# Implementation Guide

## 1. Technologies Used

List the technologies, libraries, frameworks, and tools
actually mentioned in the transcript.

## 2. Project Structure

Show a suggested project structure only when it is
supported by the transcript.

## 3. Implementation Steps

Explain the implementation step by step.

## 4. Classes

Extract classes actually described or implemented.

For each class explain:

- Purpose
- Fields
- Methods
- Responsibility

## 5. Functions

Extract functions actually described or implemented.

For each function explain:

- Purpose
- Inputs
- Processing
- Output

## 6. Agents

If agents are discussed, explain:

- Agent responsibility
- Inputs
- Outputs
- Tools
- Relationships with other agents

## 7. Tools

Extract tools mentioned in the transcript.

Explain what each tool does.

## 8. State

If state management is discussed, explain the state
structure and how information moves through the workflow.

## 9. Workflow

Create a Mermaid diagram when appropriate.

## 10. Python Implementation

Generate Python code only for implementation ideas
supported by the transcript.

IMPORTANT:

- Do NOT invent APIs that are not supported by the transcript.
- Do NOT invent libraries.
- Do NOT invent agent names.
- Do NOT invent functionality.
- Preserve actual technical names.
- If exact source code is unavailable, create a reasonable
  implementation skeleton based ONLY on the concepts
  explicitly supported by the transcript.
- Clearly mark implementation assumptions.
- Use Python code blocks.
- Use Markdown headings.
- Use Mermaid diagrams where useful.
- Return ONLY Markdown.
"""
        ),

        (
            "human",

            """
================ ORIGINAL TRANSCRIPT ================

{transcript}

================ END TRANSCRIPT =====================


================ TECHNICAL MARKDOWN ==================

{transcript_markdown}

================ END TECHNICAL MARKDOWN =============


Create the implementation-focused Markdown document.
"""
        )
    ]
)


# ============================================================
# HELPER
# REMOVE MARKDOWN CODE FENCES FROM LLM OUTPUT
# ============================================================

def clean_markdown(markdown: str) -> str:

    markdown = markdown.strip()

    if markdown.startswith("```markdown"):

        markdown = markdown[
            len("```markdown"):
        ].strip()

    elif markdown.startswith("```"):

        markdown = markdown[
            len("```"):
        ].strip()

    if markdown.endswith("```"):

        markdown = markdown[
            :-len("```")
        ].strip()

    return markdown


# ============================================================
# HELPER
# SAVE EACH STEP
# ============================================================

def save_step(
    step_number: int,
    step_name: str,
    content: str
):

    step_file = (
        STEPS_PATH
        / f"{step_number:02d}_{step_name}.md"
    )

    step_file.write_text(
        content,
        encoding="utf-8"
    )

    print(
        f"Step saved: {step_file}"
    )


# ============================================================
# NODE 1
# GET YOUTUBE TRANSCRIPT
# ============================================================

def get_youtube_transcript(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 1: GET YOUTUBE TRANSCRIPT")
    print("=" * 60)

    print(
        f"Video URL: {state.video_url}"
    )

    print(
        f"Saving transcript to: "
        f"{state.where_to_save_transcript}"
    )

    transcript = get_transcript_from_url(
        url=state.video_url,
        where_to_save=state.where_to_save_transcript
    )

    state.transcript = transcript

    print(
        f"Transcript characters: "
        f"{len(transcript)}"
    )

    save_step(
        1,
        "get_youtube_transcript",
        f"""# Step 1 - Get YouTube Transcript

## Video URL

{state.video_url}

## Transcript File

`{state.where_to_save_transcript}`

## Transcript Length

{len(state.transcript)} characters

## Transcript

{state.transcript}
"""
    )

    return state


# ============================================================
# NODE 2
# ANALYZE TRANSCRIPT
# ============================================================

def analyze_transcript(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 2: ANALYZE TRANSCRIPT")
    print("=" * 60)

    messages = prompt.format_messages(
        transcript=state.transcript
    )

    response = llm.invoke(messages)

    markdown = clean_markdown(
        response.content
    )

    state.transcription_md_file_str = markdown

    print(
        f"Generated Markdown characters: "
        f"{len(markdown)}"
    )

    save_step(
        2,
        "analyze_transcript",
        markdown
    )

    return state


# ============================================================
# NODE 3
# SAVE TECHNICAL MARKDOWN
# ============================================================

def save_markdown(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 3: SAVING TECHNICAL MARKDOWN")
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

    save_step(
        3,
        "save_markdown",
        f"""# Step 3 - Save Technical Markdown

## Output File

`{OUTPUT_FILE}`

## Characters

{len(state.transcription_md_file_str)}

## Generated Markdown

{state.transcription_md_file_str}
"""
    )

    return state


# ============================================================
# NODE 4
# GENERATE IMPLEMENTATION MARKDOWN
# ============================================================

def python_code_md_file_generate(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 4: GENERATE IMPLEMENTATION MARKDOWN")
    print("=" * 60)

    messages = prompt2.format_messages(
        transcript=state.transcript,
        transcript_markdown=(
            state.transcription_md_file_str
        )
    )

    response = llm.invoke(
        messages
    )

    markdown = clean_markdown(
        response.content
    )

    state.python_code_transcript = markdown

    print(
        f"Generated implementation Markdown "
        f"characters: {len(markdown)}"
    )

    save_step(
        4,
        "generate_implementation_markdown",
        markdown
    )

    return state


# ============================================================
# NODE 5
# SAVE IMPLEMENTATION MARKDOWN
# ============================================================

def save_markdown_python_code(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 5: SAVING IMPLEMENTATION MARKDOWN")
    print("=" * 60)

    PYTHON_OUTPUT_FILE.write_text(
        state.python_code_transcript,
        encoding="utf-8"
    )

    state.python_output_file = str(
        PYTHON_OUTPUT_FILE
    )

    print(
        f"Implementation Markdown file: "
        f"{PYTHON_OUTPUT_FILE}"
    )

    print(
        f"Characters: "
        f"{len(state.python_code_transcript)}"
    )

    save_step(
        5,
        "save_implementation_markdown",
        f"""# Step 5 - Save Implementation Markdown

## Output File

`{PYTHON_OUTPUT_FILE}`

## Characters

{len(state.python_code_transcript)}

## Implementation Markdown

{state.python_code_transcript}
"""
    )

    return state


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph_langchain():

    builder = StateGraph(
        State
    )

    builder.add_node(
        "get_youtube_transcript",
        get_youtube_transcript
    )

    builder.add_node(
        "analyze_transcript",
        analyze_transcript
    )

    builder.add_node(
        "save_markdown",
        save_markdown
    )

    builder.add_node(
        "python_code_md_file_generate",
        python_code_md_file_generate
    )

    builder.add_node(
        "save_markdown_python_code",
        save_markdown_python_code
    )

    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "get_youtube_transcript"
    )

    builder.add_edge(
        "get_youtube_transcript",
        "analyze_transcript"
    )

    builder.add_edge(
        "analyze_transcript",
        "save_markdown"
    )

    builder.add_edge(
        "save_markdown",
        "python_code_md_file_generate"
    )

    builder.add_edge(
        "python_code_md_file_generate",
        "save_markdown_python_code"
    )

    builder.add_edge(
        "save_markdown_python_code",
        END
    )

    graph = builder.compile()

    return graph


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("STARTING LANGGRAPH")
    print("=" * 60)

    print(
        f"Output directory: {OUTPUT_PATH}"
    )

    print(
        f"Steps directory: {STEPS_PATH}"
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_state = State(
        video_url=(
            "https://www.youtube.com/watch?v=FJHyCAi4GcY"
        ),
        where_to_save_transcript="transcript.txt"
    )

    # --------------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------------

    graph = build_graph_langchain()

    # --------------------------------------------------------
    # RUN GRAPH
    # --------------------------------------------------------

    final_state = graph.invoke(
        initial_state
    )

    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("WORKFLOW COMPLETED")
    print("=" * 60)

    print()
    print("Generated files:")
    print()

    print(
        f"Transcript:"
    )

    print(
        f"  {initial_state.where_to_save_transcript}"
    )

    print()

    print(
        f"Technical Markdown:"
    )

    print(
        f"  {final_state['output_file']}"
    )

    print()

    print(
        f"Implementation Markdown:"
    )

    print(
        f"  {final_state['python_output_file']}"
    )

    print()

    print(
        f"Technical Markdown characters:"
    )

    print(
        f"  {len(final_state['transcription_md_file_str'])}"
    )

    print()

    print(
        f"Implementation Markdown characters:"
    )

    print(
        f"  {len(final_state['python_code_transcript'])}"
    )

    print()

    print(
        f"Step files:"
    )

    print(
        f"  {STEPS_PATH}"
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
