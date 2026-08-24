from pathlib import Path
import re

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
    "/home/satvir/Downloads/MarkdownSatvir2"
)

STEPS_PATH = OUTPUT_PATH / "steps"

OUTPUT_FILE = OUTPUT_PATH / "transcript_analysis.md"

PYTHON_OUTPUT_FILE = (
    OUTPUT_PATH / "output_python_code.md"
)

GENERATED_PROJECT_PATH = (
    OUTPUT_PATH / "generated_project"
)

GRAPH_IMAGE_PATH = (
    OUTPUT_PATH / "langgraph.png"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

STEPS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

GENERATED_PROJECT_PATH.mkdir(
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

    # Step 2
    transcription_md_file_str: str = ""

    # Step 4
    python_code_transcript: str = ""

    # Step 3
    output_file: str = ""

    # Step 5
    python_output_file: str = ""

    # Step 6
    generated_project_spec: str = ""

    # Step 7
    generated_project_path: str = ""


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
# PROMPT 3
# IMPLEMENTATION MARKDOWN -> MULTI-FILE PYTHON PROJECT
# ============================================================

prompt3 = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are a senior Python developer and AI engineer.

Your task is to generate a COMPLETE MULTI-FILE PYTHON
PROJECT based on the implementation specification.

You are given:

1. The original YouTube transcript.
2. The implementation-focused Markdown.

The implementation Markdown is the primary specification.
The transcript provides additional context.

============================================================
IMPORTANT
============================================================

DO NOT generate one large Python file.

Generate a REAL PROJECT containing multiple files whenever
the implementation requires multiple components.

The project will be saved automatically into:

generated_project/

============================================================
PROJECT STRUCTURE
============================================================

Create appropriate folders and files.

For example:

generated_project/
    main.py
    requirements.txt
    agents/
        invoice_agent.py
        tax_agent.py
    tools/
        tax_tools.py
    models/
        schemas.py

Only create files that are actually needed.

============================================================
FILE FORMAT
============================================================

You MUST return every file using this exact format:

===== FILE: main.py =====

<complete file contents>

===== END FILE =====


===== FILE: agents/invoice_agent.py =====

<complete file contents>

===== END FILE =====


===== FILE: requirements.txt =====

<complete requirements>

===== END FILE =====

============================================================
RULES
============================================================

1. Generate complete code.

2. Generate all required imports.

3. Generate all required classes.

4. Generate all required functions.

5. Generate requirements.txt.

6. Use relative imports correctly.

7. Make the project internally consistent.

8. Make main.py the entry point when appropriate.

9. Create folders when necessary.

10. Do not put multiple Python modules into one file
    when they logically belong in separate modules.

11. Do not invent functionality that is unsupported by
    the transcript or implementation Markdown.

12. Do not invent external APIs.

13. Use the libraries mentioned in the source material
    when appropriate.

14. If the source does not provide exact implementation
    details, create a reasonable implementation based
    only on the documented concepts.

15. The generated project should be runnable after installing
    requirements.txt.

16. Do not include explanations outside the FILE sections.

17. Return ONLY FILE sections.

============================================================
ORIGINAL TRANSCRIPT
============================================================

{transcript}

============================================================
END ORIGINAL TRANSCRIPT
============================================================


============================================================
IMPLEMENTATION MARKDOWN
============================================================

{implementation_markdown}

============================================================
END IMPLEMENTATION MARKDOWN
============================================================
"""
        ),

        (
            "human",
            """
Generate the complete multi-file Python project now.

Remember:

===== FILE: path/to/file.py =====

code

===== END FILE =====

Generate every required file.
"""
        )
    ]
)


# ============================================================
# HELPER
# REMOVE MARKDOWN CODE FENCES
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
# NODE 6
# GENERATE MULTI-FILE PYTHON PROJECT
# ============================================================

def generate_python_code(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 6: GENERATE MULTI-FILE PYTHON PROJECT")
    print("=" * 60)

    print(
        "Generating project using Ollama..."
    )

    messages = prompt3.format_messages(
        transcript=state.transcript,
        implementation_markdown=(
            state.python_code_transcript
        )
    )

    response = llm.invoke(
        messages
    )

    project_spec = response.content.strip()

    state.generated_project_spec = (
        project_spec
    )

    print(
        f"Generated project specification "
        f"characters: {len(project_spec)}"
    )

    # Save raw LLM response
    save_step(
        6,
        "generate_python_code",
        f"""# Step 6 - Generate Python Project

## Generated Project Specification

{project_spec}
"""
    )

    return state


# ============================================================
# HELPER
# PARSE MULTI-FILE LLM RESPONSE
# ============================================================

def parse_generated_files(
    project_text: str
) -> dict[str, str]:

    pattern = re.compile(
        r"===== FILE: (.*?) =====\s*"
        r"(.*?)"
        r"\s*===== END FILE =====",
        re.DOTALL
    )

    matches = pattern.findall(
        project_text
    )

    files = {}

    for file_path, content in matches:

        file_path = file_path.strip()

        content = content.strip()

        if not file_path:
            continue

        files[file_path] = content

    return files


# ============================================================
# HELPER
# SAFETY CHECK FILE PATH
# ============================================================

def safe_project_path(
    project_root: Path,
    relative_path: str
) -> Path:

    relative_path = (
        relative_path
        .replace("\\", "/")
        .strip()
    )

    # Remove leading slash
    relative_path = relative_path.lstrip("/")

    target = (
        project_root / relative_path
    ).resolve()

    project_root_resolved = (
        project_root.resolve()
    )

    # Prevent ../ escaping project directory
    if (
        target != project_root_resolved
        and project_root_resolved
        not in target.parents
    ):
        raise ValueError(
            f"Unsafe file path generated: "
            f"{relative_path}"
        )

    return target


# ============================================================
# NODE 7
# SAVE PYTHON PROJECT
# ============================================================

def save_python_project(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 7: SAVE PYTHON PROJECT")
    print("=" * 60)

    project_root = (
        GENERATED_PROJECT_PATH
    )

    project_root.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Project directory: "
        f"{project_root}"
    )

    # --------------------------------------------------------
    # Parse files generated by LLM
    # --------------------------------------------------------

    files = parse_generated_files(
        state.generated_project_spec
    )

    if not files:

        raise ValueError(
            """
No files were found in the LLM response.

Expected format:

===== FILE: main.py =====

print("Hello")

===== END FILE =====
"""
        )

    print(
        f"Files generated by LLM: "
        f"{len(files)}"
    )

    saved_files = []

    # --------------------------------------------------------
    # Save every generated file
    # --------------------------------------------------------

    for relative_path, content in files.items():

        target_file = safe_project_path(
            project_root,
            relative_path
        )

        target_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        target_file.write_text(
            content,
            encoding="utf-8"
        )

        saved_files.append(
            relative_path
        )

        print(
            f"  ✓ {relative_path}"
        )

    # --------------------------------------------------------
    # Update state
    # --------------------------------------------------------

    state.generated_project_path = (
        str(project_root)
    )

    # --------------------------------------------------------
    # Save Step 7 report
    # --------------------------------------------------------

    file_list = "\n".join(
        f"- `{file}`"
        for file in saved_files
    )

    save_step(
        7,
        "save_python_project",
        f"""# Step 7 - Save Python Project

## Project Directory

`{project_root}`

## Generated Files

{file_list}

## Total Files

{len(saved_files)}
"""
    )

    print()
    print(
        "Python project successfully created."
    )

    return state


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph_langchain():

    builder = StateGraph(
        State
    )

    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

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

    builder.add_node(
        "generate_python_code",
        generate_python_code
    )

    builder.add_node(
        "save_python_project",
        save_python_project
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
        "generate_python_code"
    )

    builder.add_edge(
        "generate_python_code",
        "save_python_project"
    )

    builder.add_edge(
        "save_python_project",
        END
    )

    graph = builder.compile()

    return graph


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("STARTING LANGGRAPH AI CODE GENERATION WORKFLOW")
    print("=" * 70)

    print()
    print(
        f"Output directory:"
        f"\n{OUTPUT_PATH}"
    )

    print()
    print(
        f"Steps directory:"
        f"\n{STEPS_PATH}"
    )

    print()
    print(
        f"Generated project directory:"
        f"\n{GENERATED_PROJECT_PATH}"
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
    print("=" * 70)
    print("WORKFLOW COMPLETED")
    print("=" * 70)

    print()

    print(
        "Generated project:"
    )

    print(
        f"  {final_state['generated_project_path']}"
    )

    print()

    print(
        "Project files:"
    )

    project_path = Path(
        final_state[
            "generated_project_path"
        ]
    )

    for file in sorted(
        project_path.rglob("*")
    ):

        if file.is_file():

            print(
                f"  📄 "
                f"{file.relative_to(project_path)}"
            )

    print()

    print(
        "Step files:"
    )

    print(
        f"  {STEPS_PATH}"
    )

    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)
