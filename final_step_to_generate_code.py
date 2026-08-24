from pathlib import Path
import re

from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from youtubetranscripter import get_transcript_from_url

from IPython.display import Image, display, Markdown


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_MODEL2 = "qwen3-4b"

OUTPUT_PATH = Path(
    "/home/satvir/Downloads/MarkdownSatvir2"
)

STEPS_PATH = OUTPUT_PATH / "steps"

OUTPUT_FILE = (
    OUTPUT_PATH / "transcript_analysis.md"
)

PYTHON_OUTPUT_FILE = (
    OUTPUT_PATH / "output_python_code.md"
)

GENERATED_PROJECT_PATH = (
    OUTPUT_PATH / "generated_project"
)

GRAPH_IMAGE_PATH = (
    OUTPUT_PATH / "langgraph.png"
)

SUMMARY_FILE = (
    OUTPUT_PATH / "summary.md"
)

PDF_FILE = (
    OUTPUT_PATH / "summary.pdf"
)

# ============================================================
# TRANSCRIPT -> MERMAID FILES
# ============================================================

MERMAID_FILE = (
    OUTPUT_PATH / "transcript_workflow.mmd"
)

MERMAID_MARKDOWN_FILE = (
    OUTPUT_PATH / "transcript_workflow.md"
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
    temperature=0,
    num_ctx=32768,
)

llm2 = ChatOllama(
    model=OLLAMA_MODEL2,
    temperature=0,
    num_ctx=32768,
)


# ============================================================
# STATE
# ============================================================

class State(BaseModel):

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    video_url: str = (
        "https://www.youtube.com/watch?v=FJHyCAi4GcY"
    )

    where_to_save_transcript: str = (
        "transcript.txt"
    )

    # --------------------------------------------------------
    # Transcript
    # --------------------------------------------------------

    transcript: str = ""

    # --------------------------------------------------------
    # Technical Markdown
    # --------------------------------------------------------

    transcription_md_file_str: str = ""

    output_file: str = ""

    # --------------------------------------------------------
    # Implementation Markdown
    # --------------------------------------------------------

    python_code_transcript: str = ""

    python_output_file: str = ""

    # --------------------------------------------------------
    # Generated Python project
    # --------------------------------------------------------

    generated_project_spec: str = ""

    generated_project_path: str = ""

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary: str = ""

    summary_file: str = ""

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pdf_file: str = ""

    # --------------------------------------------------------
    # Transcript -> Mermaid
    # --------------------------------------------------------

    mermaid_code: str = ""

    mermaid_file: str = ""

    mermaid_markdown_file: str = ""


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
        ),
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

Include:

# Implementation Guide

## 1. Technologies Used

List the technologies, libraries, frameworks, and tools
actually mentioned in the transcript.

## 2. Project Structure

Show a suggested project structure only when supported
by the transcript.

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

- Do NOT invent APIs.
- Do NOT invent libraries.
- Do NOT invent agent names.
- Do NOT invent functionality.
- Preserve actual technical names.
- If exact source code is unavailable, create a reasonable
  implementation skeleton based ONLY on the concepts
  explicitly supported by the transcript.
- Clearly mark implementation assumptions.
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
        ),
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

============================================================
RULES
============================================================

1. Generate complete code.

2. Generate all required imports.

3. Generate all required classes.

4. Generate all required functions.

5. Generate requirements.txt when dependencies are needed.

6. Use relative imports correctly.

7. Make the project internally consistent.

8. Make main.py the entry point when appropriate.

9. Create folders when necessary.

10. Do not put multiple Python modules into one file
    when they logically belong in separate modules.

11. Do not invent unsupported functionality.

12. Do not invent external APIs.

13. Use libraries mentioned in the source material
    when appropriate.

14. If exact implementation details are unavailable,
    create a reasonable implementation based only on
    documented concepts.

15. The project should be runnable after installing
    requirements.txt.

16. Do not include explanations outside FILE sections.

17. Return ONLY FILE sections.
"""
        ),
        (
            "human",
            """
Generate the complete multi-file Python project now.

============================================================
ORIGINAL TRANSCRIPT
============================================================

{transcript}

============================================================
IMPLEMENTATION MARKDOWN
============================================================

{implementation_markdown}

============================================================

Remember:

===== FILE: path/to/file.py =====

code

===== END FILE =====

Generate every required file.
"""
        ),
    ]
)


# ============================================================
# PROMPT 4
# TRANSCRIPT -> SUMMARY
# ============================================================

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a technical content summarizer.

Read the provided video transcript and create a concise,
developer-focused summary.

Include:

# Summary

## Main Topic

## Key Concepts

## Technologies

## Workflow

## Important Technical Lessons

## Key Takeaways

IMPORTANT:

- Use only information supported by the transcript.
- Do not invent facts.
- Preserve technical names.
- Return ONLY Markdown.
"""
        ),
        (
            "human",
            """
================ TRANSCRIPT ================

{transcript}

================ END TRANSCRIPT =============

Create the summary.
"""
        ),
    ]
)


# ============================================================
# PROMPT 5
# TRANSCRIPT -> MERMAID WORKFLOW
# ============================================================

mermaid_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert workflow architect, AI agent architect,
and Mermaid diagram generator.

Your task is to read a YouTube transcript and reconstruct
the ACTUAL WORKFLOW described by the speaker.

The output will be used as a Mermaid flowchart.

============================================================
IMPORTANT
============================================================

Do NOT summarize the transcript.

Do NOT create a simple numbered list.

Instead, understand the workflow and represent its structure.

Identify:

- sequential steps
- branches
- decisions
- conditional paths
- loops
- retry loops
- parallel execution
- multiple agents
- tools
- memory
- state
- routers
- dispatchers
- specialists
- workers
- critics
- planners
- executors
- researchers
- synthesizers
- consolidators
- inputs
- outputs
- handoffs between agents

============================================================
SEQUENTIAL FLOW
============================================================

For sequential execution use:

A --> B

============================================================
DECISIONS
============================================================

If the transcript describes a decision:

A --> B{"Decision?"}

Then create branches:

B -->|Yes| C
B -->|No| D

============================================================
LOOPS
============================================================

If the transcript describes something that repeats:

A --> B
B --> C{"Approved?"}
C -->|No| A
C -->|Yes| D

============================================================
PARALLEL EXECUTION
============================================================

If several agents execute independently or in parallel:

A --> B
A --> C
A --> D

Then converge:

B --> E
C --> E
D --> E

============================================================
ROUTER / DISPATCHER
============================================================

For routing:

A --> B{"Question Type?"}

B -->|VAT| C["VAT Specialist"]
B -->|Corporate Tax| D["Corporate Tax Specialist"]
B -->|Payroll| E["Payroll Specialist"]

============================================================
MEMORY / STATE
============================================================

If the transcript describes memory or state, represent it.

For example:

A["Client Query"]
    --> B["Client History"]
    --> C["Client Account Agent"]

============================================================
MULTIPLE PATTERNS
============================================================

The transcript may contain multiple AI agent patterns.

Use Mermaid subgraphs when appropriate.

For example:

subgraph P1["Pattern 1 - Single Agent"]
    A --> B
end

subgraph P2["Pattern 2 - ReAct"]
    C --> D
end

============================================================
PATTERNS
============================================================

If supported by the transcript, identify patterns such as:

1. Single Agent
2. ReAct
3. Structured Output
4. Memory Augmented Agent
5. Reflection / Self Critic
6. Plan and Execute
7. Router / Dispatcher
8. Parallel Research
9. Orchestrator / Worker

Do not automatically create all nine.

Only create patterns supported by the transcript.

============================================================
AGENTS
============================================================

Represent important agents as nodes.

Examples:

["Invoice Agent"]
["Tax Researcher"]
["Tax Return Actor"]
["Tax Return Critic"]
["Audit Planner"]
["Document Collector"]
["Account Reconciler"]
["Risk Flagger"]
["VAT Specialist"]
["Tax Specialist"]
["Payroll Specialist"]
["Synthesizer"]
["Invoice Worker"]
["Tax Worker"]
["Report Generator"]
["Final Consolidator"]

============================================================
TOOLS
============================================================

If tools are an important part of the workflow, represent
them as nodes.

Examples:

["Docling"]
["Calculate VAT"]
["Calculate Total"]

============================================================
VERY IMPORTANT
============================================================

Do NOT invent workflow relationships.

Only create relationships supported by the transcript.

Correct obvious speech-to-text errors when the meaning
is clear.

Preserve important technical names such as:

- Google ADK
- Pydantic
- Docling
- Gemini
- VAT
- Tax Researcher
- Invoice Agent
- Tax Return Actor
- Tax Return Critic
- Audit Planner
- Document Collector
- Account Reconciler
- Risk Flagger
- VAT Specialist
- Tax Specialist
- Payroll Specialist
- Netherlands
- Germany
- UK
- Synthesizer
- Invoice Worker
- Tax Worker
- Report Generator
- Consolidator

============================================================
MERMAID SYNTAX
============================================================

Use:

flowchart TD

Use readable node IDs.

Example:

A["Client Query"]

Avoid special characters that could break Mermaid.

============================================================
OUTPUT
============================================================

Return ONLY valid Mermaid code.

Do NOT return markdown fences.

Do NOT provide explanations.

Do NOT provide commentary.

The first line must be:

flowchart TD
"""
        ),
        (
            "human",
            """
================ ORIGINAL TRANSCRIPT ================

{transcript}

================ END TRANSCRIPT =====================

Convert this transcript into a Mermaid workflow graph.
"""
        ),
    ]
)


# ============================================================
# HELPER
# CLEAN MARKDOWN
# ============================================================

def clean_markdown(
    markdown: str
) -> str:

    markdown = markdown.strip()

    if markdown.startswith(
        "```markdown"
    ):

        markdown = markdown[
            len("```markdown"):
        ].strip()

    elif markdown.startswith(
        "```"
    ):

        markdown = markdown[
            len("```"):
        ].strip()

    if markdown.endswith(
        "```"
    ):

        markdown = markdown[
            :-len("```")
        ].strip()

    return markdown


# ============================================================
# HELPER
# CLEAN MERMAID
# ============================================================

def clean_mermaid(
    mermaid: str
) -> str:

    mermaid = mermaid.strip()

    # --------------------------------------------------------
    # Remove markdown fences
    # --------------------------------------------------------

    mermaid = re.sub(
        r"^```mermaid\s*",
        "",
        mermaid,
        flags=re.IGNORECASE
    )

    mermaid = re.sub(
        r"^```\s*",
        "",
        mermaid
    )

    mermaid = re.sub(
        r"\s*```$",
        "",
        mermaid
    )

    mermaid = mermaid.strip()

    # --------------------------------------------------------
    # Find flowchart if model added text before it
    # --------------------------------------------------------

    if not mermaid.lower().startswith(
        "flowchart"
    ):

        match = re.search(
            r"flowchart\s+TD.*",
            mermaid,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:

            mermaid = match.group(0).strip()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not mermaid.lower().startswith(
        "flowchart"
    ):

        raise ValueError(
            "LLM did not return a Mermaid flowchart."
        )

    return mermaid


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

    transcript = get_transcript_from_url(
        url=state.video_url,
        where_to_save=(
            state.where_to_save_transcript
        )
    )

    if not transcript:

        raise ValueError(
            "YouTube transcript is empty."
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

    response = llm.invoke(
        messages
    )

    markdown = clean_markdown(
        response.content
    )

    state.transcription_md_file_str = (
        markdown
    )

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
# GENERATE SUMMARY
# ============================================================

def generate_summary(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 3: GENERATE SUMMARY")
    print("=" * 60)

    messages = summary_prompt.format_messages(
        transcript=state.transcript
    )

    response = llm2.invoke(
        messages
    )

    summary = clean_markdown(
        response.content
    )

    state.summary = summary

    print(
        f"Summary characters: "
        f"{len(summary)}"
    )

    save_step(
        3,
        "generate_summary",
        summary
    )

    return state


# ============================================================
# NODE 4
# SAVE SUMMARY
# ============================================================

def save_summary(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 4: SAVE SUMMARY")
    print("=" * 60)

    SUMMARY_FILE.write_text(
        state.summary,
        encoding="utf-8"
    )

    state.summary_file = (
        str(SUMMARY_FILE)
    )

    save_step(
        4,
        "save_summary",
        f"""# Step 4 - Save Summary

## File

`{SUMMARY_FILE}`

## Summary

{state.summary}
"""
    )

    return state


# ============================================================
# NODE 5
# GENERATE PDF
# ============================================================

def save_pdf_final(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 5: GENERATE SUMMARY PDF")
    print("=" * 60)

    try:

        from reportlab.lib.pagesizes import A4

        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle
        )

        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer
        )

        from reportlab.lib.enums import (
            TA_LEFT
        )

    except ImportError:

        raise ImportError(
            "ReportLab is required. "
            "Install with: pip install reportlab"
        )

    document = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_LEFT
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        leading=14
    )

    story = []

    for line in state.summary.splitlines():

        line = line.strip()

        if not line:

            story.append(
                Spacer(1, 8)
            )

            continue

        if line.startswith("# "):

            story.append(
                Paragraph(
                    line[2:],
                    title_style
                )
            )

        elif line.startswith("## "):

            story.append(
                Paragraph(
                    line[3:],
                    styles["Heading2"]
                )
            )

        elif line.startswith("### "):

            story.append(
                Paragraph(
                    line[4:],
                    styles["Heading3"]
                )
            )

        else:

            text = re.sub(
                r"\*\*(.*?)\*\*",
                r"<b>\1</b>",
                line
            )

            text = re.sub(
                r"`(.*?)`",
                r"<font name='Courier'>\1</font>",
                text
            )

            story.append(
                Paragraph(
                    text,
                    body_style
                )
            )

    document.build(
        story
    )

    state.pdf_file = (
        str(PDF_FILE)
    )

    save_step(
        5,
        "generate_summary_pdf",
        f"""# Step 5 - Generate Summary PDF

## PDF File

`{PDF_FILE}`

## Source

`{SUMMARY_FILE}`

PDF generated successfully.
"""
    )

    print(
        f"PDF generated: {PDF_FILE}"
    )

    return state


# ============================================================
# NODE 6
# SAVE TECHNICAL MARKDOWN
# ============================================================

def save_markdown(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 6: SAVE TECHNICAL MARKDOWN")
    print("=" * 60)

    OUTPUT_FILE.write_text(
        state.transcription_md_file_str,
        encoding="utf-8"
    )

    state.output_file = (
        str(OUTPUT_FILE)
    )

    save_step(
        6,
        "save_markdown",
        f"""# Step 6 - Save Technical Markdown

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
# NODE 7
# GENERATE IMPLEMENTATION MARKDOWN
# ============================================================

def python_code_md_file_generate(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 7: GENERATE IMPLEMENTATION MARKDOWN")
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

    state.python_code_transcript = (
        markdown
    )

    save_step(
        7,
        "generate_implementation_markdown",
        markdown
    )

    return state


# ============================================================
# NODE 8
# SAVE IMPLEMENTATION MARKDOWN
# ============================================================

def save_markdown_python_code(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 8: SAVE IMPLEMENTATION MARKDOWN")
    print("=" * 60)

    PYTHON_OUTPUT_FILE.write_text(
        state.python_code_transcript,
        encoding="utf-8"
    )

    state.python_output_file = (
        str(PYTHON_OUTPUT_FILE)
    )

    save_step(
        8,
        "save_implementation_markdown",
        f"""# Step 8 - Save Implementation Markdown

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
# NODE 9
# GENERATE MULTI-FILE PYTHON PROJECT
# ============================================================

def generate_python_code(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 9: GENERATE MULTI-FILE PYTHON PROJECT")
    print("=" * 60)

    messages = prompt3.format_messages(
        transcript=state.transcript,
        implementation_markdown=(
            state.python_code_transcript
        )
    )

    response = llm.invoke(
        messages
    )

    project_spec = (
        response.content.strip()
    )

    state.generated_project_spec = (
        project_spec
    )

    save_step(
        9,
        "generate_python_code",
        f"""# Step 9 - Generate Python Project

## Generated Project Specification

{project_spec}
"""
    )

    return state


# ============================================================
# HELPER
# PARSE GENERATED FILES
# ============================================================

def parse_generated_files(
    project_text: str
) -> dict[str, str]:

    pattern = re.compile(
        r"===== FILE:\s*(.*?)\s*=====\s*"
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

        if file_path:

            files[file_path] = content

    return files


# ============================================================
# HELPER
# SAFE PROJECT PATH
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

    relative_path = (
        relative_path.lstrip("/")
    )

    target = (
        project_root / relative_path
    ).resolve()

    root = (
        project_root.resolve()
    )

    if (
        target != root
        and root not in target.parents
    ):

        raise ValueError(
            f"Unsafe generated file path: "
            f"{relative_path}"
        )

    return target


# ============================================================
# NODE 10
# SAVE PYTHON PROJECT
# ============================================================

def save_python_project(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 10: SAVE PYTHON PROJECT")
    print("=" * 60)

    project_root = (
        GENERATED_PROJECT_PATH
    )

    project_root.mkdir(
        parents=True,
        exist_ok=True
    )

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

    saved_files = []

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

    state.generated_project_path = (
        str(project_root)
    )

    file_list = "\n".join(
        f"- `{file}`"
        for file in saved_files
    )

    save_step(
        10,
        "save_python_project",
        f"""# Step 10 - Save Python Project

## Project Directory

`{project_root}`

## Generated Files

{file_list}

## Total Files

{len(saved_files)}
"""
    )

    return state


# ============================================================
# NODE 11
# GENERATE MERMAID FROM TRANSCRIPT
# ============================================================

def generate_mermaid_from_transcript(
    state: State
):

    print()
    print("=" * 60)
    print("STEP 11: GENERATE MERMAID FROM TRANSCRIPT")
    print("=" * 60)

    print(
        "Sending original transcript to LLM..."
    )

    messages = mermaid_prompt.format_messages(
        transcript=state.transcript
    )

    response = llm.invoke(
        messages
    )

    mermaid = clean_mermaid(
        response.content
    )

    state.mermaid_code = (
        mermaid
    )

    print()
    print(
        "Generated Mermaid:"
    )

    print(
        "-" * 60
    )

    print(
        mermaid
    )

    print(
        "-" * 60
    )

    save_step(
        11,
        "generate_mermaid_from_transcript",
        f"""# Step 11 - Generate Mermaid From Transcript

## Video

{state.video_url}

## Mermaid

```mermaid
{mermaid}
