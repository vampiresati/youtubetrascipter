from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from youtubetranscripter import get_transcript_from_url


where_to_save_transcript="/home/satvir/youtubetrascipter/transcript.txt"
where_to_save_transcript_markup="/home/satvir/youtubetrascipter/transcript_analysis.md"
transcript=get_transcript_from_url(url="https://www.youtube.com/watch?v=FJHyCAi4GcY",where_to_save=where_to_save_transcript)
with open(where_to_save_transcript, "w", encoding="utf-8") as file:
    file.write(transcript)

INPUT_FILE = Path(where_to_save_transcript)
OUTPUT_FILE = Path(where_to_save_transcript_markup)
MODEL = "qwen2.5-coder:7b"


llm = ChatOllama(model=MODEL,temperature=0)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert technical content analyst and senior python developer.

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
