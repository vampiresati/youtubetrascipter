from typing import TypedDict
from pathlib import Path
from pydantic import BaseModel, Field
import json

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from IPython.display import Image, display


# ============================================================
# Ollama
# ============================================================

llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)


# ============================================================
# 2. PYDANTIC MODELS
# ============================================================

class ProjectFile(BaseModel):
    path: str = Field(
        description="Relative path of the file"
    )

    description: str = Field(
        description="What this file should contain"
    )


class ProjectStructure(BaseModel):

    project_name: str

    files: list[ProjectFile]


class GeneratedFile(BaseModel):

    path: str

    content: str


class GeneratedProject(BaseModel):

    files: list[GeneratedFile]


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class State(TypedDict):
    transcript: str
    analysis: str
    project_structure: dict
    files: dict
    review: str
    output_path: str


# ============================================================
# Save LangGraph as PNG
# ============================================================

def save_graph_png(app, output_path: str):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    png_bytes = app.get_graph().draw_mermaid_png()

    output_path.write_bytes(png_bytes)

    return str(output_path)


# ============================================================
# Node 1: Analyze Transcript
# ============================================================

def analyze_transcript(state: State):

    print("\n")
    print("=" * 70)
    print("🔍 STEP 1: ANALYZING YOUTUBE TRANSCRIPT")
    print("=" * 70)

    prompt = f"""
You are a senior Python software architect.

Analyze this YouTube tutorial transcript.

Your job is to understand exactly what project the
tutorial is teaching.

Extract:

1. Project purpose
2. Features
3. Python libraries
4. Classes
5. Functions
6. Agents
7. Tools
8. Data structures
9. Workflow
10. Configuration
11. Database requirements
12. File/module organization
13. Important implementation details

Do NOT generate code yet.

Give a detailed technical analysis.

YOUTUBE TRANSCRIPT
==================

{state["transcript"]}
"""

    response = llm.invoke(prompt)

    print("✅ Transcript analysis completed")

    return {
        "analysis": response.content
    }


# ============================================================
# 6. NODE: DESIGN PROJECT STRUCTURE
# ============================================================

def design_project(state: State):
    print("\n")
    print("=" * 70)
    print("🏗️ STEP 2: DESIGNING PROJECT STRUCTURE")
    print("=" * 70)

    structured_llm = llm.with_structured_output(
        ProjectStructure
    )

    prompt = f"""
You are a senior Python software architect.

Based on the tutorial analysis below,
design a proper Python project structure.

Do NOT put everything into main.py.

Separate responsibilities into appropriate modules.

For example, depending on the tutorial you might create:

main.py
config.py
state.py
graph.py

agents/
tools/
services/
models/

requirements.txt
README.md

Only create files that are actually useful for
the project.

The structure must be based on the tutorial.

TUTORIAL ANALYSIS
=================

{state["analysis"]}
"""

    result = structured_llm.invoke(prompt)

    print("\n📁 PROJECT STRUCTURE:")

    for file in result.files:

        print(
            f"   📄 {file.path}"
            f" → {file.description}"
        )

    return {
        "project_structure": result.model_dump()
    }


# ============================================================
# 7. NODE: GENERATE PROJECT FILES
# ============================================================

def generate_files(state: State):
    print("\n")
    print("=" * 70)
    print("💻 STEP 3: GENERATING PROJECT FILES")
    print("=" * 70)
    structured_llm = llm.with_structured_output(
        GeneratedProject
    )
    structure = state["project_structure"]
    files_description = "\n".join(
        f"- {file['path']}: {file['description']}"
        for file in structure["files"]
    )

    prompt = f"""
You are an expert Python developer.

Create the complete Python project described below.

TUTORIAL ANALYSIS
=================

{state["analysis"]}


PROJECT STRUCTURE
=================

{files_description}


IMPORTANT REQUIREMENTS
=====================

1. Generate real runnable Python code.

2. Generate every requested file.

3. Keep responsibilities separated.

4. Use correct imports between files.

5. Do not use fake functions.

6. Do not leave TODO placeholders.

7. Include proper error handling.

8. Include requirements.txt.

9. Include README.md.

10. Make the project internally consistent.

11. Use the APIs and frameworks described in
the tutorial whenever possible.

Return every file with its relative path
and complete content.
"""

    result = structured_llm.invoke(prompt)

    files = {}

    for file in result.files:

        files[file.path] = file.content

        print(
            f"   ✅ Generated: {file.path}"
        )

    return {
        "files": files
    }


# ============================================================
# 8. NODE: REVIEW PROJECT
# ============================================================

def review_project(state: State):

    print("\n")
    print("=" * 70)
    print("🔎 STEP 4: REVIEWING PROJECT")
    print("=" * 70)

    project_text = ""

    for path, content in state["files"].items():

        project_text += f"""

==================================================
FILE: {path}
==================================================

{content}

"""

    prompt = f"""
You are a senior Python code reviewer.

Review this complete generated project.

PROJECT:

{project_text}

Check:

1. Python syntax
2. Missing imports
3. Incorrect imports
4. Missing dependencies
5. Incorrect LangGraph APIs
6. Incorrect LangChain APIs
7. Incorrect Ollama usage
8. Circular imports
9. Missing functions
10. Missing classes
11. Incorrect file paths
12. Logical problems
13. Inconsistent code between files

Provide a detailed review.

If the project is correct, say:

PROJECT_VALID

Otherwise explain every problem that must be fixed.
"""

    response = llm.invoke(prompt)

    print(response.content)

    return {
        "review": response.content
    }


# ============================================================
# 9. NODE: SAVE PROJECT
# ============================================================

def save_project(state: State):

    print("\n")
    print("=" * 70)
    print("💾 STEP 5: SAVING PROJECT")
    print("=" * 70)

    project_path = Path(
        state["output_path"]
    )

    project_path.mkdir(
        parents=True,
        exist_ok=True
    )

    for relative_path, content in state["files"].items():

        file_path = (
            project_path / relative_path
        )

        # Prevent accidental absolute paths
        if file_path.is_absolute():

            print(
                f"⚠️ Skipping unsafe path: "
                f"{relative_path}"
            )

            continue

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path.write_text(
            content,
            encoding="utf-8"
        )

        print(
            f"   📄 Saved: {file_path}"
        )

    print("\n✅ PROJECT SAVED")
    print(
        f"📁 Location: {project_path}"
    )

    return {
        "output_path": str(project_path)
    }


# ============================================================
# 10. BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(State)

    # Nodes

    graph.add_node(
        "analyze",
        analyze_transcript
    )

    graph.add_node(
        "design_project",
        design_project
    )

    graph.add_node(
        "generate_files",
        generate_files
    )

    graph.add_node(
        "review",
        review_project
    )

    graph.add_node(
        "save_project",
        save_project
    )

    # Edges

    graph.add_edge(
        START,
        "analyze"
    )

    graph.add_edge(
        "analyze",
        "design_project"
    )

    graph.add_edge(
        "design_project",
        "generate_files"
    )

    graph.add_edge(
        "generate_files",
        "review"
    )

    graph.add_edge(
        "review",
        "save_project"
    )

    graph.add_edge(
        "save_project",
        END
    )

    return graph.compile()


# ============================================================
# 11. RUN GRAPH WITH LIVE PROGRESS
# ============================================================

def run_graph(
    app,
    transcript: str,
    output_path: str
):

    initial_state = {

        "transcript": transcript,

        "analysis": "",

        "project_structure": {},

        "files": {},

        "review": "",

        "output_path": output_path
    }

    print("\n")
    print("=" * 70)
    print("🚀 YOUTUBE → PYTHON PROJECT AGENT")
    print("=" * 70)

    print("\nSTART")
    print("  ↓")

    final_state = initial_state.copy()

    for event in app.stream(
        initial_state,
        stream_mode="updates"
    ):

        for node_name, output in event.items():

            print(
                f"\n▶️ RUNNING NODE: {node_name}"
            )

            if node_name == "analyze":

                print(
                    "   ✓ Tutorial understood"
                )

            elif node_name == "design_project":

                print(
                    "   ✓ Project structure created"
                )

            elif node_name == "generate_files":

                print(
                    f"   ✓ Generated "
                    f"{len(output.get('files', {}))} files"
                )

            elif node_name == "review":

                print(
                    "   ✓ Project reviewed"
                )

            elif node_name == "save_project":

                print(
                    "   ✓ Project saved"
                )

            final_state.update(output)

            print("  ↓")

    print("END")

    print("\n")
    print("=" * 70)
    print("🎉 WORKFLOW COMPLETED")
    print("=" * 70)

    return final_state


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Replace this with your actual YouTube transcript
    # --------------------------------------------------------

    transcript = """
    Today we are going to build an AI agent system.

    We will use LangGraph for orchestration.

    We will use Ollama as the local language model.

    First we create a State class.

    Then we create an analyzer agent.

    Then we create a code generator.

    Finally we create a review agent.

    The project should contain separate modules
    for agents, tools and graph configuration.

    We also create a requirements.txt file.
    """

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    app = build_graph()

    print(
        "\n✅ LangGraph compiled successfully."
    )

    # --------------------------------------------------------
    # Save graph image
    # --------------------------------------------------------

    graph_path = save_graph_png(
        app,
        "/home/satvir/youtubetrascipter/langgraph.png"
    )

    print(
        f"\n📊 LangGraph saved to:"
        f"\n{graph_path}"
    )

    # --------------------------------------------------------
    # Display graph in Jupyter
    # --------------------------------------------------------

    try:

        display(
            Image(
                filename=graph_path
            )
        )

    except Exception:

        print(
            "Graph image saved successfully."
        )

    # --------------------------------------------------------
    # Run workflow
    # --------------------------------------------------------

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

    for file_path in result.get(
        "files",
        {}
    ):

        print(
            f"   📄 {file_path}"
        )
