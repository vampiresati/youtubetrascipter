from typing import TypedDict
from pathlib import Path
from datetime import datetime
import json
from pydantic import BaseModel, Field
from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langchain_ollama import ChatOllama

from IPython.display import (
    Image,
    display,
)


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = "qwen2.5-coder:7b"

OUTPUT_PATH = Path(
    "/home/satvir/Downloads/TestCreation"
)

STEPS_PATH = OUTPUT_PATH / "steps"

GRAPH_IMAGE_PATH = Path(
    "/home/satvir/youtubetrascipter/langgraph.png"
)


# ============================================================
# OLLAMA
# ============================================================

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)


# ============================================================
# PYDANTIC MODELS
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
# LANGGRAPH STATE
# ============================================================

class State(TypedDict):

    transcript: str

    analysis: str

    project_structure: dict

    files: dict

    review: str

    output_path: str

    saved_files: int


# ============================================================
# STEP MARKDOWN LOGGER
# ============================================================

class StepMarkdownLogger:

    def __init__(
        self,
        project_path: Path,
    ):

        self.project_path = Path(
            project_path
        )

        self.steps_path = (
            self.project_path / "steps"
        )

        self.steps_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Create a new step file
    # --------------------------------------------------------

    def create_step(
        self,
        step_number: int,
        step_name: str,
    ):

        filename = (
            f"{step_number:02d}_"
            f"{step_name}.md"
        )

        path = (
            self.steps_path / filename
        )

        content = f"""# Step {step_number}: {step_name}

**Started:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""

        path.write_text(
            content,
            encoding="utf-8",
        )

        return path

    # --------------------------------------------------------
    # Write Markdown section
    # --------------------------------------------------------

    def write(
        self,
        path: Path,
        title: str,
        content: str,
    ):

        with path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                f"\n## {title}\n\n"
            )

            f.write(
                content
            )

            if not content.endswith("\n"):

                f.write("\n")

            f.write("\n---\n")

    # --------------------------------------------------------
    # Write code / text block
    # --------------------------------------------------------

    def write_code(
        self,
        path: Path,
        title: str,
        content: str,
        language: str = "text",
    ):

        with path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                f"\n## {title}\n\n"
            )

            f.write(
                f"```{language}\n"
            )

            f.write(
                content
            )

            if not content.endswith("\n"):

                f.write("\n")

            f.write(
                "```\n\n---\n"
            )

    # --------------------------------------------------------
    # Write JSON
    # --------------------------------------------------------

    def write_json(
        self,
        path: Path,
        title: str,
        data,
    ):

        content = json.dumps(
            data,
            indent=4,
            ensure_ascii=False,
        )

        self.write_code(
            path,
            title,
            content,
            "json",
        )


# ============================================================
# LOGGER
# ============================================================

logger = StepMarkdownLogger(
    OUTPUT_PATH
)


# ============================================================
# SAVE LANGGRAPH PNG
# ============================================================

def save_graph_png(
    app,
    output_path: str,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_bytes = (
        app
        .get_graph()
        .draw_mermaid_png()
    )

    output_path.write_bytes(
        png_bytes
    )

    return str(
        output_path
    )


# ============================================================
# STEP 1
# ANALYZE TRANSCRIPT
# ============================================================

def analyze_transcript(
    state: State,
):

    print()
    print("=" * 70)
    print(
        "🔍 STEP 1: ANALYZING YOUTUBE TRANSCRIPT"
    )
    print("=" * 70)

    step_file = logger.create_step(
        1,
        "analyze_transcript",
    )

    # --------------------------------------------------------
    # Save input
    # --------------------------------------------------------

    logger.write_code(
        step_file,
        "Input Transcript",
        state["transcript"],
        "text",
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a senior Python software architect.

Analyze this YouTube tutorial transcript.

Your job is to understand exactly what
project the tutorial is teaching.

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

IMPORTANT:

- Do NOT generate code yet.
- Do NOT invent unnecessary features.
- Base the analysis on the transcript.

YOUTUBE TRANSCRIPT
==================

{state["transcript"]}
"""

    logger.write_code(
        step_file,
        "LLM Prompt",
        prompt,
        "text",
    )

    # --------------------------------------------------------
    # Call LLM
    # --------------------------------------------------------

    try:

        response = llm.invoke(
            prompt
        )

    except Exception as e:

        logger.write_code(
            step_file,
            "ERROR",
            f"{type(e).__name__}: {e}",
            "text",
        )

        raise

    analysis = response.content

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    logger.write_code(
        step_file,
        "LLM Output - Analysis",
        analysis,
        "text",
    )

    print(
        "✅ Transcript analysis completed"
    )

    return {
        "analysis": analysis
    }


# ============================================================
# STEP 2
# DESIGN PROJECT
# ============================================================

def design_project(
    state: State,
):

    print()
    print("=" * 70)
    print(
        "🏗️ STEP 2: DESIGNING PROJECT STRUCTURE"
    )
    print("=" * 70)

    step_file = logger.create_step(
        2,
        "design_project",
    )

    # --------------------------------------------------------
    # Save previous step
    # --------------------------------------------------------

    logger.write_code(
        step_file,
        "Previous Step - Analysis",
        state["analysis"],
        "text",
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a senior Python software architect.

Based on the tutorial analysis below,
design a proper Python project structure.

Do NOT put everything into main.py.

Separate responsibilities into appropriate
modules.

Depending on the tutorial, use structures
such as:

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

Only create files that are actually useful.

TUTORIAL ANALYSIS
=================

{state["analysis"]}

IMPORTANT:

1. Every file must have a relative path.
2. Every file must have a description.
3. Keep responsibilities separated.
4. Do not invent unnecessary files.
"""

    logger.write_code(
        step_file,
        "LLM Prompt",
        prompt,
        "text",
    )

    # --------------------------------------------------------
    # Structured output
    # --------------------------------------------------------

    structured_llm = (
        llm.with_structured_output(
            ProjectStructure
        )
    )

    try:

        result = structured_llm.invoke(
            prompt
        )

    except Exception as e:

        logger.write_code(
            step_file,
            "ERROR",
            f"{type(e).__name__}: {e}",
            "text",
        )

        raise

    structure = result.model_dump()

    # --------------------------------------------------------
    # Save structure
    # --------------------------------------------------------

    logger.write_json(
        step_file,
        "Generated Project Structure",
        structure,
    )

    print()
    print(
        f"📁 Project: "
        f"{result.project_name}"
    )

    print(
        f"📦 Files planned: "
        f"{len(result.files)}"
    )

    for file in result.files:

        print(
            f"   📄 {file.path}"
        )

    return {
        "project_structure": structure
    }


# ============================================================
# STEP 3
# GENERATE FILES
# ============================================================

def generate_files(
    state: State,
):

    print()
    print("=" * 70)
    print(
        "💻 STEP 3: GENERATING PROJECT FILES"
    )
    print("=" * 70)

    step_file = logger.create_step(
        3,
        "generate_files",
    )

    structure = state[
        "project_structure"
    ]

    files_description = "\n".join(
        f"- {file['path']}: "
        f"{file['description']}"
        for file in structure["files"]
    )

    # --------------------------------------------------------
    # Save analysis
    # --------------------------------------------------------

    logger.write_code(
        step_file,
        "Tutorial Analysis",
        state["analysis"],
        "text",
    )

    # --------------------------------------------------------
    # Save requested files
    # --------------------------------------------------------

    logger.write_code(
        step_file,
        "Requested Project Files",
        files_description,
        "text",
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an expert Python developer.

Create the COMPLETE Python project
described below.

TUTORIAL ANALYSIS
=================

{state["analysis"]}

PROJECT STRUCTURE
=================

{files_description}

IMPORTANT REQUIREMENTS
======================

1. Generate REAL runnable Python code.

2. Generate EVERY requested file.

3. Keep responsibilities separated.

4. Use correct imports between files.

5. Do not use fake functions.

6. Do not leave TODO placeholders.

7. Include proper error handling.

8. Include requirements.txt.

9. Include README.md.

10. Make the project internally consistent.

11. Use the APIs and frameworks
    described in the tutorial.

12. Return every file with:

    path
    content

13. The content must be complete.

14. Do not summarize the files.

15. Do not omit any requested file.

16. Use relative paths.

17. Create __init__.py where needed.

18. Make imports consistent with
    the generated directory structure.
"""

    logger.write_code(
        step_file,
        "LLM Prompt",
        prompt,
        "text",
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    structured_llm = (
        llm.with_structured_output(
            GeneratedProject
        )
    )

    try:

        result = structured_llm.invoke(
            prompt
        )

    except Exception as e:

        logger.write_code(
            step_file,
            "ERROR",
            f"{type(e).__name__}: {e}",
            "text",
        )

        raise

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    files = {}

    print()

    for generated_file in result.files:

        path = generated_file.path

        content = generated_file.content

        if not path:

            continue

        if not content:

            logger.write(
                step_file,
                "WARNING",
                f"Empty file content: `{path}`",
            )

            continue

        files[path] = content

        print(
            f"   ✅ Generated: {path}"
        )

        # ----------------------------------------------------
        # SAVE FULL GENERATED FILE TO STEP MD
        # ----------------------------------------------------

        logger.write_code(
            step_file,
            f"Generated File: {path}",
            content,
            "python",
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    logger.write(
        step_file,
        "Generation Summary",
        f"**Files generated:** {len(files)}",
    )

    if not files:

        logger.write_code(
            step_file,
            "ERROR",
            "LLM returned ZERO usable files.",
            "text",
        )

        raise RuntimeError(
            "LLM returned zero usable files."
        )

    print()
    print(
        f"✅ Total generated: "
        f"{len(files)} files"
    )

    return {
        "files": files
    }


# ============================================================
# STEP 4
# REVIEW PROJECT
# ============================================================

def review_project(
    state: State,
):

    print()
    print("=" * 70)
    print(
        "🔎 STEP 4: REVIEWING PROJECT"
    )
    print("=" * 70)

    step_file = logger.create_step(
        4,
        "review_project",
    )

    # --------------------------------------------------------
    # Build project text
    # --------------------------------------------------------

    project_text = ""

    for path, content in state[
        "files"
    ].items():

        project_text += f"""

==================================================
FILE: {path}
==================================================

{content}

"""

    # --------------------------------------------------------
    # Save project being reviewed
    # --------------------------------------------------------

    logger.write_code(
        step_file,
        "Generated Project Being Reviewed",
        project_text,
        "text",
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a senior Python code reviewer.

Review this complete generated project.

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
13. Inconsistent code
14. requirements.txt
15. README.md
16. Package structure

If the project is correct, say:

PROJECT_VALID

Otherwise explain every problem
that must be fixed.

PROJECT:

{project_text}
"""

    logger.write_code(
        step_file,
        "Review Prompt",
        prompt,
        "text",
    )

    # --------------------------------------------------------
    # Review
    # --------------------------------------------------------

    try:

        response = llm.invoke(
            prompt
        )

    except Exception as e:

        logger.write_code(
            step_file,
            "ERROR",
            f"{type(e).__name__}: {e}",
            "text",
        )

        raise

    review = response.content

    # --------------------------------------------------------
    # Save review
    # --------------------------------------------------------

    logger.write_code(
        step_file,
        "Review Result",
        review,
        "text",
    )

    print()
    print(
        review
    )

    return {
        "review": review
    }


# ============================================================
# STEP 5
# SAVE PROJECT
# ============================================================

def save_project(
    state: State,
):

    print()
    print("=" * 70)
    print(
        "💾 STEP 5: SAVING PROJECT"
    )
    print("=" * 70)

    step_file = logger.create_step(
        5,
        "save_project",
    )

    project_path = Path(
        state["output_path"]
    ).resolve()

    project_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = state.get(
        "files",
        {},
    )

    # --------------------------------------------------------
    # Save destination
    # --------------------------------------------------------

    logger.write(
        step_file,
        "Output Directory",
        f"`{project_path}`",
    )

    logger.write(
        step_file,
        "Files To Save",
        "\n".join(
            f"- `{path}`"
            for path in files
        ),
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not files:

        logger.write_code(
            step_file,
            "ERROR",
            "No generated files exist.",
            "text",
        )

        raise RuntimeError(
            "No files available to save."
        )

    saved_files = []

    # --------------------------------------------------------
    # Save every generated file
    # --------------------------------------------------------

    for relative_path, content in files.items():

        relative_path = Path(
            relative_path
        )

        # ----------------------------------------------------
        # Prevent absolute paths
        # ----------------------------------------------------

        if relative_path.is_absolute():

            logger.write(
                step_file,
                "Skipped Unsafe File",
                f"`{relative_path}` is absolute.",
            )

            continue

        file_path = (
            project_path /
            relative_path
        ).resolve()

        # ----------------------------------------------------
        # Prevent ../ path traversal
        # ----------------------------------------------------

        try:

            file_path.relative_to(
                project_path
            )

        except ValueError:

            logger.write(
                step_file,
                "Skipped Unsafe File",
                f"`{relative_path}` escapes project directory.",
            )

            continue

        # ----------------------------------------------------
        # Create directory
        # ----------------------------------------------------

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Write
        # ----------------------------------------------------

        file_path.write_text(
            content,
            encoding="utf-8",
        )

        saved_files.append(
            relative_path.as_posix()
        )

        print(
            f"   📄 Saved: {file_path}"
        )

        logger.write(
            step_file,
            "Saved File",
            f"✅ `{relative_path}`",
        )

    # --------------------------------------------------------
    # Save analysis
    # --------------------------------------------------------

    analysis_file = (
        project_path /
        "analysis.json"
    )

    analysis_file.write_text(
        json.dumps(
            {
                "analysis":
                    state["analysis"]
            },
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.write(
        step_file,
        "Saved Metadata",
        "✅ `analysis.json`",
    )

    # --------------------------------------------------------
    # Save structure
    # --------------------------------------------------------

    structure_file = (
        project_path /
        "project_structure.json"
    )

    structure_file.write_text(
        json.dumps(
            state[
                "project_structure"
            ],
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.write(
        step_file,
        "Saved Metadata",
        "✅ `project_structure.json`",
    )

    # --------------------------------------------------------
    # Save review
    # --------------------------------------------------------

    review_file = (
        project_path /
        "review.txt"
    )

    review_file.write_text(
        state.get(
            "review",
            "",
        ),
        encoding="utf-8",
    )

    logger.write(
        step_file,
        "Saved Metadata",
        "✅ `review.txt`",
    )

    # --------------------------------------------------------
    # Verify actual disk
    # --------------------------------------------------------

    actual_files = []

    for path in project_path.rglob("*"):

        if path.is_file():

            actual_files.append(
                path
            )

    disk_files = []

    for path in sorted(
        actual_files
    ):

        relative = path.relative_to(
            project_path
        )

        disk_files.append(
            f"- `{relative}`"
        )

    logger.write(
        step_file,
        "Final Files On Disk",
        "\n".join(
            disk_files
        ),
    )

    logger.write(
        step_file,
        "Final Summary",
        f"""
**Generated files:** {len(files)}

**Saved files:** {len(saved_files)}

**Files physically on disk:** {len(actual_files)}

**Project location:** `{project_path}`
""",
    )

    print()
    print(
        f"✅ {len(saved_files)} generated files saved"
    )

    print(
        f"📁 Location: {project_path}"
    )

    return {
        "output_path":
            str(project_path),

        "saved_files":
            len(saved_files),
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        State
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    graph.add_node(
        "analyze",
        analyze_transcript,
    )

    graph.add_node(
        "design_project",
        design_project,
    )

    graph.add_node(
        "generate_files",
        generate_files,
    )

    graph.add_node(
        "review",
        review_project,
    )

    graph.add_node(
        "save_project",
        save_project,
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "analyze",
    )

    graph.add_edge(
        "analyze",
        "design_project",
    )

    graph.add_edge(
        "design_project",
        "generate_files",
    )

    graph.add_edge(
        "generate_files",
        "review",
    )

    graph.add_edge(
        "review",
        "save_project",
    )

    graph.add_edge(
        "save_project",
        END,
    )

    return graph.compile()


# ============================================================
# RUN GRAPH
# ============================================================

def run_graph(
    app,
    transcript: str,
    output_path: str,
):

    initial_state = {

        "transcript":
            transcript,

        "analysis":
            "",

        "project_structure":
            {},

        "files":
            {},

        "review":
            "",

        "output_path":
            output_path,

        "saved_files":
            0,
    }

    print()
    print("=" * 70)
    print(
        "🚀 YOUTUBE → PYTHON PROJECT AGENT"
    )
    print("=" * 70)

    print()
    print("START")
    print("  ↓")

    final_state = (
        initial_state.copy()
    )

    try:

        for event in app.stream(
            initial_state,
            stream_mode="updates",
        ):

            for node_name, output in event.items():

                print()
                print(
                    f"▶️ RUNNING NODE: "
                    f"{node_name}"
                )

                if node_name == "analyze":

                    print(
                        "   ✓ Tutorial understood"
                    )

                elif node_name == "design_project":

                    structure = output.get(
                        "project_structure",
                        {},
                    )

                    planned_files = (
                        structure.get(
                            "files",
                            [],
                        )
                    )

                    print(
                        f"   ✓ "
                        f"{len(planned_files)} "
                        f"files planned"
                    )

                elif node_name == "generate_files":

                    generated = output.get(
                        "files",
                        {},
                    )

                    print(
                        f"   ✓ "
                        f"{len(generated)} "
                        f"files generated"
                    )

                elif node_name == "review":

                    print(
                        "   ✓ Project reviewed"
                    )

                elif node_name == "save_project":

                    print(
                        f"   ✓ Saved "
                        f"{output.get('saved_files', 0)} "
                        f"files"
                    )

                final_state.update(
                    output
                )

                print(
                    "  ↓"
                )

    except Exception as e:

        print()
        print("=" * 70)
        print(
            "❌ WORKFLOW FAILED"
        )
        print("=" * 70)

        print(
            f"{type(e).__name__}: {e}"
        )

        # Create failure log
        error_file = (
            STEPS_PATH /
            "99_workflow_error.md"
        )

        error_file.write_text(
            f"""# Workflow Error
            **Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            ---
            ## Error Type

            `{type(e).__name__}`
            ## Error
            ```text{e}
