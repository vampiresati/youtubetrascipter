from typing import TypedDict
from pathlib import Path

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
# State
# ============================================================

class State(TypedDict):
    transcript: str
    analysis: str
    code: str
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

    print("\n" + "=" * 60)
    print("🔍 STEP 1: ANALYZING TRANSCRIPT")
    print("=" * 60)

    prompt = f"""
You are a senior Python architect.

Analyze the following YouTube tutorial transcript.

Extract:

1. What project is being built
2. Required Python packages
3. Important classes
4. Important functions
5. Data structures
6. Architecture
7. Workflow
8. Important implementation details

Do NOT generate code yet.

Transcript:
----------------------------

{state["transcript"]}

----------------------------

Provide a detailed technical analysis.
"""

    response = llm.invoke(prompt)

    print("✅ Transcript analysis completed")

    return {
        "analysis": response.content
    }


# ============================================================
# Node 2: Generate Python Code
# ============================================================

def generate_code(state: State):

    print("\n" + "=" * 60)
    print("💻 STEP 2: GENERATING PYTHON CODE")
    print("=" * 60)

    prompt = f"""
You are an expert Python developer.

Based on the following tutorial analysis,
generate the complete Python implementation.

ANALYSIS:
----------------------------

{state["analysis"]}

----------------------------

Requirements:

- Write real runnable Python code.
- Use modern Python.
- Include imports.
- Include error handling.
- Include comments.
- Include required packages.
- Do not leave TODO placeholders.
- If multiple files are required, clearly identify them.
- Follow the architecture described in the analysis.

Return the complete implementation.
"""

    response = llm.invoke(prompt)

    print("✅ Python code generated")

    return {
        "code": response.content
    }


# ============================================================
# Node 3: Review Code
# ============================================================

def review_code(state: State):

    print("\n" + "=" * 60)
    print("🔎 STEP 3: REVIEWING GENERATED CODE")
    print("=" * 60)

    prompt = f"""
You are a senior Python code reviewer.

Review the following generated code.

CODE:
----------------------------

{state["code"]}

----------------------------

Check for:

- Syntax errors
- Missing imports
- Incorrect APIs
- Missing dependencies
- Logical errors
- Incomplete implementation
- Problems that would prevent execution
- Incorrect Python syntax
- Bad LangChain/LangGraph APIs if used

Fix all problems you find.

Return ONLY the corrected Python implementation.
"""

    response = llm.invoke(prompt)

    print("✅ Code review completed")

    return {
        "review": response.content
    }


# ============================================================
# Node 4: Save Code
# ============================================================

def save_code(state: State):

    print("\n" + "=" * 60)
    print("💾 STEP 4: SAVING CODE")
    print("=" * 60)

    output_path = Path(
        state["output_path"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    code = state["review"]

    # Remove markdown code fences if Ollama returns them
    if code.startswith("```python"):

        code = code[len("```python"):]

    elif code.startswith("```"):

        code = code[len("```"):]

    if code.endswith("```"):

        code = code[:-3]

    code = code.strip()

    output_path.write_text(
        code,
        encoding="utf-8"
    )

    print(f"✅ Code saved:")
    print(f"   {output_path}")

    return {
        "output_path": str(output_path)
    }


# ============================================================
# Build LangGraph
# ============================================================

def build_graph():

    graph = StateGraph(State)

    # Nodes
    graph.add_node(
        "analyze",
        analyze_transcript
    )

    graph.add_node(
        "generate_code",
        generate_code
    )

    graph.add_node(
        "review",
        review_code
    )

    graph.add_node(
        "save_code",
        save_code
    )

    # Flow
    graph.add_edge(
        START,
        "analyze"
    )

    graph.add_edge(
        "analyze",
        "generate_code"
    )

    graph.add_edge(
        "generate_code",
        "review"
    )

    graph.add_edge(
        "review",
        "save_code"
    )

    graph.add_edge(
        "save_code",
        END
    )

    return graph.compile()


# ============================================================
# Run Graph with Live Progress
# ============================================================

def run_graph(
    app,
    transcript: str,
    output_path: str
):

    initial_state = {

        "transcript": transcript,

        "analysis": "",

        "code": "",

        "review": "",

        "output_path": output_path
    }

    print("\n")
    print("=" * 60)
    print("🚀 STARTING YOUTUBE → PYTHON CODE AGENT")
    print("=" * 60)

    print("\nSTART")
    print("  ↓")

    final_state = None

    for event in app.stream(
        initial_state,
        stream_mode="updates"
    ):

        for node_name, output in event.items():

            print(
                f"▶️  NODE: {node_name}"
            )

            if node_name == "analyze":

                print(
                    "    ✓ Transcript analyzed"
                )

            elif node_name == "generate_code":

                print(
                    "    ✓ Python code generated"
                )

            elif node_name == "review":

                print(
                    "    ✓ Code reviewed and corrected"
                )

            elif node_name == "save_code":

                print(
                    "    ✓ Code written to disk"
                )

            print("  ↓")

            final_state = output

    print("END")

    print("\n" + "=" * 60)
    print("🎉 WORKFLOW COMPLETED")
    print("=" * 60)

    return final_state


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Your transcript
    # --------------------------------------------------------

    transcript = """
    Today we are going to build a simple Python AI agent.

    The agent will use an Ollama model and LangGraph.

    First we create a StateGraph.

    Then we create an agent node.

    Finally we execute the graph and return the result.
    """

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    app = build_graph()

    print("\n✅ LangGraph compiled successfully.")

    # --------------------------------------------------------
    # Save graph PNG
    # --------------------------------------------------------

    graph_path = save_graph_png(
        app,
        "/home/satvir/youtubetrascipter/langgraph.png"
    )

    print(
        f"\n📊 Graph saved to:"
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
            "Running outside Jupyter - "
            "graph image saved successfully."
        )

    # --------------------------------------------------------
    # Run LangGraph
    # --------------------------------------------------------

    result = run_graph(

        app,

        transcript,

        "/home/satvir/Downloads/TestCreation/main.py"
    )

    print("\nFinal result:")
    print(result)
