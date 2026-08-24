from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pathlib import Path
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)

class State(TypedDict):
    transcript: str
    analysis: str
    code: str
    review: str
    output_path: str

def save_graph_png(app, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    png_bytes = app.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_bytes)
    return str(output_path)

def save_code(state: State):
    output_path = Path(state["output_path"])
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(state["review"],encoding="utf-8")
    return {
        "output_path": str(output_path)
        }
def analyze_transcript(state: State):
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

{state["transcript"]}
"""

    response = llm.invoke(prompt)

    return {
        "analysis": response.content
    }


def generate_code(state: State):

    prompt = f"""
You are an expert Python developer.

Based on the following tutorial analysis,
generate the complete Python implementation.

ANALYSIS:
{state["analysis"]}

Requirements:

- Write real runnable Python code.
- Use modern Python.
- Include imports.
- Include error handling.
- Include comments.
- Include required packages.
- Do not leave TODO placeholders.
- If multiple files are required, clearly identify them.

Return the implementation.
"""

    response = llm.invoke(prompt)

    return {
        "code": response.content
    }


def review_code(state: State):

    prompt = f"""
You are a senior Python code reviewer.

Review the following generated code.

CODE:
{state["code"]}

Check for:

- Syntax errors
- Missing imports
- Incorrect APIs
- Missing dependencies
- Logical errors
- Incomplete implementation
- Problems that would prevent execution

Then provide corrected code.

Return the corrected implementation.
"""

    response = llm.invoke(prompt)

    return {
        "review": response.content
    }


def build_graph():
    graph = StateGraph(State)
    graph.add_node("analyze", analyze_transcript)
    graph.add_node("generate_code", generate_code)
    graph.add_node("review", review_code)
    graph.add_node("save_code",save_code)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "generate_code")
    graph.add_edge("generate_code", "review")
    graph.add_edge("review","save_code")
    graph.add_edge("save_code",END)
    return graph.compile()




if __name__ == "__main__":
    app = build_graph()
    from IPython.display import Image, display
    graph_path = save_graph_png(
        app,
        "/home/satvir/youtubetrascipter/langgraph.png"
    )

    print(f"Graph saved to: {graph_path}")
    display(
        Image(
            graph_path
        )
    )
#     result = app.invoke({
#         "transcript": transcript,
#         "analysis": "",
#         "code": "",
#         "review": "",
#         "output_path": "/home/satvir/Downloads/TestCreation/"
#     })
