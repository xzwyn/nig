!apt-get install graphviz -y
!pip install graphviz
# In a new Colab cell, run this to generate the high-level diagram.
from graphviz import Digraph

# Create a new directed graph
dot = Digraph('FinanzAgent_High_Level_Flow', comment='High-Level Application Flow')
dot.attr(rankdir='TB', label='Diagram 1: High-Level Application Flow', fontsize='20')
dot.attr('node', shape='box', style='rounded,filled', fillcolor='skyblue', fontname='Helvetica')
dot.attr('edge', fontname='Helvetica')

# Define the nodes
dot.node('start', 'Start: User Uploads PDF', shape='oval', fillcolor='lightgreen')
dot.node('parse', 'Parse PDF with Unstructured.io\n(strategy="hi_res")', width='4')
dot.node('filter', 'Filter and Create List of Elements')

# Use a subgraph for the loop to visually group them
with dot.subgraph(name='cluster_loop') as c:
    c.attr(label='For each element in List:', style='dashed', color='grey')
    c.node('decision_table', 'Is element a "Table"?', shape='diamond', fillcolor='khaki')
    c.node('process_table', 'Translate Table\n(Use BeautifulSoup & invoke Agent per cell)')
    c.node('process_text', 'Translate Text Element\n(Invoke Agent on element.text)')
    c.node('invoke_agent', 'Invoke Translation Agent (LangGraph)\n(See Diagram 2 for details)', shape='box', style='rounded,filled', fillcolor='orange', peripheries='2')
    c.node('append_result', 'Append Translated HTML to Results List', shape='cylinder', fillcolor='lightgrey')
    
    # Edges within the loop
    c.edge('decision_table', 'process_table', label='Yes')
    c.edge('decision_table', 'process_text', label='No')
    c.edge('process_table', 'invoke_agent')
    c.edge('process_text', 'invoke_agent')
    c.edge('invoke_agent', 'append_result')


dot.node('join_parts', 'Join All Translated HTML Parts')
dot.node('display', 'Display Final Assembled Document in UI', shape='parallelogram', fillcolor='lightblue')
dot.node('end', 'End', shape='oval', fillcolor='lightcoral')

# Define the main flow edges
dot.edge('start', 'parse')
dot.edge('parse', 'filter')
dot.edge('filter', 'decision_table', lhead='cluster_loop') # Arrow points to the loop cluster
dot.edge('append_result', 'join_parts', ltail='cluster_loop') # Arrow comes from the loop cluster
dot.edge('join_parts', 'display')
dot.edge('display', 'end')

# Render the diagram in Colab
dot


# Cell 1: Installation
# Install the system dependency for graph visualization
!apt-get install -y graphviz

# Install the required Python libraries
!pip install langgraph langchain-ollama pygraphviz
# Cell 2: Define Mock Tools

# These are placeholder functions that mimic our tools.py file for visualization purposes.
# They have the same names and inputs but don't perform any real work.

def translate_with_helsinki(text: str) -> str:
    """Mock: Translates English text to German."""
    print("Called translate_with_helsinki")
    return "mock german translation"

def grade_translation(original_text: str, translated_text: str) -> dict:
    """Mock: Grades a translation."""
    print("Called grade_translation")
    # We can simulate different outcomes by changing the return value here
    return {"grade": "REJECT", "critique": "Mock critique"}

def retranslate_with_feedback(original_text: str, failed_translation: str, critique: str) -> str:
    """Mock: Attempts a corrected translation."""
    print("Called retranslate_with_feedback")
    return "mock corrected german translation"

def choose_best_translation(original_text: str, translation_a: str, translation_b: str) -> str:
    """Mock: Chooses the better of two translations."""
    print("Called choose_best_translation")
    return "mock best translation"

print("✅ Mock tools defined successfully.")
# Cell 3: Define the LangGraph Agent

from typing import List, TypedDict
from langgraph.graph import StateGraph, END

# This is the exact code from our translation_agent.py file

# 1. Define the State
class GraphState(TypedDict):
    original_text: str
    translation_history: List[str]
    critique_history: List[str]
    retry_count: int
    final_translation: str
    grade: str 

# 2. Define the Nodes
def translate_node(state: GraphState):
    original = state['original_text']
    translation = translate_with_helsinki(original)
    return {"translation_history": [translation], "retry_count": 0}

def grade_node(state: GraphState):
    original = state['original_text']
    latest_translation = state['translation_history'][-1]
    grade_result = grade_translation(original, latest_translation)
    return {"critique_history": state['critique_history'] + [grade_result['critique']], "grade": grade_result['grade']}

def retranslate_node(state: GraphState):
    original = state['original_text']
    failed_translation = state['translation_history'][-1]
    critique = state['critique_history'][-1]
    corrected_translation = retranslate_with_feedback(original, failed_translation, critique)
    return {"translation_history": state['translation_history'] + [corrected_translation], "retry_count": state['retry_count'] + 1}

def choose_best_node(state: GraphState):
    original = state['original_text']
    first_translation = state['translation_history'][0]
    last_translation = state['translation_history'][-1]
    best_translation = choose_best_translation(original, first_translation, last_translation)
    return {"final_translation": best_translation}
    
def finalize_node(state: GraphState):
    final_translation = state['translation_history'][-1]
    return {"final_translation": final_translation}

# 3. Define the Edges
def should_continue(state: GraphState):
    grade = state.get('grade')
    retries = state['retry_count']
    if grade == "APPROVE":
        return "finalize"
    elif retries >= 3:
        return "choose_best"
    else:
        return "retranslate"

# 4. Assemble the Graph
def create_graph():
    graph = StateGraph(GraphState)
    graph.add_node("translate", translate_node)
    graph.add_node("grade", grade_node)
    graph.add_node("retranslate", retranslate_node)
    graph.add_node("choose_best", choose_best_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("translate")
    graph.add_edge("choose_best", END)
    graph.add_edge("finalize", END)
    graph.add_edge("translate", "grade")
    graph.add_edge("retranslate", "grade")
    graph.add_conditional_edges(
        "grade",
        should_continue,
        {"retranslate": "retranslate", "choose_best": "choose_best", "finalize": "finalize"}
    )
    return graph.compile()

print("✅ Agent graph structure defined successfully.")
# Cell 4: Generate and Display the Diagram

from IPython.display import Image, display

# Create the compiled graph object
agent_graph = create_graph()

# Generate the diagram as a PNG image in bytes
png_bytes = agent_graph.get_graph().draw_png()

# Display the image in the Colab output
print("Displaying the agent graph:")
display(Image(png_bytes))

# Optionally, save the diagram to a file so you can download it
output_filename = "finanzagent_graph.png"
with open(output_filename, "wb") as f:
    f.write(png_bytes)
print(f"\nGraph saved to {output_filename}. You can download it from the Colab file browser (folder icon on the left).")
