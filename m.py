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
import graphviz

# Create a new directed graph
dot = graphviz.Digraph(comment='FinanzAgent - Detailed Agent Flow')

# --- Define Global Graph Attributes for a clean, professional look ---
dot.attr('graph', rankdir='TB', splines='ortho', bgcolor='transparent', label='Detailed Agent (LangGraph) Flow', fontname='Arial', fontsize='14', fontcolor='#333333')
dot.attr('node', shape='box', style='filled,rounded', fillcolor='#E6F0FF', fontname='Arial', fontsize='11', color='#6C8EBF')
dot.attr('edge', fontname='Arial', fontsize='9', color='#555555')

# --- Define Nodes (Steps in the process) ---
# We use different shapes and colors for clarity

# Start and End nodes
dot.node('start', 'Agent Invoked with\noriginal_text', shape='ellipse', fillcolor='#D5E8D4', color='#82B366')
dot.node('end', 'End Agent:\nReturn final_translation', shape='ellipse', fillcolor='#D5E8D4', color='#82B336')

# Processing nodes
dot.node('translate', 'NODE: TRANSLATE\nTool: translate_with_helsinki()')
dot.node('retranslate', 'NODE: RETRANSLATE\nTool: retranslate_with_feedback()')
dot.node('choose_best', 'NODE: CHOOSE_BEST\nTool: choose_best_translation()')
dot.node('finalize', 'NODE: FINALIZE\nSet final_translation')

# Decision node (visually distinct)
dot.node('grade', 'NODE: GRADE\nTool: grade_translation()', shape='diamond', fillcolor='#FFF2CC', color='#D6B656')


# --- Define Edges (Connections and Logic Flow) ---
dot.edge('start', 'translate')
dot.edge('translate', 'grade')

# Decision paths from the 'GRADE' node
dot.edge('grade', 'finalize', label='Grade == "APPROVE"', color='#2E7D32', fontcolor='#2E7D32', penwidth='1.5')
dot.edge('grade', 'retranslate', label='Grade == "REJECT"\n& retries < 3', color='#C62828', fontcolor='#C62828', penwidth='1.5')
dot.edge('grade', 'choose_best', label='Grade == "REJECT"\n& retries >= 3', color='#EF6C00', fontcolor='#EF6C00', penwidth='1.5')

# The main correction loop
dot.edge('retranslate', 'grade', label='Submit for re-grading', style='dashed')

# Final paths to the end state
dot.edge('finalize', 'end')
dot.edge('choose_best', 'end')

# --- Render the Diagram in the Colab output ---
dot
