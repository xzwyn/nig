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
