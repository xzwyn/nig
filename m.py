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

English:
A core pillar of our sustainability ambition is to follow clear, transparent practices and provide high-quality, verifiable reporting that reflects our ongoing commitment to measurable sustainability outcomes. Our Sustainability Statement outlines the sustainability matters material for Allianz and how we address them. We adhere to the European Sustainability Reporting Standards (ESRS) structure, which focuses on key sustainability aspects (environment, social, and governance) and provides a robust disclosure framework for comparability and credibility. In alignment with these standards, we have structured our Sustainability Statement in key sections, as detailed in the following paragraphs
German:
Ein zentraler Pfeiler unserer Nachhaltigkeitsambition ist es, klare, transparente Praktiken zu verfolgen und hochwertige, verifizierbare Berichterstattung zu liefern, die unser fortlaufendes Engagement für messbare Nachhaltigkeitsziele widerspiegelt. Unser Nachhaltigkeitsstatement umfasst die für Allianz wesentlichen Nachhaltigkeitsthemen und beschreibt, wie wir sie angehen. Wir orientieren uns an der Struktur der Europäischen Nachhaltigkeitsberichterstattungsstandards (ESRS), die sich auf die zentralen Nachhaltigkeitsaspekte (Umwelt, Soziales und Governance) konzentriert und einen robusten Offenlegungsrahmen für Vergleichbarkeit und Glaubwürdigkeit bietet. In Übereinstimmung mit diesen Standards haben wir unser Nachhaltigkeitsstatement in Schlüsselabschnitte gegliedert, wie in den folgenden Absätzen erläutert.
Omitted_lines = 0, [ ]
Mistranslated lines = 2, [“A core pillar of our sustainability ambition is to follow clear, transparent practices and provide high-quality, verifiable reporting that reflects our ongoing commitment to measurable sustainability outcomes.” issue with core pillar as zentraler Pfeiler, “We adhere to the European Sustainability Reporting Standards (ESRS) structure, which focuses on key sustainability aspects (environment, social, and governance) and provides a robust disclosure framework for comparability and credibility.”  'Nachhaltigkeitsberichterstattungsstandards' is closer to the English term 'sustainability reporting standards'.]

English:
A core pillar of our sustainability ambition is to follow clear, transparent practices and provide high-quality, verifiable reporting that reflects our ongoing commitment to measurable sustainability outcomes. Our Sustainability Statement outlines the sustainability matters material for Allianz and how we address them. We adhere to the European Sustainability Reporting Standards (ESRS) structure, which focuses on key sustainability aspects (environment, social, and governance) and provides a robust disclosure framework for comparability and credibility. In alignment with these standards, we have structured our Sustainability Statement in key sections, as detailed in the following paragraphs.
German:
Ein Kernpfeiler unseres Nachhaltigkeits-Ehrgeizes ist es, klare, transparente Praktiken zu verfolgen und qualitativ hochwertige, überprüfbare Berichterstattung zu liefern, die unser anhaltendes Engagement für messbare Nachhaltigkeitsergebnisse widerspiegelt. In unserem Nachhaltigkeits-Statement werden die Nachhaltigkeitsthemen der Allianz und ihre Adressierung erläutert. Wir halten uns an die Struktur der European Sustainability Reporting Standards (ESRS), die sich auf wichtige Nachhaltigkeitsaspekte (Umwelt, Soziales und Governance) konzentriert und einen robusten Offenlegungsrahmen für Vergleichbarkeit und Glaubwürdigkeit bietet. In Übereinstimmung mit diesen Standards haben wir unser Nachhaltigkeits-Statement in wichtigen Abschnitten strukturiert, wie in den folgenden Abschnitten beschrieben.
Omitted_lines = 0, [ ]
Mistranslated lines = 1, [“A core pillar of our sustainability ambition is to follow clear, transparent practices and provide high-quality, verifiable reporting that reflects our ongoing commitment to measurable sustainability outcomes.” German word for 'pillar' is 'Pfeiler', not 'kerne'] 


English:
Comparative information
Comparatives are disclosed only if published in the Allianz Group
Annual Report 2023 or the Allianz Group Sustainability Report 2023.
Comparatives from the Allianz Group Sustainability Report 2023 were
assured with limited assurance, and respective columns in the
disclosure tables are marked with an asterisk. If no data was disclosed
previously, “n. a.” is disclosed for the comparison period data.
Incorporation by reference
We incorporate information into our Sustainability Statement
prescribed by an ESRS disclosure requirement, including specific
datapoints, also by reference. We ensure incorporation by reference
does not impair the readability of our Sustainability Statement and
considers the overall cohesiveness of the reported information.
Incorporation by reference (BP-2)
Disclosure requirement	Reference	Sustainability Statement section
BP-1.5 (b) ii.	8.20 List of participations of the Allianz Group as of 31 December 2024 according to § 313 (2) HGB	Basis for preparation
SBM-1.40 (a)		Allianz business model and value chain
SBM-1.42 (a)-(c)	Business Operations	
GOV-1.20 (a)		Role of administrative, management, and supervisory bodies
GOV-1.21 (a)-(d)	Corporate Governance Statement	
GOV-1.21 (e)		
GOV-3.29 (a)-(e)		Integration of sustainability-related performance in incentive schemes (e) GOV-3.29 (a)-(e) Remuneration Report Integration of sustainabilityrelated performance in incentive GOV schemes -3.29 AR7
German:
Vergleichende Informationen
"Vergleiche werden nur dann offengelegt, wenn sie im Allianz-Konzern-Geschäftsbericht 2023 oder im Allianz-Konzern-Nachhaltigkeitsbericht 2023 veröffentlicht werden. Vergleiche aus dem Nachhaltigkeitsbericht 2023 wurden mit begrenzter Sicherheit gewährleistet, und die entsprechenden Spalten in den Offenlegungstabellen sind mit einem Sternchen gekennzeichnet."
Einschließlich durch Bezugnahme
Wir integrieren Informationen in unsere Nachhaltigkeitserklärung, die durch eine ESRS-Publikationspflicht, einschließlich bestimmter Datenpunkte, auch durch Verweise vorgeschrieben ist. Wir stellen sicher, dass die Einbeziehung per Verweis die Lesbarkeit unserer Nachhaltigkeitserklärung nicht beeinträchtigt und die Gesamtzusammenhaltbarkeit der gemeldeten Informationen berücksichtigt.
Einschließlich durch Verweis (BP-2)
Offenlegungspflicht	Sachgebietsnummer	Abschnitt Nachhaltigkeitserklärung
BP-1, 5 b) ii.	8.20 Liste der Beteiligungen der Allianz Gruppe zum 31. Dezember 2024 nach § 313 (2) HGB	Grundlage für die Vorbereitung
SBM-1.40 (a)		Allianz-Geschäftsmodell und
SBM-1.42 (a)-(c)	Geschäftstätigkeit	Wertschöpfungskette
GOV-1.20 (a)		Rolle der Verwaltung,
GOV-1.21 (a)-(d)	Erklärung zur Corporate Governance	Management und Aufsicht
GOV-1.21 (e)		Einrichtungen
GOV-3.29 (a)-(e)		Integration der Nachhaltigkeit
GOV-3.29 AR7	Vergütungsbericht	damit verbundene Leistung bei Anreizregelungen

Omitted_lines = 0, [ ]
Mistranslated lines = 1, [“We ensure incorporation by reference does not impair the readability of our Sustainability Statement and considers the overall cohesiveness of the reported information.”   The translation of 'considers the overall cohesiveness' might be more accurately expressed as, 'considering the overall coherence or consistency , “Comparatives from the Allianz Group Sustainability Report 2023 were assured with limited assurance, and respective columns in the disclosure tables are marked with an asterisk.”  'with limited assurance' is not typically used in this context. It might be more accurate to say, 'with limited assurance or scope.' This phrase usually indicates that the audit scope was limited for reasons such as time constraints or the nature of the entity.]


Model 3:
English:
A core pillar of our sustainability ambition is to follow clear, transparent practices and provide high-quality, verifiable reporting that reflects our ongoing commitment to measurable sustainability outcomes. Our Sustainability Statement outlines the sustainability matters material for Allianz and how we address them. We adhere to the European Sustainability Reporting Standards (ESRS) structure, which focuses on key sustainability aspects (environment, social, and governance) and provides a robust disclosure framework for comparability and credibility. In alignment with these standards, we have structured our Sustainability Statement in key sections, as detailed in the following paragraphs.
German:
Ein Kernpfeiler unseres Nachhaltigkeits-Ehrgeizes ist es, klare, transparente Praktiken zu verfolgen und qualitativ hochwertige, überprüfbare Berichterstattung zu liefern, die unser anhaltendes Engagement für messbare Nachhaltigkeitsergebnisse widerspiegelt. In unserem Nachhaltigkeits-Statement werden die Nachhaltigkeitsthemen der Allianz und ihre Adressierung erläutert. Wir halten uns an die Struktur der European Sustainability Reporting Standards (ESRS), die sich auf wichtige Nachhaltigkeitsaspekte (Umwelt, Soziales und Governance) konzentriert und einen robusten Offenlegungsrahmen für Vergleichbarkeit und Glaubwürdigkeit bietet. In Übereinstimmung mit diesen Standards haben wir unser Nachhaltigkeits-Statement in wichtigen Abschnitten strukturiert, wie in den folgenden Abschnitten beschrieben.
Omitted_lines = 0, [ ]
Mistranslated lines = 0, [ ] 

English:
Comparative information
Comparatives are disclosed only if published in the Allianz Group
Annual Report 2023 or the Allianz Group Sustainability Report 2023.
Comparatives from the Allianz Group Sustainability Report 2023 were
assured with limited assurance, and respective columns in the
disclosure tables are marked with an asterisk. If no data was disclosed
previously, “n. a.” is disclosed for the comparison period data.
Incorporation by reference
We incorporate information into our Sustainability Statement
prescribed by an ESRS disclosure requirement, including specific
datapoints, also by reference. We ensure incorporation by reference
does not impair the readability of our Sustainability Statement and
considers the overall cohesiveness of the reported information.
Incorporation by reference (BP-2)
Disclosure requirement	Reference	Sustainability Statement section
BP-1.5 (b) ii.	8.20 List of participations of the Allianz Group as of 31 December 2024 according to § 313 (2) HGB	Basis for preparation
SBM-1.40 (a)		Allianz business model and value chain
SBM-1.42 (a)-(c)	Business Operations	
GOV-1.20 (a)		Role of administrative, management, and supervisory bodies
GOV-1.21 (a)-(d)	Corporate Governance Statement	
GOV-1.21 (e)		
GOV-3.29 (a)-(e)		Integration of sustainability-related performance in incentive schemes (e) GOV-3.29 (a)-(e) Remuneration Report Integration of sustainabilityrelated performance in incentive GOV schemes -3.29 AR7
German:
Vergleichende Informationen
"Vergleiche werden nur dann offengelegt, wenn sie im Allianz-Konzern-Geschäftsbericht 2023 oder im Allianz-Konzern-Nachhaltigkeitsbericht 2023 veröffentlicht werden. Vergleiche aus dem Nachhaltigkeitsbericht 2023 wurden mit begrenzter Sicherheit gewährleistet, und die entsprechenden Spalten in den Offenlegungstabellen sind mit einem Sternchen gekennzeichnet."
Einschließlich durch Bezugnahme
Wir integrieren Informationen in unsere Nachhaltigkeitserklärung, die durch eine ESRS-Publikationspflicht, einschließlich bestimmter Datenpunkte, auch durch Verweise vorgeschrieben ist. Wir stellen sicher, dass die Einbeziehung per Verweis die Lesbarkeit unserer Nachhaltigkeitserklärung nicht beeinträchtigt und die Gesamtzusammenhaltbarkeit der gemeldeten Informationen berücksichtigt.
Einschließlich durch Verweis (BP-2)
Offenlegungspflicht	Sachgebietsnummer	Abschnitt Nachhaltigkeitserklärung
BP-1, 5 b) ii.	8.20 Liste der Beteiligungen der Allianz Gruppe zum 31. Dezember 2024 nach § 313 (2) HGB	Grundlage für die Vorbereitung
SBM-1.40 (a)		Allianz-Geschäftsmodell und
SBM-1.42 (a)-(c)	Geschäftstätigkeit	Wertschöpfungskette
GOV-1.20 (a)		Rolle der Verwaltung,
GOV-1.21 (a)-(d)	Erklärung zur Corporate Governance	Management und Aufsicht
GOV-1.21 (e)		Einrichtungen
GOV-3.29 (a)-(e)		Integration der Nachhaltigkeit
GOV-3.29 AR7	Vergütungsbericht	damit verbundene Leistung bei Anreizregelungen

Omitted_lines = 0, [ ]
Mistranslated lines = 2, [“is disclosed for the comparison period data”   period at the end of the German sentence is missing. Also, the number 'n' should be written as a word (keine) in German., , “8.20 List of participations of the Allianz Group as of 31 December 2024 according to § 313 (2) HGB Basis for preparation SBM-1.40 (a) Allianz business model and value chain SBM-1.42 (a)-(c) Business Operations GOV-1.20 (a) Role of administrative, management, and supervisory bodies GOV-1.21 (a)-(d) Corporate Governance Statement GOV-1.21 (e) GOV-3.29 (a)-(e) Integration of sustainability-related performance in incentive schemes GOV-3.29 AR7 Remuneration Report.”  The period at the end of the German sentence is missing. Also, the number '2' should be written as a word (zwei) in German.]
