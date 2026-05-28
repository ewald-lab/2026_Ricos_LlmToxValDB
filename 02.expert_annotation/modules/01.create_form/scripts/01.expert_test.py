import json
import argparse
import os
import sys
import sqlite3

def load_ontology_data(ontology_db: str) -> dict:
    """
    Description:
    Extracts ontology nodes from the embeddings database for the tree interface.

    Args:
        ontology_db (str): Path to the ontology database file.

    Returns:
        dict: A dictionary mapping node IDs to node data.
    """
    if not os.path.exists(ontology_db):
        print(f"Error: Ontology DB not found at {ontology_db}")
        return {}

    conn = sqlite3.connect(ontology_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sanitized_name, parent, definition, is_llm_generated FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    nodes = {}
    for name, sanitized_name, parent, definition, is_llm_generated in rows:
        # Use full name as key to avoid collisions with same sanitized_names
        node_id = f"node_{name}"
        nodes[node_id] = {
            "data": {
                "name": name,
                "sanitized_name": sanitized_name,
                "parent": parent,
                "definition": definition,
                "is_llm_generated": is_llm_generated
            }
        }
    return nodes

def generate_html(
    pool_data: list,
    ontology_data: dict,
    output_file: str,
    contact_email: str,
    context_ontology: str,
    debug: bool = False,
    skip_first_level: bool = False
) -> None:
    """
    Description:
    Generate HTML form for expert annotation, embedding the pool and ontology data.

    Args:
        pool_data (list): List of candidate pool phrases.
        ontology_data (dict): Dictionary of ontology nodes.
        output_file (str): Path to write the generated HTML file.
        contact_email (str): Email for sending results.
        context_ontology (str): Context of the ontology.
        debug (bool): If True, renders extra debugging RAG scores. Defaults to False.
        skip_first_level (bool): If True, skips the root node in paths. Defaults to False.

    Returns:
        None
    """
    # Convert data to JSON strings for embedding in JS
    pool_json = json.dumps(pool_data)
    ontology_json = json.dumps(ontology_data)

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Expert Annotation: Candidate Selection</title>
        <style>
            body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; font-size: 14px; }
            .container { width: 100%; box-sizing: border-box; }

            /* Header & Sticky Controls */
            .header-panel {
                position: sticky; top: 0; background: white; z-index: 1000;
                padding: 14px 18px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                margin-bottom: 20px; border: 1px solid #e0e0e0;
            }
            h1 { margin: 0 0 10px 0; font-size: 1.3em; color: #333; }

            .progress-bar-bg { width: 100%; background-color: #eee; height: 8px; border-radius: 4px; overflow: hidden; }
            .progress-bar-fill { height: 100%; background-color: #27ae60; width: 0%; transition: width 0.3s; }
            .stats { display: flex; justify-content: space-between; font-size: 0.8em; color: #7f8c8d; margin-top: 4px; }

            .btn-group { display: flex; gap: 8px; margin-top: 10px; }
            .btn { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; color: white; transition: opacity 0.2s; font-size: 0.85em; }
            .btn:hover { opacity: 0.9; }
            .btn-save { background-color: #2980b9; }
            .btn-email { background-color: #8e44ad; }
            .btn-clear { background-color: #e67e22; }
            .btn-tree { background-color: #16a085; margin-left: 0; width: auto; padding: 6px 12px; font-size: 0.9em; }

            /* Task Card */
            .task-card {
                background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.02); border: 1px solid #eee;
            }
            .phrase-box {
                background-color: #f8f9fa; border-left: 4px solid #333;
                padding: 10px 12px; font-size: 0.95em; font-weight: 500; color: #333;
                margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; gap: 8px;
            }
            .phrase-text {
                flex: 1;
                max-height: 50px;
                overflow-y: auto;
                padding-right: 10px;
            }

            /* Candidates Grid */
            .candidates-grid {
                display: flex; flex-direction: column; gap: 8px;
                margin-bottom: 14px;
            }
            .cand-path {
                display: flex; flex-wrap: wrap; align-items: center; gap: 4px; padding: 4px 0;
            }
            .path-node {
                cursor: pointer; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; transition: all 0.2s;
                background-color: #f0f0f0; color: #7f8c8d; position: relative;
            }
            .custom-tooltip {
                visibility: hidden; position: absolute; top: 120%; left: 50%; width: max-content;
                max-width: 350px; transform: translateX(-50%); background-color: #fff; color: #333;
                border: 1px solid #ccc; padding: 8px 10px; border-radius: 6px; z-index: 1000;
                font-size: 0.9em; font-weight: normal; text-align: left;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.15); opacity: 0; transition: opacity 0.2s, top 0.2s;
                pointer-events: none; line-height: 1.4; white-space: normal;
            }
            .path-node:hover .custom-tooltip {
                visibility: visible; opacity: 1; top: 130%;
            }
            .path-node.selected-path {
                background-color: #27ae60; color: white; border-color: #27ae60;
            }
            .path-node:hover {
                filter: brightness(0.9);
                z-index: 1000;
            }
            .path-node-leaf {
                font-weight: bold;
            }
            .path-sep {
                color: #ccc; font-size: 0.8em;
            }
            .candidate-option {
                display: flex; flex-direction: column; padding: 7px 9px;
                border: 1px solid #eee; border-radius: 5px;
                transition: background 0.2s, border-color 0.2s;
                position: relative;
            }
            .candidate-option:hover { background-color: #f9f9f9; border-color: #ccc; z-index: 100; }
            .candidate-option.selected { background-color: #e8f5e9; border-color: #27ae60; }
            .cand-info { flex: 1; }
            .cand-name { font-weight: 700; display: block; color: #333; font-size: 0.9em; }
            .cand-def { font-size: 0.8em; color: #666; margin-top: 2px; display: none; line-height: 1.3; }
            .cand-def.visible { display: block; }
            .def-source { font-size: 0.75em; font-weight: 600; padding: 2px 6px; border-radius: 3px; margin-bottom: 3px; display: inline-block; }
            .def-source-llm { background-color: #fdecea; color: #c0392b; }
            .def-source-ontology { background-color: #e8f5e9; color: #27ae60; }
            .cand-source { font-size: 0.75em; color: #999; margin-top: 4px; display: block; text-transform: uppercase; }
            .btn-toggle-def { background: none; border: 1px solid #ccc; border-radius: 50%; width: 22px; height: 22px; cursor: pointer; font-size: 0.8em; color: #888; margin-left: 6px; padding: 0; line-height: 22px; text-align: center; transition: all 0.2s; }
            .btn-toggle-def:hover { background-color: #eee; color: #333; border-color: #aaa; }

            /* Flag Buttons */
            .flag-group { display: flex; gap: 8px; flex-shrink: 0; }
            .btn-flag { padding: 4px 9px; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; font-size: 0.75em; background: #fff; color: #666; transition: all 0.2s; white-space: nowrap; }
            .btn-flag:hover { border-color: #aaa; }
            .btn-flag.active-bad { background-color: #fdecea; border-color: #e74c3c; color: #c0392b; }
            .btn-flag.active-skip { background-color: #fff8e1; border-color: #f39c12; color: #e67e22; }
            .btn-flag.active-no-match { background-color: #e1f5fe; border-color: #0288d1; color: #01579b; }

            /* LLM Indicator */
            .llm-indicator {
                display: flex;
                align-items: center;
                gap: 5px;
                margin-top: 5px;
                font-size: 0.8em;
            }
            .dot {
                height: 10px;
                width: 10px;
                border-radius: 50%;
                display: inline-block;
            }
            .dot-accepted { background-color: #27ae60; box-shadow: 0 0 5px #27ae60; }
            .dot-rejected { background-color: #e74c3c; box-shadow: 0 0 5px #e74c3c; }
            .llm-reasoning-box {
                font-style: italic;
                color: #555;
                background: #f1f1f1;
                padding: 8px;
                border-radius: 4px;
                margin-top: 5px;
                display: none;
                border-left: 3px solid #ccc;
            }
            .llm-reasoning-box.visible { display: block; }
            .btn-reasoning {
                background: none;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 0.75em;
                padding: 2px 6px;
                cursor: pointer;
                color: #666;
            }
            .btn-reasoning:hover { background: #eee; }

            /* Manual Search */
            .manual-section {
                margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;
            }
            .manual-label { font-weight: 600; color: #333; margin-bottom: 8px; display: block; }
            .search-wrapper { position: relative; display: flex; gap: 10px; }
            .search-input {
                flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px;
                font-family: inherit;
            }
            .manual-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
            .tag {
                background-color: #e1f5fe; color: #0277bd; padding: 5px 10px;
                border-radius: 15px; font-size: 0.9em; display: flex; align-items: center;
            }
            .tag-remove { margin-left: 6px; cursor: pointer; font-weight: bold; }
            .tag-remove:hover { color: #d32f2f; }

            /* User Modal */
            .modal { display: none; position: fixed; z-index: 2000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); align-items: center; justify-content: center; }
            .modal-content { background: white; padding: 30px; border-radius: 8px; width: 400px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
            .input-group { margin-bottom: 15px; text-align: left; }
            .input-group label { display: block; margin-bottom: 5px; font-weight: 600; }
            .input-group input { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }

            /* Tree View */
            .tree-modal-content { height: 80%; display: flex; flex-direction: column; width: 600px; }
            .tree-scroll-area {
                flex: 1; overflow-y: auto; text-align: left; padding: 20px;
                border: 1px solid #eee; margin-top: 10px; background: #fff; border-radius: 4px;
            }
            ul.tree-ul { list-style-type: none; padding-left: 20px; margin: 0; }

            /* Hierarchy Lines */
            ul.tree-ul ul {
                border-left: 1px solid #e0e0e0;
                margin-left: 7px;
                padding-left: 15px;
            }

            li.tree-li { margin: 4px 0; }
            .caret { cursor: pointer; user-select: none; display: inline-block; width: 20px; color: #95a5a6; text-align: center; margin-right: 2px; transition: color 0.2s; }
            .caret:hover { color: #34495e; }
            .caret::before { content: "\\25B6"; font-size: 0.8em; display: inline-block; transition: transform 0.2s; }
            .caret-down::before { transform: rotate(90deg); }
            .nested { display: none; }
            .active { display: block; }
            .tree-item { cursor: pointer; padding: 4px 8px; border-radius: 4px; transition: all 0.1s; display: inline-block; color: #34495e; font-size: 0.95em; }
            .tree-item:hover { background-color: #e0f7fa; color: #006064; }
            .tree-item.is-leaf { color: #2c3e50; }
            .tree-item.selected-in-tree { background-color: #d4edda; color: #155724; padding: 2px 6px; }
            .tree-search { width: 100%; padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; font-family: inherit; font-size: 0.95em; box-sizing: border-box; }
            .tree-search:focus { outline: none; border-color: #3498db; box-shadow: 0 0 0 2px rgba(52,152,219,0.15); }
            .tree-item.search-match { background-color: #ffe0b2; color: #5f4339; font-weight: 700; }
            li.tree-li.hidden-by-search { display: none; }
            .modal-text { text-align: justify; line-height: 1.6; margin: 15px 0; }

            /* Expander Panels (Instructions) */
            .expander-row {
                display: flex;
                gap: 20px;
                margin-top: 5px;
                margin-bottom: 10px;
            }
            .panel-expander {
                flex: 1;
                overflow: hidden;
            }
            .panel-expander summary {
                padding: 6px 0;
                font-weight: 600;
                cursor: pointer;
                background-color: transparent;
                color: #333;
                list-style: none;
                display: flex;
                align-items: center;
                gap: 8px;
                width: fit-content;
                font-size: 0.9em;
            }
            .panel-expander summary::-webkit-details-marker {
                display: none;
            }
            .panel-expander summary::before {
                content: "▶";
                display: inline-block;
                font-size: 0.75em;
                color: #2980b9;
                transition: transform 0.2s;
            }
            .panel-expander[open] summary::before {
                transform: rotate(90deg);
            }
            .panel-content {
                padding: 4px 0 10px 20px;
                font-size: 0.85em;
                line-height: 1.5;
                text-align: justify;
            }
            .panel-content ul {
                margin: 0;
                padding-left: 0;
            }
            .panel-content li {
                margin-bottom: 6px;
            }
            .panel-content li:last-child {
                margin-bottom: 0;
            }
        </style>
        <script>{jszip_inline}</script>
    </head>
    <body>

    <!-- User Info Modal -->
    <div id="userModal" class="modal" style="display: flex;">
        <div class="modal-content">
            <h2>Expert Annotator</h2>
            <p>Please enter your details to start.</p>
            <div class="input-group">
                <label>Full Name</label>
                <input type="text" id="evaluatorName" placeholder="e.g. Jane Doe">
            </div>
            <div class="input-group">
                <label>Institution</label>
                <input type="text" id="evaluatorInstitution" placeholder="e.g. EMBL">
            </div>
            <button class="btn btn-save" style="width:100%" onclick="showInstructions()">Next</button>
        </div>
    </div>

    <div id="instructionsModal" class="modal">
        <div class="modal-content" style="width: 680px; text-align: left;">
            <h2 style="text-align: center; margin-top: 0;">Instructions</h2>
            <ul style="margin: 15px 0; padding-left: 20px; line-height: 1.5; text-align: justify; font-size: 0.92em;">
                <li><strong>Review Context:</strong> Read the general context of the ontology provided at the top of the interface.</li>
                <li><strong>Select Terms:</strong> For each toxicological effect, choose the appropriate terms from the suggested candidates, taking into account their hierarchy. You can select an intermediate node in the hierarchy path if the final term is too specific. Hover over any term in the path to view its description.</li>
                <li><strong>Mapping Specificity:</strong> More specific terms can be mapped to more general terms, but <strong>not</strong> the other way around. <em>Example: "hepatocellular necrosis" → "necrosis" ✓, but "necrosis" → "hepatocellular necrosis" ✗.</em></li>
                <li><strong>Directionalities:</strong> Specific directionalities (e.g. <em>increased</em>) can be matched to general perturbations (e.g. <em>abnormal</em>), but <strong>not</strong> the other way around.</li>
                <li><strong>Answer Matching:</strong> If none of the suggested candidates are appropriate, click the "No Matches" button. You can optionally use the "Browse Ontology" button or the search bar if you have a specific term in mind, but you are not required to manually check the ontology each time. Note that there can be multiple answers or no answers, as not all terms are expected to have a match in each ontology.</li>
                <li><strong>Feedback &amp; Comments:</strong> Use "Toxicological effect does not make sense" if the phrase description is incorrect or misleading. Use "Not My Expertise" to skip phrases outside your area of knowledge. You can also leave an optional comment on each evaluated toxicological effect, and provide general feedback at the very bottom of the page.</li>
                <li><strong>Progress:</strong> Your work is auto-saved. You can close the window and resume later. When finished, click "Download Results" to save the ZIP file, then click "Send Results" to automatically open an email draft where you can attach the downloaded file.</li>
                <li><strong>External Search:</strong> You are allowed to search term definitions on the internet, but please <strong>do not</strong> ask Google / LLMs whether an effect–term pair matches.</li>
            </ul>
            <button class="btn btn-save" style="width:100%" onclick="startAnnotation()">Start Annotation</button>
        </div>
    </div>


    <!-- Ontology Tree Modal -->
    <div id="treeModal" class="modal">
        <div class="modal-content tree-modal-content">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2 style="margin: 0;">Select Ontology Term</h2>
                <span class="close" onclick="closeTreeModal()" style="font-size:24px; cursor:pointer;">&times;</span>
            </div>
            <p style="color:#666; margin: 5px 0 10px 0; text-align: left;">Click on a term to add it to the current task.</p>
            <input type="text" id="treeSearchInput" class="tree-search" placeholder="Search terms..." oninput="filterTree(this.value)">
            <div id="ontologyTree" class="tree-scroll-area"></div>
        </div>
    </div>

    <div class="container">
        <div class="header-panel">
            <h1>Expert Annotation: Toxicological Effects</h1>
            <div style="margin-bottom: 10px; color: #333; font-size: 0.92em; text-align: justify; line-height: 1.5; max-height: 80px; overflow-y: auto; padding-right: 5px; border-bottom: 1px solid #f0f0f0; margin-bottom: 15px;"><strong>Context of the ontology:</strong> {context_ontology}</div>

            <div class="expander-row">
            <details class="panel-expander">
                <summary>Instructions</summary>
                <div class="panel-content">
                    <ul>
                        <li><strong>Review Context:</strong> Read the general context of the ontology provided above.</li>
                        <li><strong>Select Terms:</strong> For each toxicological effect, choose the appropriate terms from the suggested candidates, taking into account their hierarchy. You can select an intermediate node in the hierarchy path if the final term is too specific. Hover over any term in the path to view its description.</li>
                        <li><strong>Mapping Specificity:</strong> More specific terms can be mapped to more general terms, but <strong>not</strong> the other way around. <em>Example: "hepatocellular necrosis" can be mapped to "necrosis", but "necrosis" should not be mapped to "hepatocellular necrosis".</em></li>
                        <li><strong>Directionalities:</strong> Specific directionalities (e.g. <em>increased</em>) can be matched to general perturbations (e.g. <em>abnormal</em>), but <strong>not</strong> the other way around.</li>
                        <li><strong>Answer Matching:</strong> If none of the suggested candidates are appropriate, click the "No Matches" button. You can optionally use the "Browse Ontology" button or the search bar if you have a specific term in mind, but you are not required to manually check the ontology each time. Note that there can be multiple answers or no answers, as not all terms are expected to have a match in each ontology.</li>
                        <li><strong>Feedback &amp; Comments:</strong> Use "Toxicological effect does not make sense" if the phrase description is incorrect or misleading. Use "Not My Expertise" to skip phrases outside your area of knowledge. You can also leave an optional comment on each evaluated toxicological effect, and provide general feedback at the very bottom of the page.</li>
                        <li><strong>Progress:</strong> Your work is auto-saved. You can close the window and resume later. When finished, click "Download Results" to save the ZIP file, then click "Send Results" to automatically open an email draft where you can attach the downloaded file.</li>
                        <li><strong>External Search:</strong> You are allowed to search term definitions on the internet, but please <strong>do not</strong> ask Google / LLMs whether an effect–term pair matches.</li>
                    </ul>
                </div>
            </details>

            </div>

            <div class="progress-bar-bg"><div id="progressBar" class="progress-bar-fill"></div></div>
            <div class="stats">
                <span id="progressText">0 / 0 Completed</span>
                <span>Auto-save enabled</span>
            </div>
            <div class="btn-group">
                <button class="btn btn-save" onclick="exportCSV()">Download Results</button>
                <button class="btn btn-email" onclick="sendResults()">Send Results</button>
                <button class="btn btn-clear" onclick="clearProgress()">Reset</button>
            </div>
        </div>

        <div id="tasksContainer"></div>
    </div>

    <script>
        // --- DATA INJECTION ---
        const rawPoolData = {pool_json};
        const ontologyNodes = {ontology_json};
        const ONTOLOGY_NAME = "{ontology_name}";
        const CONTACT_EMAIL = "{contact_email}";
        const DEBUG_MODE = {debug_mode};
        const SKIP_FIRST_LEVEL = {skip_first_level};

        const nodesByName = {};
        Object.values(ontologyNodes).forEach(n => {
            if(n.data && n.data.name) {
                const nameKey = n.data.name.trim();
                nodesByName[nameKey] = {
                    name: n.data.name,
                    sanitizedName: n.data.sanitized_name,
                    displayName: n.data.name,
                    parent: n.data.parent ? n.data.parent.trim() : null,
                    definition: n.data.definition,
                    isLlmGenerated: n.data.is_llm_generated,
                    children: []
                };
            }
        });
        Object.values(nodesByName).forEach(node => {
            if (node.parent && nodesByName[node.parent]) {
                nodesByName[node.parent].children.push(node);
            }
        });

        let evaluatorName = "";
        let evaluatorInstitution = "";
        let userState = {}; // { phrase_index: { selected: [names], manual: [names] } }
        let currentTreeTaskIdx = null;
        const allUpdateVisuals = {}; // { idx: [callbacks] }

        // Generate a unique ID for this specific test pool to avoid localStorage collisions
        // Using a simple hash function robust to non-ASCII characters
        function getHash(str) {
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                hash = ((hash << 5) - hash) + str.charCodeAt(i);
                hash |= 0;
            }
            return Math.abs(hash).toString(16);
        }
        const POOL_HASH = "pool_" + getHash(rawPoolData.map(p => p.phrase).join("|")).substring(0, 16);

        // --- INITIALIZATION ---
        function showInstructions() {
            const name = document.getElementById('evaluatorName').value.trim();
            const inst = document.getElementById('evaluatorInstitution').value.trim();
            if (!name || !inst) { alert("Please fill in all fields."); return; }

            evaluatorName = name;
            evaluatorInstitution = inst;
            document.getElementById('userModal').style.display = 'none';
            document.getElementById('instructionsModal').style.display = 'flex';
        }

        function startAnnotation() {
            document.getElementById('instructionsModal').style.display = 'none';

            loadState();
            renderTasks();
            updateProgress();
            // Pre-build tree
            setTimeout(buildTreeHTML, 100);
        }

        function renderTasks() {
            const container = document.getElementById('tasksContainer');
            container.innerHTML = '';

            rawPoolData.forEach((item, idx) => {
                const card = document.createElement('div');
                card.className = 'task-card';
                card.dataset.idx = idx;

                // Phrase + Flag Buttons
                const phraseBox = document.createElement('div');
                phraseBox.className = 'phrase-box';

                const phraseText = document.createElement('span');
                phraseText.className = 'phrase-text';
                phraseText.textContent = item.phrase;
                phraseBox.appendChild(phraseText);

                const flagGroup = document.createElement('div');
                flagGroup.className = 'flag-group';

                const noMatchBtn = document.createElement('button');
                noMatchBtn.className = 'btn-flag';
                noMatchBtn.textContent = 'No Matches';
                noMatchBtn.id = `noMatch-${idx}`;
                if (userState[idx] && userState[idx].noMatches) noMatchBtn.classList.add('active-no-match');
                noMatchBtn.onclick = () => toggleFlag(idx, 'noMatches', noMatchBtn, 'active-no-match');

                const badBtn = document.createElement('button');
                badBtn.className = 'btn-flag';
                badBtn.textContent = 'Toxicological effect does not make sense';
                badBtn.id = `bad-${idx}`;
                if (userState[idx] && userState[idx].badDescription) badBtn.classList.add('active-bad');
                badBtn.onclick = () => toggleFlag(idx, 'badDescription', badBtn, 'active-bad');

                const skipBtn = document.createElement('button');
                skipBtn.className = 'btn-flag';
                skipBtn.textContent = 'Not My Expertise';
                skipBtn.id = `skip-${idx}`;
                if (userState[idx] && userState[idx].notMyExpertise) skipBtn.classList.add('active-skip');
                skipBtn.onclick = () => toggleFlag(idx, 'notMyExpertise', skipBtn, 'active-skip');

                flagGroup.appendChild(noMatchBtn);
                flagGroup.appendChild(badBtn);
                flagGroup.appendChild(skipBtn);
                phraseBox.appendChild(flagGroup);
                card.appendChild(phraseBox);

                // Candidates
                const grid = document.createElement('div');
                grid.className = 'candidates-grid';

                // Add "None of the above" logic implicitly by allowing empty selection or manual only
                // But let's add the candidates from the pool
                let candidatesToRender = item.candidates.slice();
                if (!DEBUG_MODE) {
                    for (let i = candidatesToRender.length - 1; i > 0; i--) {
                        const j = Math.floor(Math.random() * (i + 1));
                        [candidatesToRender[i], candidatesToRender[j]] = [candidatesToRender[j], candidatesToRender[i]];
                    }
                }

                const updateVisuals = [];
                allUpdateVisuals[idx] = updateVisuals;
                candidatesToRender.forEach((cand, candIdx) => {
                    const opt = document.createElement('div');
                    opt.className = 'candidate-option';

                    const safeCandId = cand.name.replace(/[^a-zA-Z0-9]/g, '_') + '_' + candIdx;

                    let pathNodes = [];
                    let curr = cand.name;
                    while (curr && nodesByName[curr]) {
                        pathNodes.unshift(nodesByName[curr]);
                        curr = nodesByName[curr].parent;
                    }
                    if (pathNodes.length === 0) {
                        pathNodes = [{name: cand.name, displayName: cand.name}];
                    }

                    const pathContainer = document.createElement('div');
                    pathContainer.className = 'cand-path';

                    const nodesToRender = (SKIP_FIRST_LEVEL && pathNodes.length > 1) ? pathNodes.slice(1) : pathNodes;

                    nodesToRender.forEach((pn, pnIdx) => {
                        const nodeEl = document.createElement('span');
                        nodeEl.className = 'path-node';
                        if (pnIdx === nodesToRender.length - 1) nodeEl.classList.add('path-node-leaf');

                        nodeEl.textContent = pn.displayName;
                        const nodeData = nodesByName[pn.name];
                        let defStr = nodeData?.definition;
                        if (defStr && defStr.toLowerCase() !== 'under construction') {
                            defStr = defStr.charAt(0).toUpperCase() + defStr.slice(1);
                            const sourceText = nodeData.isLlmGenerated ? 'LLM Generated Definition' : 'Ontology Definition';
                            const sourceClass = nodeData.isLlmGenerated ? 'def-source-llm' : 'def-source-ontology';
                            const tooltip = document.createElement('div');
                            tooltip.className = 'custom-tooltip';
                            tooltip.innerHTML = `<span class="def-source ${sourceClass}" style="margin-bottom: 6px;">${sourceText}</span><br>${defStr}`;
                            nodeEl.appendChild(tooltip);

                            nodeEl.addEventListener('mouseenter', () => {
                                tooltip.style.left = '50%';
                                tooltip.style.right = 'auto';
                                tooltip.style.transform = 'translateX(-50%)';

                                const rect = tooltip.getBoundingClientRect();
                                const padding = 15;
                                if (rect.right > window.innerWidth - padding) {
                                    tooltip.style.left = 'auto';
                                    tooltip.style.right = '0';
                                    tooltip.style.transform = 'none';
                                } else if (rect.left < padding) {
                                    tooltip.style.left = '0';
                                    tooltip.style.right = 'auto';
                                    tooltip.style.transform = 'none';
                                }
                            });
                        } else {
                            nodeEl.title = 'Description Unavailable';
                        }
                        nodeEl.onclick = (e) => {
                            e.preventDefault();
                            e.stopPropagation();

                            if (!userState[idx]) userState[idx] = { selected: [], manual: [], badDescription: false, notMyExpertise: false, comments: "" };
                            if (!userState[idx].selected) userState[idx].selected = [];

                            const clickedNodeName = pn.name;
                            const currentSelected = userState[idx].selected;
                            selectLineage(idx, clickedNodeName);
                        };

                        pathContainer.appendChild(nodeEl);

                        if (pnIdx < nodesToRender.length - 1) {
                            const sep = document.createElement('span');
                            sep.className = 'path-sep';
                            sep.innerHTML = '&#9656;';
                            pathContainer.appendChild(sep);
                        }
                    });

                    updateVisuals.push(() => {
                        const activeNodes = new Set();
                        if (userState[idx]) {
                            const combined = [...(userState[idx].selected || []), ...(userState[idx].manual || [])];
                            combined.forEach(s => {
                                let curr = s;
                                while (curr) {
                                    activeNodes.add(curr);
                                    curr = nodesByName[curr]?.parent;
                                }
                            });
                        }

                        let isPathActive = false;
                        for(let i=0; i < nodesToRender.length; i++) {
                            const el = pathContainer.children[i*2]; // nodes are at even indices
                            if (el && activeNodes.has(nodesToRender[i].name)) {
                                el.classList.add('selected-path');
                                isPathActive = true;
                            } else if (el) {
                                el.classList.remove('selected-path');
                            }
                        }

                        const leafNode = pathNodes[pathNodes.length - 1];
                        if (activeNodes.has(leafNode.name)) {
                            opt.classList.add('selected');
                        } else {
                            opt.classList.remove('selected');
                        }
                    });

                    const info = document.createElement('div');
                    info.className = 'cand-info';
                    info.style.marginTop = '0px';

                    let llmHtml = '';
                    if (DEBUG_MODE) {
                        const ragScore = cand.score !== undefined ? cand.score : 'N/A';
                        const rerankScore = cand.rerank_score !== undefined ? Number(cand.rerank_score).toFixed(4) : 'N/A';
                        const origOrder = cand.original_order !== undefined ? cand.original_order : (candIdx + 1);
                        llmHtml += `
                            <div style="font-size: 0.85em; color: #856404; background-color: #fff3cd; padding: 4px 8px; border-radius: 4px; display: inline-block; margin-top: 5px;">
                                <strong>Order:</strong> ${origOrder} | <strong>RAG:</strong> ${ragScore} | <strong>Rerank:</strong> ${rerankScore}
                            </div>
                        `;
                        if (cand.llm_decision) {
                            const dotClass = cand.llm_decision === 'ACCEPTED' ? 'dot-accepted' : 'dot-rejected';
                            const reasoningId = `reasoning-${idx}-${safeCandId}`;
                            llmHtml += `
                                <div class="llm-indicator" style="margin-top: 5px;">
                                    <span class="dot ${dotClass}" title="LLM Decision: ${cand.llm_decision}"></span>
                                    <strong>LLM: ${cand.llm_decision}</strong>
                                    <button class="btn-reasoning" onclick="event.preventDefault(); event.stopPropagation(); toggleDef('${reasoningId}')">Reasoning</button>
                                </div>
                                <div class="llm-reasoning-box" id="${reasoningId}">
                                    ${cand.llm_reasoning || 'No reasoning available.'}
                                </div>
                            `;
                        }
                    }

                    info.innerHTML = `${llmHtml}`;

                    opt.appendChild(pathContainer);
                    opt.appendChild(info);
                    grid.appendChild(opt);
                });
                updateVisuals.forEach(cb => cb());
                card.appendChild(grid);



                // Manual Search Section
                const manualSec = document.createElement('div');
                manualSec.className = 'manual-section';

                const label = document.createElement('span');
                label.className = 'manual-label';
                label.innerHTML = "If <span style='color: red;'>None</span> of these terms apply or there are <span style='color: red;'>No</span> terms presented and you have a specific term in mind, browse the ontology.";
                manualSec.appendChild(label);

                const searchWrap = document.createElement('div');
                searchWrap.className = 'search-wrapper';

                const treeBtn = document.createElement('button');
                treeBtn.className = 'btn btn-tree';
                treeBtn.textContent = 'Browse Ontology';
                treeBtn.onclick = () => openTreeModal(idx);

                searchWrap.appendChild(treeBtn);
                manualSec.appendChild(searchWrap);

                // Manual Tags Container
                const tagsCont = document.createElement('div');
                tagsCont.className = 'manual-tags';
                tagsCont.id = `tags-${idx}`;

                // Render existing manual tags
                if (userState[idx] && userState[idx].manual) {
                    userState[idx].manual.forEach(term => {
                        addTagElement(tagsCont, term, idx);
                    });
                }

                manualSec.appendChild(tagsCont);

                // Comments Section
                const commentWrapper = document.createElement('div');
                commentWrapper.style.marginTop = '15px';
                commentWrapper.style.borderTop = '1px solid #eee';
                commentWrapper.style.paddingTop = '10px';

                const commentLabel = document.createElement('label');
                commentLabel.className = 'manual-label';
                commentLabel.textContent = 'Do you have any comments to add about your evaluation of this toxicological effect? (Optional)';
                commentWrapper.appendChild(commentLabel);

                const commentArea = document.createElement('textarea');
                commentArea.className = 'search-input';
                commentArea.style.width = '100%';
                commentArea.style.boxSizing = 'border-box';
                commentArea.style.minHeight = '60px';
                commentArea.style.marginTop = '5px';
                commentArea.style.resize = 'vertical';
                commentArea.placeholder = 'Add any notes or comments here...';

                if (userState[idx] && userState[idx].comments) {
                    commentArea.value = userState[idx].comments;
                }

                commentArea.oninput = (e) => {
                    if (!userState[idx]) userState[idx] = { selected: [], manual: [], badDescription: false, notMyExpertise: false, comments: "" };
                    userState[idx].comments = e.target.value;
                    saveState();
                };

                commentWrapper.appendChild(commentArea);
                manualSec.appendChild(commentWrapper);
                card.appendChild(manualSec);

                container.appendChild(card);
            });

            // General Feedback Card
            const generalCard = document.createElement('div');
            generalCard.className = 'task-card';
            generalCard.style.marginTop = '40px';
            generalCard.style.border = '2px solid #3498db';
            generalCard.style.backgroundColor = '#f4f9fd';

            const generalTitle = document.createElement('h3');
            generalTitle.style.marginTop = '0';
            generalTitle.style.color = '#333';
            generalTitle.textContent = 'General Feedback';
            generalCard.appendChild(generalTitle);

            const generalDesc = document.createElement('p');
            generalDesc.style.color = '#333';
            generalDesc.style.marginBottom = '15px';
            generalDesc.textContent = 'Please provide any general comments about your level of expertise, how you felt answering this questionnaire, or any other feedback (Optional).';
            generalCard.appendChild(generalDesc);

            const generalArea = document.createElement('textarea');
            generalArea.className = 'search-input';
            generalArea.style.width = '100%';
            generalArea.style.boxSizing = 'border-box';
            generalArea.style.minHeight = '100px';
            generalArea.style.resize = 'vertical';
            generalArea.placeholder = 'Add your general feedback here...';

            if (userState.generalFeedback) {
                generalArea.value = userState.generalFeedback;
            }

            generalArea.oninput = (e) => {
                userState.generalFeedback = e.target.value;
                saveState();
            };

            generalCard.appendChild(generalArea);
            container.appendChild(generalCard);
        }

        // --- TREE LOGIC ---
        function buildTreeHTML() {
            const trueRoots = [];
            Object.values(nodesByName).forEach(node => {
                if (!node.parent || !nodesByName[node.parent]) {
                    trueRoots.push(node);
                }
            });

            // 3. Prepare and Sort roots
            let rootsToRender = trueRoots;
            if (SKIP_FIRST_LEVEL && trueRoots.length === 1 && trueRoots[0].children.length > 0) {
                rootsToRender = trueRoots[0].children;
            }

            const sortFn = (a, b) => a.name.localeCompare(b.name);
            rootsToRender.sort(sortFn);

            function sortChildren(node) {
                node.children.sort(sortFn);
                node.children.forEach(sortChildren);
            }
            rootsToRender.forEach(sortChildren);

            // 4. Render
            function createUl(nodes) {
                if (nodes.length === 0) return null;
                const ul = document.createElement('ul');
                ul.className = 'tree-ul';
                nodes.forEach(node => {
                    const li = document.createElement('li');
                    li.className = 'tree-li';

                    const hasChildren = node.children.length > 0;

                    if (hasChildren) {
                        const span = document.createElement('span');
                        span.className = 'caret';
                        span.onclick = function() {
                            this.parentElement.querySelector('.nested').classList.toggle('active');
                            this.classList.toggle('caret-down');
                        };
                        li.appendChild(span);
                    } else {
                        const spacer = document.createElement('span');
                        spacer.style.display = 'inline-block';
                        spacer.style.width = '22px';
                        li.appendChild(spacer);
                    }

                    const text = document.createElement('span');
                    text.className = 'tree-item';
                    text.textContent = node.displayName;
                    text.setAttribute('data-term', node.name);
                    text.onclick = function() {
                        selectTreeTerm(node.name);
                    };

                    const infoBtn = document.createElement('button');
                    infoBtn.className = 'btn-toggle-def';
                    infoBtn.style.width = '18px';
                    infoBtn.style.height = '18px';
                    infoBtn.style.lineHeight = '18px';
                    infoBtn.style.fontSize = '0.7em';
                    infoBtn.textContent = 'i';
                    const safeNodeId = node.name.replace(/[^a-zA-Z0-9]/g, '_');
                    const defId = `tree-def-${safeNodeId}`;
                    infoBtn.onclick = (e) => {
                        e.stopPropagation();
                        toggleDef(defId);
                    };

                    li.appendChild(text);
                    li.appendChild(infoBtn);

                    const defSpan = document.createElement('div');
                    defSpan.id = defId;
                    defSpan.className = 'cand-def';
                    defSpan.style.marginLeft = '25px';
                    defSpan.style.marginBottom = '5px';

                    const isLlm = node.isLlmGenerated;
                    const sourceClass = isLlm ? 'def-source-llm' : 'def-source-ontology';
                    const sourceText = isLlm ? 'LLM Generated Description' : 'Ontology Description';

                    let nodeDef = node.definition;
                    if (!nodeDef || nodeDef.toLowerCase() === 'under construction') nodeDef = 'Description Unavailable';

                    defSpan.innerHTML = `<span class="def-source ${sourceClass}">${sourceText}</span> ${nodeDef}`;
                    li.appendChild(defSpan);

                    if (hasChildren) {
                        const nestedUl = createUl(node.children);
                        nestedUl.className = 'nested tree-ul';
                        li.appendChild(nestedUl);
                    }

                    ul.appendChild(li);
                });
                return ul;
            }

            const treeContainer = document.getElementById('ontologyTree');
            treeContainer.innerHTML = '';
            const rootUl = createUl(rootsToRender);
            if (rootUl) {
                rootUl.classList.remove('nested');
                treeContainer.appendChild(rootUl);
            } else {
                treeContainer.textContent = "No ontology structure found.";
            }
        }

        function openTreeModal(idx) {
            currentTreeTaskIdx = idx;
            document.getElementById('treeModal').style.display = 'flex';
            refreshTreeVisuals();
        }

        function closeTreeModal() {
            document.getElementById('treeModal').style.display = 'none';
            document.getElementById('treeSearchInput').value = '';
            filterTree('');
        }

        function filterTree(query) {
            const allLis = document.querySelectorAll('#ontologyTree li.tree-li');
            const q = query.trim().toLowerCase();

            if (!q) {
                allLis.forEach(li => {
                    li.classList.remove('hidden-by-search');
                    const item = li.querySelector(':scope > .tree-item');
                    if (item) item.classList.remove('search-match');
                    const nested = li.querySelector(':scope > .nested');
                    if (nested) nested.classList.remove('active');
                    const caret = li.querySelector(':scope > .caret');
                    if (caret) caret.classList.remove('caret-down');
                });
                return;
            }

            // First pass: mark matches
            allLis.forEach(li => {
                const item = li.querySelector(':scope > .tree-item');
                if (!item) return;
                const text = item.textContent.toLowerCase();
                if (text.includes(q)) {
                    item.classList.add('search-match');
                } else {
                    item.classList.remove('search-match');
                }
            });

            // Second pass: hide non-matching branches, show matching + ancestors
            function hasMatchInSubtree(li) {
                const item = li.querySelector(':scope > .tree-item');
                if (item && item.classList.contains('search-match')) return true;
                const nested = li.querySelector(':scope > .nested');
                if (!nested) return false;
                const children = nested.querySelectorAll(':scope > li.tree-li');
                for (const child of children) {
                    if (hasMatchInSubtree(child)) return true;
                }
                return false;
            }

            function applyVisibility(li) {
                const match = hasMatchInSubtree(li);
                if (match) {
                    li.classList.remove('hidden-by-search');
                    const nested = li.querySelector(':scope > .nested');
                    if (nested) {
                        nested.classList.add('active');
                        const caret = li.querySelector(':scope > .caret');
                        if (caret) caret.classList.add('caret-down');
                        nested.querySelectorAll(':scope > li.tree-li').forEach(applyVisibility);
                    }
                } else {
                    li.classList.add('hidden-by-search');
                }
            }

            document.querySelectorAll('#ontologyTree > ul > li.tree-li').forEach(applyVisibility);
        }

        function updateTreeItemVisual(term, selected) {
            refreshTreeVisuals();
        }

        function refreshTreeVisuals() {
            if (currentTreeTaskIdx === null) return;
            const activeNodes = new Set();
            const explicitNodes = new Set();
            const state = userState[currentTreeTaskIdx];
            if (state) {
                const combined = [ ...(state.selected || []), ...(state.manual || []) ];
                combined.forEach(s => {
                    explicitNodes.add(s);
                    let curr = s;
                    while (curr) {
                        activeNodes.add(curr);
                        curr = nodesByName[curr]?.parent;
                    }
                });
            }

            document.querySelectorAll('.tree-item').forEach(el => {
                const termName = el.dataset.term;
                const li = el.parentElement;
                if (activeNodes.has(termName)) {
                    el.classList.add('selected-in-tree');

                    // Auto-expand ancestors
                    const nested = li.querySelector(':scope > .nested');
                    const caret = li.querySelector(':scope > .caret');
                    if (nested && !nested.classList.contains('active')) {
                        // Only expand if it has selected children
                        let hasSelectedChild = false;
                        const childrenTerms = nested.querySelectorAll('.tree-item');
                        for(let childItem of childrenTerms) {
                            if (activeNodes.has(childItem.dataset.term)) { hasSelectedChild = true; break; }
                        }
                        if (hasSelectedChild) {
                            nested.classList.add('active');
                            if (caret) caret.classList.add('caret-down');
                        }
                    }
                } else {
                    el.classList.remove('selected-in-tree');
                }
            });
        }

        function isAncestorOf(parentName, childName) {
            let curr = nodesByName[childName]?.parent;
            while (curr) {
                if (curr === parentName) return true;
                curr = nodesByName[curr]?.parent;
            }
            return false;
        }

        function selectLineage(idx, clickedNodeName, isFromBrowseOntology = false) {
            if (!userState[idx]) userState[idx] = { selected: [], manual: [], badDescription: false, notMyExpertise: false, comments: "", noMatches: false };
            if (!userState[idx].selected) userState[idx].selected = [];
            if (!userState[idx].manual) userState[idx].manual = [];

            const currentSelected = userState[idx].selected;
            const currentManual = userState[idx].manual;

            // Calculate active nodes before click
            const activeNodes = new Set();
            [...currentSelected, ...currentManual].forEach(s => {
                let curr = s;
                while (curr) {
                    activeNodes.add(curr);
                    curr = nodesByName[curr]?.parent;
                }
            });

            const isActive = activeNodes.has(clickedNodeName);

            if (isActive) {
                // Clicked something in the green path -> Deselect branch from here down
                userState[idx].selected = currentSelected.filter(existing => {
                    if (existing === clickedNodeName) return false;
                    if (isAncestorOf(clickedNodeName, existing)) return false;
                    return true;
                });
                userState[idx].manual = currentManual.filter(existing => {
                    if (existing === clickedNodeName) return false;
                    if (isAncestorOf(clickedNodeName, existing)) return false;
                    return true;
                });

                const parentName = nodesByName[clickedNodeName]?.parent;
                if (parentName) {
                    if (!userState[idx].selected.includes(parentName) && !userState[idx].manual.includes(parentName)) {
                        if (!isFromBrowseOntology) {
                            userState[idx].selected.push(parentName);
                        } else {
                            const isCandidate = rawPoolData[idx].candidates.some(c => c.name === parentName);
                            if (isCandidate) userState[idx].selected.push(parentName);
                            else userState[idx].manual.push(parentName);
                        }
                    }
                }
            } else {
                // Clicked something not active -> Select it and clear conflicting lineage
                userState[idx].selected = currentSelected.filter(existing => {
                    if (isAncestorOf(existing, clickedNodeName)) return false;
                    if (isAncestorOf(clickedNodeName, existing)) return false;
                    return true;
                });
                userState[idx].manual = currentManual.filter(existing => {
                    if (isAncestorOf(existing, clickedNodeName)) return false;
                    if (isAncestorOf(clickedNodeName, existing)) return false;
                    return true;
                });

                if (!isFromBrowseOntology) {
                    userState[idx].selected.push(clickedNodeName);
                } else {
                    const isCandidate = rawPoolData[idx].candidates.some(c => c.name === clickedNodeName);
                    if (isCandidate) userState[idx].selected.push(clickedNodeName);
                    else userState[idx].manual.push(clickedNodeName);
                }
            }

            // Sync tags
            const tagContainer = document.getElementById(`tags-${idx}`);
            if (tagContainer) {
                tagContainer.innerHTML = '';
                userState[idx].manual.forEach(t => addTagElement(tagContainer, t, idx));
            }

            saveState();
            if (allUpdateVisuals[idx]) {
                allUpdateVisuals[idx].forEach(cb => cb());
            }
            refreshTreeVisuals();
            updateProgress();
        }

        function selectTreeTerm(term) {
            if (currentTreeTaskIdx !== null) {
                selectLineage(currentTreeTaskIdx, term, true);
            }
        }

        // --- LOGIC ---
        function handleSelection(idx, term, isChecked, element) {
            if (!userState[idx]) userState[idx] = { selected: [], manual: [], badDescription: false, notMyExpertise: false, noMatches: false, comments: "" };
            if (userState[idx].notMyExpertise || userState[idx].noMatches) return;

            if (isChecked) {
                userState[idx].selected.push(term);
                element.classList.add('selected');
            } else {
                userState[idx].selected = userState[idx].selected.filter(t => t !== term);
                element.classList.remove('selected');
            }
            saveState();
            updateProgress();
        }

        function addManualTerm(idx, term) {
            if (!userState[idx]) userState[idx] = { selected: [], manual: [], badDescription: false, notMyExpertise: false, noMatches: false, comments: "" };
            if (userState[idx].notMyExpertise || userState[idx].noMatches) return;
            if (!userState[idx].manual.includes(term)) {
                userState[idx].manual.push(term);
                const container = document.getElementById(`tags-${idx}`);
                addTagElement(container, term, idx);
                saveState();
                updateProgress();
            }
        }

        function removeManualTerm(idx, term) {
            if (userState[idx] && userState[idx].manual) {
                userState[idx].manual = userState[idx].manual.filter(t => t !== term);
                // Re-render tags is easiest or remove element
                const container = document.getElementById(`tags-${idx}`);
                container.innerHTML = '';
                userState[idx].manual.forEach(t => addTagElement(container, t, idx));
                saveState();
                updateProgress();
            }
        }

        function addTagElement(container, term, idx) {
            const tag = document.createElement('div');
            tag.className = 'tag';

            let pathNodes = [];
            let curr = term;
            while (curr && nodesByName[curr]) {
                pathNodes.unshift(nodesByName[curr]);
                curr = nodesByName[curr].parent;
            }
            if (pathNodes.length === 0) {
                pathNodes = [{displayName: term}];
            }

            const nodesToRender = (SKIP_FIRST_LEVEL && pathNodes.length > 1) ? pathNodes.slice(1) : pathNodes;

            const pathHtml = nodesToRender.map((pn, i) => {
                if (i === nodesToRender.length - 1) return `<strong>${pn.displayName}</strong>`;
                return pn.displayName;
            }).join(' <span style="margin: 0 5px; opacity: 0.6; font-size: 0.9em;">&#9656;</span> ');

            tag.innerHTML = `${pathHtml} <span class="tag-remove" style="margin-left: 10px;" onclick="removeManualTerm(${idx}, '${term}')">&times;</span>`;
            container.appendChild(tag);
        }

        // --- STATE & EXPORT ---
        function saveState() {
            localStorage.setItem('expert_annotation_state', JSON.stringify(userState));
        }

        function loadState() {
            const saved = localStorage.getItem('expert_annotation_state');
            if (saved) userState = JSON.parse(saved);
        }

        function toggleFlag(idx, flagKey, btn, activeClass) {
            if (!userState[idx]) userState[idx] = { selected: [], manual: [], badDescription: false, notMyExpertise: false, noMatches: false, comments: "" };

            // Mutual exclusivity for skip and no matches
            if (flagKey === 'noMatches' && !userState[idx].noMatches) {
                if (userState[idx].notMyExpertise) {
                    userState[idx].notMyExpertise = false;
                    document.getElementById(`skip-${idx}`).classList.remove('active-skip');
                    document.querySelector(`.task-card[data-idx="${idx}"]`).style.opacity = '1';
                }
            } else if (flagKey === 'notMyExpertise' && !userState[idx].notMyExpertise) {
                if (userState[idx].noMatches) {
                    userState[idx].noMatches = false;
                    document.getElementById(`noMatch-${idx}`).classList.remove('active-no-match');
                }
            }

            userState[idx][flagKey] = !userState[idx][flagKey];
            btn.classList.toggle(activeClass);

            if ((flagKey === 'notMyExpertise' && userState[idx].notMyExpertise) ||
                (flagKey === 'noMatches' && userState[idx].noMatches) ||
                (flagKey === 'badDescription' && userState[idx].badDescription)) {
                // Clear all selections and manual terms
                userState[idx].selected = [];
                userState[idx].manual = [];
                const card = document.querySelector(`.task-card[data-idx="${idx}"]`);
                if (card) {
                    const tagsCont = document.getElementById(`tags-${idx}`);
                    if (tagsCont) tagsCont.innerHTML = '';
                    if (allUpdateVisuals[idx]) allUpdateVisuals[idx].forEach(cb => cb());
                    if (userState[idx].notMyExpertise || userState[idx].badDescription) {
                        card.style.opacity = '0.5';
                    }
                }
            } else if ((flagKey === 'notMyExpertise' && !userState[idx].notMyExpertise) ||
                       (flagKey === 'badDescription' && !userState[idx].badDescription)) {
                const card = document.querySelector(`.task-card[data-idx="${idx}"]`);
                if (card && !userState[idx].notMyExpertise && !userState[idx].badDescription) {
                    card.style.opacity = '1';
                }
            }

            saveState();
            updateProgress();
        }

        function updateProgress() {
            const total = rawPoolData.length;
            const completed = Object.keys(userState).filter(k =>
                (userState[k].selected && userState[k].selected.length > 0) ||
                (userState[k].manual && userState[k].manual.length > 0) ||
                userState[k].notMyExpertise ||
                userState[k].noMatches ||
                userState[k].badDescription
            ).length;

            const pct = Math.round((completed / total) * 100);
            document.getElementById('progressBar').style.width = pct + '%';
            document.getElementById('progressText').innerText = `${completed} / ${total} Completed`;
        }

        function clearProgress() {
            if(confirm("Clear all progress?")) {
                localStorage.removeItem('expert_annotation_state');
                location.reload();
            }
        }

        function getCSVContent() {
            let csv = "Index,Phrase,Selected_Candidates,Selected_URIs,Manual_Terms,Toxicological_Effect_Does_Not_Make_Sense,Not_My_Expertise,Comments\\n";
            rawPoolData.forEach((item, idx) => {
                const state = userState[idx] || { selected: [], manual: [], badDescription: false, notMyExpertise: false, noMatches: false, comments: "" };

                let sel = (state.selected && state.selected.length > 0) ? state.selected.join("|") : "None";
                if (state.noMatches) sel = "-";

                // Map selected names to URIs
                let selUris = "None";
                if (state.selected && state.selected.length > 0) {
                    const uriList = state.selected.map(name => {
                        const cand = item.candidates.find(c => c.name === name);
                        return cand ? cand.uri : "Unknown";
                    });
                    selUris = uriList.join("|");
                } else if (state.noMatches) {
                    selUris = "-";
                }

                let man = (state.manual && state.manual.length > 0) ? state.manual.join("|") : "None";
                if (state.noMatches) man = "-";

                const comm = (state.comments) ? state.comments.replace(/"/g, '""').replace(/\\n/g, ' ') : "";

                const bad = state.badDescription ? 1 : 0;
                const skip = state.notMyExpertise ? 1 : 0;

                // Escape quotes
                const phrase = item.phrase.replace(/"/g, '""');
                csv += `${idx},"${phrase}","${sel}","${selUris}","${man}","${bad}","${skip}","${comm}"\\n`;
            });
            return csv;
        }

        function downloadFile(content, filename, type) {
            const blob = content instanceof Blob ? content : new Blob([content], { type: type });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.click();
        }

        function exportCSV() {
            const cleanName = evaluatorName.replace(/\\s+/g, '_');
            const cleanInst = evaluatorInstitution.replace(/\\s+/g, '_');
            const ontSuffix = ONTOLOGY_NAME ? `_ontology_${ONTOLOGY_NAME}` : "";

            const zip = new JSZip();

            // Add CSV
            const csv = getCSVContent();
            zip.file(`ToxValLlm_expert_${cleanName}_${cleanInst}${ontSuffix}.csv`, csv);

            // Add Feedback txt always
            const feedbackText = userState.generalFeedback ? userState.generalFeedback : "";
            zip.file(`general_feedback_${cleanName}_${cleanInst}${ontSuffix}.txt`, feedbackText);

            // Generate zip and download
            zip.generateAsync({type:"blob"}).then(function(blob) {
                downloadFile(blob, `ToxValLlm_expert_results_${cleanName}_${cleanInst}${ontSuffix}.zip`, 'application/zip');
            });
        }

        function sendResults() {
            exportCSV();

            const subject = `Expert Annotation Results - ${evaluatorName}`;
            const body = `Please find attached the annotation files.\\n\\nEvaluator: ${evaluatorName}\\nInstitution: ${evaluatorInstitution}`;
            window.location.href = `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

            alert("Files downloaded. Please attach them to the email.");
        }
        function toggleDef(defId) {
            const el = document.getElementById(defId);
            if (el) el.classList.toggle('visible');
        }
    </script>
    </body>
    </html>
    """

    final_html = html_template.replace("{pool_json}", pool_json)
    final_html = final_html.replace("{ontology_json}", ontology_json)
    final_html = final_html.replace("{contact_email}", contact_email)
    final_html = final_html.replace("{context_ontology}", context_ontology)
    final_html = final_html.replace("{debug_mode}", "true" if debug else "false")
    final_html = final_html.replace("{skip_first_level}", "true" if skip_first_level else "false")

    # Embed JSZip inline so the HTML works without internet access
    jszip_url = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"
    try:
        import urllib.request
        with urllib.request.urlopen(jszip_url, timeout=10) as resp:
            jszip_content = resp.read().decode('utf-8')
    except Exception:
        # Fallback: try local cache
        jszip_cache = os.path.join(os.path.dirname(__file__), 'jszip.min.js')
        if os.path.exists(jszip_cache):
            with open(jszip_cache, 'r') as f:
                jszip_content = f.read()
        else:
            jszip_content = "console.error('JSZip not available - download will not work');"
    final_html = final_html.replace("{jszip_inline}", jszip_content)

    # Get ontology name from output file path to inject in JS
    ont_name = os.path.basename(output_file).replace("expert_annotation_", "").replace(".html", "")
    final_html = final_html.replace("{ontology_name}", ont_name)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"Expert annotation HTML saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Expert Annotation HTML Interface")
    parser.add_argument("--input_pool", required=True, help="Path to JSON file from test.py containing candidate pools.")
    parser.add_argument("--ontology_db", required=True, help="Path to ontology embeddings database.")
    parser.add_argument("--output_file", required=True, help="Path to output HTML file.")
    parser.add_argument("--email", default="ricos@ebi.ac.uk", help="Contact email for results.")
    parser.add_argument("--context_ontology", default="", help="Context definition of the ontology.")
    parser.add_argument("--debug", action="store_true", help="Show LLM reasoning indicators in the interface.")
    parser.add_argument("--skip_first_level", action="store_true", help="Skip the first level of the hierarchy (e.g. 'mammalian phenotype') in the UI paths.")
    parser.add_argument("--top_reranker", type=int, default=8, help="Maximum number of candidates to show from the reranker (truncates if the pool has more). (default: 8)")
    parser.add_argument("--num_phrases", type=int, default=None, help="Maximum number of phrases (toxicological effects) to include in the test. If not set, all phrases are included.")

    args = parser.parse_args()

    if not os.path.exists(args.input_pool):
        sys.exit(f"Error: Input pool file {args.input_pool} not found.")
    if not os.path.exists(args.ontology_db):
        sys.exit(f"Error: Ontology DB {args.ontology_db} not found.")

    with open(args.input_pool, 'r', encoding='utf-8') as f:
        pool_data = json.load(f)

    if args.num_phrases is not None and args.num_phrases > 0:
        pool_data = pool_data[:args.num_phrases]

    ontology_data = load_ontology_data(args.ontology_db)

    # helper for hierarchy
    def get_hierarchy_path(node_name: str, nodes_dict: dict) -> str:
        """
        Description:
        Helper function to generate the hierarchy path string for a node.

        Args:
            node_name (str): The name of the node.
            nodes_dict (dict): A dictionary containing ontology nodes.

        Returns:
            str: The hierarchy path string.
        """
        path = []
        curr = node_name
        visited = set()
        while curr and curr in nodes_dict and curr not in visited:
            visited.add(curr)
            path.append(curr)
            curr = nodes_dict[curr]["data"]["parent"]
        return " > ".join(reversed(path))

    # Filter top reranker results for each item in the pool
    all_ontology_term_names = [n["data"]["name"] for n in ontology_data.values()]
    ontology_by_name = {n["data"]["name"]: n for n in ontology_data.values()}

    for item in pool_data:
        # 1. Filter top reranker results
        candidates = item.get("candidates", [])
        # Sort by rerank_score descending, handle missing scores
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        # Keep top N
        top_candidates = candidates[:args.top_reranker]

        item["candidates"] = top_candidates

    generate_html(pool_data, ontology_data, args.output_file, args.email, args.context_ontology, debug=args.debug, skip_first_level=args.skip_first_level)
