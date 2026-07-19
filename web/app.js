// Dynamic Multi-Category Test Harness
window.globalRunData = [];
let SYSTEM_PRESETS = [];
let selectedPresetIndex = 0;

// Initialize and pull the regression matrix from the root directory via Eel
window.addEventListener("DOMContentLoaded", () => {
    // Request the root JSON file from the Python backend
    eel.load_regression_suite_from_root()(function(response) {
        if (response && !response.error) {
            SYSTEM_PRESETS = response;
            buildPresetMatrixUI();
            selectPreset(0);
        } else {
            console.error("Failed to load regression_suite.json from root:", response ? response.error : "Unknown error");
            alert("Warning: regression_suite.json not found in root directory.");
        }
    });

    // Existing settings loader...
    eel.ui_get_current_settings()(function(data) {
        if(data) {
            if(document.getElementById("sl-minp")) document.getElementById("sl-minp").value = data.LLM_MIN_P || 0.05;
            if(document.getElementById("val-minp")) document.getElementById("val-minp").innerText = data.LLM_MIN_P || 0.05;
            if(document.getElementById("sl-topp")) document.getElementById("sl-topp").value = data.LLM_TOP_P || 0.9;
            if(document.getElementById("val-topp")) document.getElementById("val-topp").innerText = data.LLM_TOP_P || 0.9;
            if(document.getElementById("sl-speed")) document.getElementById("sl-speed").value = data.TTS_SPEED || 1.15;
            if(document.getElementById("val-speed")) document.getElementById("val-speed").innerText = (data.TTS_SPEED || 1.15) + "x";
            if(document.getElementById("sl-steps")) document.getElementById("sl-steps").value = data.TTS_QUALITY_STEPS || 5;
            if(document.getElementById("val-steps")) document.getElementById("val-steps").innerText = data.TTS_QUALITY_STEPS || 5;
        }
    });

    // Attach click-outside handler for the diagnostics modal
    const modal = document.getElementById('diagnostics-modal');
    if(modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === this) closeDiagnosticsModal();
        });
    }
});

// Dynamically inject categories into the sleek dropdown
function buildPresetMatrixUI() {
    const selector = document.getElementById('preset-selector');
    if (!selector) return;
    selector.innerHTML = ""; 

    SYSTEM_PRESETS.forEach((preset, index) => {
        let opt = document.createElement('option');
        opt.value = index;
        opt.innerText = `[${preset.category.toUpperCase()}] ${preset.target_mock}`;
        selector.appendChild(opt);
    });
}

function selectPreset(index) {
    if (!SYSTEM_PRESETS || SYSTEM_PRESETS.length === 0) return;
    selectedPresetIndex = parseInt(index);
    
    // Bind current ledger payload to the configuration form inputs
    document.getElementById("bench-vision-mock").value = SYSTEM_PRESETS[selectedPresetIndex].mock_vision_text;
    document.getElementById("bench-prompts").value = SYSTEM_PRESETS[selectedPresetIndex].prompt_text;
    document.getElementById("bench-target-mock").value = SYSTEM_PRESETS[selectedPresetIndex].target_mock;
}

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    
    let targetId = tabId === 'transcript' ? 'tab-transcript' : 'tab-' + tabId;
    let targetElement = document.getElementById(targetId);
    if(targetElement) targetElement.classList.add('active');
    
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }

    if (tabId === 'memory') eel.ui_fetch_memory()(renderMemory);
    if (tabId === 'transcript') eel.ui_fetch_logs()(renderLogs);
    if (tabId === 'vts') eel.ui_fetch_vts_state()(renderVTS);
}

function renderMemory(data) {
    document.getElementById("mem-short").innerText = data.short_term || "Context window empty.";
    document.getElementById("mem-long").innerText = data.long_term || "No disk memories found.";
}

let currentRawTranscript = "";

function renderLogs(logText) {
    currentRawTranscript = logText;
    let viewer = document.getElementById("logs-viewer");
    viewer.innerHTML = "";
    
    let lines = logText.split('\n');
    let displayHtml = "";
    
    lines.forEach(line => {
        if (!line.trim()) return;
        try {
            let data = JSON.parse(line);
            if (data.messages) {
                displayHtml += "<div style='margin-bottom: 15px; padding: 12px; background: #151522; border-radius: 8px; border: 1px solid var(--border-color);'>";
                data.messages.forEach(msg => {
                    let color = msg.role === 'user' ? '#3b82f6' : (msg.role === 'assistant' ? '#a78bfa' : '#64748b');
                    let name = msg.role.toUpperCase();
                    displayHtml += `<div style='margin-bottom: 6px;'>
                                      <strong style='color:${color};'>[${name}]:</strong> 
                                      <span style='color:#cdd6f4; line-height:1.4;'>${msg.content}</span>
                                    </div>`;
                });
                displayHtml += "</div>";
            }
        } catch (e) {
            console.error("Parse error on transcript line:", e);
        }
    });
    
    viewer.innerHTML = displayHtml || "<div style='color:#64748b;'>No transcript data found.</div>";
    viewer.scrollTop = viewer.scrollHeight;
}

function downloadTranscript() {
    if (!currentRawTranscript) return alert("No transcript data available to export.");
    let blob = new Blob([currentRawTranscript], { type: 'text/plain' });
    let a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'ada_finetune_dataset.jsonl';
    a.click();
}

function renderVTS(vtsData) {
    document.getElementById("vts-model").innerText = vtsData.model_name || "Unknown Model";
    document.getElementById("vts-socket").innerText = vtsData.connected ? "Connected" : "Disconnected";
    document.getElementById("vts-socket").style.color = vtsData.connected ? "#10b981" : "#ef4444";
}

function updateTempSetting(configKey, value, displayId, suffix = "") {
    document.getElementById(displayId).innerText = value + suffix;
    let parsedValue = value.includes('.') ? parseFloat(value) : parseInt(value);
    eel.ui_update_temp_config(configKey, parsedValue);
}

function saveConfigToDisk() {
    let settings = {
        "LLM_MIN_P": parseFloat(document.getElementById("sl-minp").value),
        "LLM_TOP_P": parseFloat(document.getElementById("sl-topp").value),
        "TTS_SPEED": parseFloat(document.getElementById("sl-speed").value),
        "TTS_QUALITY_STEPS": parseInt(document.getElementById("sl-steps").value)
    };
    
    eel.ui_save_config_to_disk(settings)(function(success) {
        if(success) alert("Config saved successfully!");
        else alert("Error saving config to disk.");
    });
}

eel.expose(update_chat);
function update_chat(role, text, isNewTurn = true, msgId = null, msgType = "normal", thought = "") {
    try {
        let chatBox = document.getElementById("chat-box");
        let existing = msgId ? document.getElementById(msgId) : null;
        
        if (existing) {
            if (msgType !== "normal") existing.classList.add(msgType + "-bubble");
            if (isNewTurn === false) existing.innerText += text;
            else existing.innerText = text;
            chatBox.scrollTop = chatBox.scrollHeight;
            return;
        }

        let div = document.createElement("div");
        if (msgId) div.id = msgId;
        
        if (role === "ada") div.className = "bubble msg-ada";
        else if (role === "user") div.className = "bubble msg-user";
        else div.className = "bubble msg-system";
        
        div.innerText = text;
        chatBox.appendChild(div);
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (e) {
        console.error("DOM Update Error:", e);
    }
}

eel.expose(update_telemetry_detailed);
function update_telemetry_detailed(engineName, statusText, colorHex, statsText) {
    let dot = document.getElementById("dot-" + engineName);
    let txt = document.getElementById("txt-" + engineName);
    let stats = document.getElementById("stats-" + engineName);
    if(dot) dot.style.backgroundColor = colorHex;
    if(txt) { txt.innerText = statusText; txt.style.color = colorHex; }
    if (stats && statsText !== undefined && statsText !== "") stats.innerText = statsText;
}

function triggerSystemCoreShutdown() {
    let verify = confirm("Terminate full system pipeline?");
    if (verify) {
        eel.ui_hard_shutdown_pipeline();
    }
}

// ==================================================
// RADICALLY UPGRADED GRID SHOWCASE & QUALITY GRADER PIPELINE
// ==================================================
function setHeroState(state, title, subtitle) {
    let hero = document.getElementById("bench-hero");
    let t = document.getElementById("hero-main-title");
    let s = document.getElementById("hero-sub-title");
    let b = document.getElementById("hero-status-badge");
    let track = document.getElementById("hero-progress-track");
    let flow = document.getElementById("hero-pipeline-flow");

    t.innerText = title;
    s.innerText = subtitle;
    hero.className = "hero-container " + state;
    
    if(state === "running") {
        b.innerText = "Profiling Active";
        b.className = "hero-badge active-run";
        track.style.display = "block";
        flow.style.display = "flex";
    } else if (state === "success") {
        b.innerText = "Sequence Perfect";
        b.className = "hero-badge complete-run";
        track.style.display = "none";
        flow.style.display = "none";
    } else {
        b.innerText = "System Ready";
        b.className = "hero-badge";
        track.style.display = "none";
        flow.style.display = "none";
    }
}

function animateHeroProgress(callback) {
    let bar = document.getElementById("hero-progress-bar");
    let nodes = ["fn-scribe", "fn-retrieval", "fn-vllm", "fn-supertonic"];
    document.querySelectorAll('.flow-node').forEach(n => n.classList.remove('active'));
    bar.style.width = "0%";
    
    let currentStep = 0;
    let interval = setInterval(() => {
        if(currentStep < nodes.length) {
            document.getElementById(nodes[currentStep]).classList.add('active');
            bar.style.width = ((currentStep + 1) * 25) + "%";
            currentStep++;
        } else {
            clearInterval(interval);
            if(callback) callback();
        }
    }, 250);
}

// SINGLE CONSOLIDATED RENDERER: Combines Macro Math + Detailed Performance Panels
function renderVisualTelemetryShowcase(runsArray) {
    let panel = document.getElementById("metrics-view-panel");
    panel.innerHTML = "";
    if (!runsArray || runsArray.length === 0) return;

    // 1. Calculate Overall Suite Averages
    let totalTPS = 0, totalTTFT = 0, totalRTF = 0;
    let totalPipelineTime = 0, totalGenerationTime = 0, totalTokens = 0, totalPromptTokens = 0, totalCtxTokensFound = 0;
    let totalVisionTime = 0, totalVectorTime = 0, totalShortTermTokens = 0, totalKvCache = 0;
    let totalTtsSynthTime = 0, totalTtsSamples = 0;
    
    runsArray.forEach(run => {
        let metrics = run.ml_load_metrics || run;
        let latencies = run.system_latencies || run;
        let acoustics = run.tts_acoustics || run;

        totalTPS += (metrics.tps || 0);
        totalTTFT += (latencies.ttft || 0);
        totalRTF += (acoustics.tts_rtf || 0);
        totalPipelineTime += (latencies.total_pipeline_time || 0);
        totalGenerationTime += (latencies.generation_time || 0);
        totalTokens += (metrics.completion_tokens || 0);
        totalPromptTokens += (metrics.prompt_tokens || 0);
        totalCtxTokensFound += (metrics.context_tokens_found || 0);
        totalVisionTime += (latencies.vision_scribe_time || 0);
        totalVectorTime += (latencies.vector_retrieval_time || 0);
        totalShortTermTokens += (metrics.short_term_ctx_tokens || 0);
        totalKvCache += (metrics.kv_cache_utilization || 0);
        totalTtsSynthTime += (latencies.tts_synth_time || 0);
        totalTtsSamples += (acoustics.tts_sample_count || 0);
    });

    let numRuns = runsArray.length;
    let avgTPS = (totalTPS / numRuns).toFixed(1);
    let avgTTFT = (totalTTFT / numRuns).toFixed(3);
    let avgRTF = (totalRTF / numRuns).toFixed(3);
    let avgTPOT = totalTokens > 0 ? (totalGenerationTime / totalTokens).toFixed(4) : "0.0000";

    // Grab evaluations from the final run (which stores the macro grades)
    let finalRun = runsArray[runsArray.length - 1];
    let evalStatusCode = finalRun.evaluation?.status_code || "UNKNOWN";

    // 2. RENDER THE TOP HIGHLIGHT MINI DASHBOARD
    let summaryHtml = `
    <div class="summary-dashboard-grid">
        <div class="summary-mini-card">
            <span class="smc-label">Suite Velocity Avg</span>
            <span class="smc-value">${avgTPS} <span style="font-size:0.6em; color:var(--text-muted);">TPS</span></span>
        </div>
        <div class="summary-mini-card">
            <span class="smc-label">Suite Latency Avg</span>
            <span class="smc-value purple-metric">${avgTTFT}s <span style="font-size:0.55em; color:var(--text-muted);">TTFT</span></span>
        </div>
        <div class="summary-mini-card">
            <span class="smc-label">Suite Voice Factor</span>
            <span class="smc-value green-metric">${avgRTF} <span style="font-size:0.6em; color:var(--text-muted);">RTF</span></span>
        </div>
    </div>`;
    
    panel.innerHTML += summaryHtml;

    // Helper to build collapsible/static panels
    function buildPanel(title, rows, isCollapsible = false) {
        let rowsHtml = rows.map(r => `
            <div class="tg-row">
                <span class="tg-param">${r.label}</span>
                <span class="tg-value ${r.class || ''}">${r.value}</span>
            </div>
        `).join('');
        
        let headerHtml = isCollapsible 
            ? `<div class="tg-header collapsible" onclick="this.nextElementSibling.classList.toggle('collapsed'); this.querySelector('.tg-chevron').classList.toggle('collapsed');">
                   <span>${title}</span>
                   <span class="tg-chevron">&#9660;</span>
               </div>`
            : `<div class="tg-header"><span>${title}</span></div>`;

        let bodyWrapHtml = `<div class="tg-body-wrap"><div class="tg-body">${rowsHtml}</div></div>`;

        return `
        <div class="telemetry-group-panel">
            ${headerHtml}
            ${bodyWrapHtml}
        </div>`;
    }

    // 3. RESTORED & AVERAGED: CORE COMPUTE & MEMORY SPOOLS (Made Collapsible)
    panel.innerHTML += buildPanel("CORE COMPUTE & MEMORY SPOOLS (SUITE AVERAGES)", [
        { label: "Total Interconnect Loop Duration", value: (totalPipelineTime / numRuns).toFixed(4) + " s", class: "blue-metric" },
        { label: "System Loop Status Return Code", value: evalStatusCode, class: evalStatusCode === "COMPLETED_RUN" ? "success-text" : "danger-text" },
        { label: "Vision Frame Ingestion Delay", value: (totalVisionTime / numRuns).toFixed(4) + " s", class: "blue-metric" },
        { label: "Vector Index Embedding Space Lookup", value: (totalVectorTime / numRuns).toFixed(4) + " s", class: "blue-metric" },
        { label: "Short-Term Volatile Window Load", value: (totalShortTermTokens / numRuns).toFixed(0) + " tokens", class: "blue-metric" },
        { label: "Context Window Saturation Ceiling", value: (totalKvCache / numRuns).toFixed(1) + " %", class: "blue-metric" },
        { label: "Long-Term Associated Blocks Found", value: (totalCtxTokensFound / numRuns).toFixed(0) + " keys", class: "blue-metric" }
    ], true);

    // 4. RESTORED & AVERAGED: VLLM & AUDIO DSP TELEMETRY MATRICES (Made Collapsible)
    panel.innerHTML += buildPanel("VLLM & AUDIO DSP TELEMETRY MATRICES (SUITE AVERAGES)", [
        { label: "Pure Token Generation Duration", value: (totalGenerationTime / numRuns).toFixed(4) + " s", class: "blue-metric" },
        { label: "Time Per Output Token (TPOT)", value: avgTPOT + " s/token", class: "blue-metric" },
        { label: "Prompt Sequence Token Load", value: (totalPromptTokens / numRuns).toFixed(0) + " tokens", class: "blue-metric" },
        { label: "Outbound Production Payload Weight", value: (totalTokens / numRuns).toFixed(0) + " tokens", class: "blue-metric" },
        { label: "K-V Attention Compaction Ratio", value: (totalKvCache / numRuns).toFixed(2) + " %", class: "blue-metric" },
        { label: "Voice Synthesis Waveform Latency", value: (totalTtsSynthTime / numRuns).toFixed(4) + " s", class: "blue-metric" },
        { label: "Signal Processing Audio Sample Count", value: (totalTtsSamples / numRuns).toFixed(0) + " samples", class: "blue-metric" }
    ], true);

    // 5. RENDER THE DOMAIN-SPECIFIC CATEGORY MATRICES WITH DIAGNOSTIC PILLS
    if (finalRun.quality_scores && finalRun.quality_scores.category_data) {
        let cats = finalRun.quality_scores.category_data;
        let catHtml = `<div class="telemetry-group-panel">
            <div class="tg-header"><span>DOMAIN-SPECIFIC REGRESSION BREAKDOWN</span></div>
            <div class="tg-body" style="padding: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; background: #08090e;">`;

        for (let key in cats) {
            let cData = cats[key];
            let adherencePct = ((cData.adherenceHits / cData.count) * 100).toFixed(0);
            let catTps = (cData.totalTPS / cData.count).toFixed(1);
            let catTtft = (cData.totalTTFT / cData.count).toFixed(3);
            let totalFlags = cData.slaViolations + cData.funcFailures;
            
            let statusColor = totalFlags === 0 ? "var(--accent-green)" : (cData.funcFailures > 0 ? "var(--accent-red)" : "var(--accent-amber)");
            let statusBg = totalFlags === 0 ? "rgba(16, 185, 129, 0.05)" : (cData.funcFailures > 0 ? "rgba(239, 68, 68, 0.05)" : "rgba(245, 158, 11, 0.05)");
            let statusText = totalFlags === 0 ? "SYS_STABLE" : `[${totalFlags}] FAULTS`;
            let cleanKey = key.replace(/_/g, ' ').toUpperCase();

            // Construct explicit descriptive problem messages based on mapped counts
            let flagsList = [];
            if (cData.ttftFailures > 0) flagsList.push(`TTFT Exceeded (${cData.ttftFailures}/${cData.count} runs > 400ms)`);
            if (cData.rtfFailures > 0) flagsList.push(`Audio Latency Trip (${cData.rtfFailures}/${cData.count} runs RTF > 0.10)`);
            if (cData.tpsFailures > 0) flagsList.push(`Throughput Dropped (${cData.tpsFailures}/${cData.count} runs < 95 TPS)`);
            if (cData.semanticFailures > 0) flagsList.push(`Poor Semantic Match (${cData.semanticFailures}/${cData.count} runs < 7.0/10)`);
            if (cData.crashFailures > 0) flagsList.push(`Pipeline Timeout/Crash (${cData.crashFailures}/${cData.count} runs broken)`);
            if (cData.entropyFailures > 0) flagsList.push(`Context Sync Miss (${cData.entropyFailures}/${cData.count} runs high entropy)`);

            let flagPills = "";
            if (flagsList.length > 0) {
                flagPills = flagsList.map(f => 
                    `<span style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.25); padding: 4px 10px; border-radius: 4px; font-size: 0.76em; font-weight: 700; color: #ff6b6b; display: inline-block; margin: 3px 6px 3px 0; font-family: monospace;">${f}</span>`
                ).join('');
            } else {
                flagPills = `<span style="color: var(--accent-green); font-size: 0.8em; font-weight: 700; font-family: monospace;">✓ All automated verification pipelines passing smoothly</span>`;
            }

            catHtml += `
            <div onclick="openDiagnosticsModal('${key}')" 
                 style="cursor: pointer; background: var(--bg-surface); border: 1px solid var(--border-color); border-left: 3px solid ${statusColor}; border-radius: 6px; padding: 14px; position: relative; overflow: hidden; display: flex; flex-direction: column; justify-content: space-between; transition: transform 0.1s, box-shadow 0.1s;"
                 onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)';"
                 onmouseout="this.style.transform='none'; this.style.boxShadow='none';">
                <div style="position: absolute; top: 0; right: 0; padding: 4px 10px; background: ${statusBg}; color: ${statusColor}; font-size: 0.65em; font-weight: 800; font-family: monospace; border-bottom-left-radius: 6px; border-left: 1px solid var(--border-color); border-bottom: 1px solid var(--border-color);">
                    ${statusText}
                </div>
                
                <div style="margin-bottom: 12px; padding-right: 70px;">
                    <div style="font-weight: 800; font-size: 0.85em; color: #fff; letter-spacing: 0.5px;">${cleanKey}</div>
                    <div style="font-size: 0.65em; color: var(--text-muted); font-family: monospace; text-transform: uppercase;">Batch Load: ${cData.count} runs</div>
                </div>

                <div style="display: flex; gap: 15px; border-top: 1px dashed rgba(36, 41, 66, 0.6); padding-top: 12px; padding-bottom: 8px;">
                    <div style="flex: 1; display:flex; flex-direction:column; gap:4px;">
                        <span style="color:var(--text-muted); font-size:0.65em; text-transform:uppercase; font-weight:800; letter-spacing: 0.5px;">Adherence Target</span>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-family:monospace; color:${adherencePct >= 80 ? 'var(--accent-green)' : (adherencePct > 0 ? 'var(--accent-amber)' : 'var(--accent-red)')}; font-weight:800; font-size:1.1em;">${adherencePct}%</span>
                            <div style="flex-grow: 1; height: 4px; background: #0c0e17; border-radius: 2px; overflow: hidden;">
                                <div style="height: 100%; width: ${adherencePct}%; background: ${adherencePct >= 80 ? 'var(--accent-green)' : (adherencePct > 0 ? 'var(--accent-amber)' : 'var(--accent-red)')};"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- DETAILED METRIC FAULT DESCRIPTIONS -->
                <div style="margin-bottom: 12px; display: flex; flex-wrap: wrap;">
                    ${flagPills}
                </div>

                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 0.82em; background: #08090e; padding: 10px; border-radius: 4px; border: 1px solid var(--border-color);">
                    <div style="display:flex; flex-direction:column; gap:2px;">
                        <span style="color:var(--text-muted); font-size:0.7em; text-transform:uppercase; font-weight:700;">Avg Velocity</span>
                        <span style="font-family:monospace; color:#cdd6f4; font-weight:700; font-size:0.95em;">${catTps} <span style="font-size: 0.7em; color: #64748b;">t/s</span></span>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:2px;">
                        <span style="color:var(--text-muted); font-size:0.7em; text-transform:uppercase; font-weight:700;">Avg TTFT</span>
                        <span style="font-family:monospace; color:#cdd6f4; font-weight:700; font-size:0.95em;">${catTtft} <span style="font-size: 0.7em; color: #64748b;">s</span></span>
                    </div>
                </div>
            </div>`;
        }
        
        catHtml += `</div></div>`;
        panel.innerHTML += catHtml;
    }

    // 6. OVERALL SUITE VERDICT
    if (finalRun.quality_scores) {
        let q = finalRun.quality_scores;
        panel.innerHTML += buildPanel("MACRO SUITE POST-EXECUTION ANALYTICS", [
            { label: "Semantic Ground-Truth Accuracy", value: q.accuracy.toFixed(1) + " / 10.0", class: q.accuracy >= 8.5 ? "success-text" : "" },
            { label: "Hardware SLA Alignment Metric", value: q.alignment.toFixed(1) + " / 10.0", class: "blue-metric" },
            { label: "Pipeline Cognitive Stability", value: q.structural.toFixed(1) + " / 10.0", class: "blue-metric" },
            { label: "Aggregate Macro Verdict", value: q.verdict, class: "success-text" }
        ], false);
    }
}

// ==================================================
// ADVANCED MULTI-STEP CATEGORICAL GRADING ENGINE
// ==================================================
function computeQualityScores(runs) {
    if (!runs || runs.length === 0) return null;

    let totalSLA = 0;
    let totalFunctional = 0;
    let totalContext = 0;
    let totalSlaViolations = 0;
    let totalFunctionalFailures = 0;

    const BUDGETS = {
        max_ttft: 0.400,          
        max_tts_rtf: 0.100,       
        min_tps: 95.0             
    };

    let categories = {};

    // First Pass: Initialize data structures with counter tracking arrays
    runs.forEach(run => {
        let cat = run.category || "Uncategorized_Test";
        if (!categories[cat]) {
            categories[cat] = {
                count: 0, totalTPS: 0, totalTTFT: 0,
                slaViolations: 0, funcFailures: 0, adherenceHits: 0,
                ttftFailures: 0, rtfFailures: 0, tpsFailures: 0,
                semanticFailures: 0, crashFailures: 0, entropyFailures: 0
            };
        }
    });

    runs.forEach(run => {
        let latencies = run.system_latencies || {};
        let mlMetrics = run.ml_load_metrics || {};
        let acoustics = run.tts_acoustics || {};
        let evaluation = run.evaluation || {};
        let cat = run.category || "Uncategorized_Test";

        let runSLA = 10.0;
        let runFunctional = 10.0;
        let runContext = 10.0;
        
        let runSlaViolations = 0;
        let runFuncFailures = 0;

        // --- EVALUATE LATENCY & THROUGHPUT BUDGETS ---
        if (latencies.ttft > BUDGETS.max_ttft) { runSLA -= 3.5; runSlaViolations++; categories[cat].ttftFailures++; }
        if (acoustics.tts_rtf > BUDGETS.max_tts_rtf) { runSLA -= 2.5; runSlaViolations++; categories[cat].rtfFailures++; }
        if (mlMetrics.tps < BUDGETS.min_tps) { runSLA -= 4.0; runSlaViolations++; categories[cat].tpsFailures++; }
        runSLA = Math.max(0, runSLA);

        // --- EVALUATE FUNCTIONAL INTEGRITY ---
        let semanticScore = evaluation.target_convergence_accuracy || 0.0;
        if (semanticScore < 7.0) {
            runFunctional -= (10.0 - semanticScore); 
            runFuncFailures++;
            categories[cat].semanticFailures++;
        }
        
        if (evaluation.status_code !== "COMPLETED_RUN") {
            runFunctional -= 10.0; 
            runFuncFailures++;
            categories[cat].crashFailures++;
        }
        runFunctional = Math.max(0, runFunctional);

        // --- EVALUATE CONTEXT STABILITY ---
        if (mlMetrics.prediction_entropy > 0.70 && mlMetrics.context_tokens_found === 0) {
            runContext -= 5.0;
            categories[cat].entropyFailures++;
        }
        runContext = Math.max(0, runContext);

        // --- AGGREGATE CORE STATS ---
        categories[cat].count++;
        categories[cat].totalTPS += (mlMetrics.tps || 0);
        categories[cat].totalTTFT += (latencies.ttft || 0);
        categories[cat].slaViolations += runSlaViolations;
        categories[cat].funcFailures += runFuncFailures;
        
        if (semanticScore >= 7.0 && evaluation.status_code === "COMPLETED_RUN") {
            categories[cat].adherenceHits++;
        }

        totalSLA += runSLA;
        totalFunctional += runFunctional;
        totalContext += runContext;
        totalSlaViolations += runSlaViolations;
        totalFunctionalFailures += runFuncFailures;
    });

    let processedCount = runs.length;
    let macroSLA = totalSLA / processedCount;
    let macroFunctional = totalFunctional / processedCount;
    let macroContext = totalContext / processedCount;
    let aggregateScore = (macroSLA + macroFunctional + macroContext) / 3;

    let verdict = "CRITICAL PERFORMANCE REGRESSION DETECTED";
    if (aggregateScore >= 9.2 && totalSlaViolations === 0 && totalFunctionalFailures === 0) {
        verdict = "PIPELINE VERIFIED: ZERO REGRESSIONS";
    } else if (aggregateScore >= 7.5 && totalFunctionalFailures === 0) {
        verdict = "FUNCTIONAL PASS WITH SYSTEM PERF DEGRADATION";
    } else if (totalFunctionalFailures > 0) {
        verdict = `REGRESSION WARNING: ${totalFunctionalFailures} FUNCTIONAL FAILURES RECORDED`;
    }

    return {
        accuracy: macroFunctional, 
        alignment: macroSLA,
        structural: macroContext,
        verdict: `${verdict} (${totalSlaViolations} SLA Trips)`,
        category_data: categories
    };
}

function runSingleBenchmark() {
    let targetPreset = SYSTEM_PRESETS[selectedPresetIndex];
    let promptText = document.getElementById("bench-prompts").value;
    
    if (!promptText.trim()) return alert("Instruction prompt required.");
    
    setHeroState("running", "Profiling Workload Preset", "Processing pipeline context frames, checking attention vectors...");
    
    animateHeroProgress(() => {
            eel.ui_run_monolithic_system_benchmark(
                targetPreset.mock_vision_text, 
                targetPreset.prompt_text, 
                "Suite Auto-Run", 
                targetPreset.target_mock,
                targetPreset.category // Pass the category to Python
            )(function(response) {
                if(!response.error) {
                    response.category = targetPreset.category; 
                    // Quick render for single run
                    let mockArray = [response];
                    window.globalRunData = mockArray; // Inject to global modal array
                    mockArray[0].quality_scores = computeQualityScores(mockArray);
                    renderVisualTelemetryShowcase(mockArray);
                }
                setHeroState("success", "Preset Profiled", "Single sequence test complete.");
            });
        });
}

function runSequentialTestSuite() {
    setHeroState("running", `Executing Full ${SYSTEM_PRESETS.length}x Preset Suite`, "Engaging multi-pass system processing loop...");
    
    let suiteResults = [];
    let currentIndex = 0;
    
    function runNext() {
        if(currentIndex >= SYSTEM_PRESETS.length) {
            // End of Suite: Trigger macro evaluation pass to grade all combined outputs
            window.globalRunData = suiteResults; // Inject to global modal array
            let coreGrades = computeQualityScores(suiteResults);
            
            // Apply evaluation pass to final item for showcase representation
            suiteResults[suiteResults.length - 1].quality_scores = coreGrades;
            
            renderVisualTelemetryShowcase(suiteResults);
            setHeroState("success", `${SYSTEM_PRESETS.length}-Preset Test Suite Complete`, "All responses collected. Macro grading evaluation pass completed successfully.");
            alert("Sequential suite execution complete. Graded output logs written.");
            return;
        }
        
        let targetPreset = SYSTEM_PRESETS[currentIndex];
        selectPreset(currentIndex);
        
        animateHeroProgress(() => {
            eel.ui_run_monolithic_system_benchmark(
                targetPreset.mock_vision_text, 
                targetPreset.prompt_text, 
                "Suite Auto-Run", 
                targetPreset.target_mock,
                targetPreset.category // Pass the category to Python
            )(function(response) {
                if(!response.error) {
                    // Re-attach the category from the local preset array
                    response.category = targetPreset.category; 
                    suiteResults.push(response);
                }
                currentIndex++;
                runNext();
            });
        });
    }
    
    runNext();
}

// ==================================================
// MODAL CONTROLLERS & DATA INJECTION
// ==================================================
function openDiagnosticsModal(categoryName) {
    const modal = document.getElementById('diagnostics-modal');
    const title = document.getElementById('modal-category-title');
    const body = document.getElementById('modal-body');
    
    title.innerText = categoryName.replace(/_/g, ' ').toUpperCase();
    
    // Filter the raw runs that belong specifically to this category
    const relevantRuns = window.globalRunData.filter(r => (r.category || "Uncategorized_Test") === categoryName);
    
    document.getElementById('modal-category-subtitle').innerText = `Batch Load: ${relevantRuns.length} EXECUTIONS`;
    
    let htmlContent = "";
    
    relevantRuns.forEach((run, index) => {
        let l = run.system_latencies || {};
        let m = run.ml_load_metrics || {};
        let t = run.tts_acoustics || {};
        let e = run.evaluation || {};
        let p = run.input_parameters || {};
        let txt = run.text_data || {};
        
        let evalColor = e.target_convergence_accuracy >= 7.0 ? "var(--accent-green)" : "var(--accent-red)";
        
        htmlContent += `
        <div class="run-block">
            <div class="run-header">
                <span style="font-family: monospace; font-weight: 800; color: #fff;">EXECUTION_ID: #${index + 1}</span>
                <span style="font-family: monospace; font-size: 0.85em; font-weight: 700; color: ${evalColor}; background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 4px;">
                    ACCURACY: ${e.target_convergence_accuracy}/10.0
                </span>
            </div>
            
            <div class="run-grid">
                <!-- LATENCY & COMPUTE -->
                <div class="data-panel">
                    <div class="data-panel-title">System Latencies</div>
                    <div class="data-row"><span class="data-label">Time To First Token</span><span class="data-value">${(l.ttft || 0).toFixed(3)}s</span></div>
                    <div class="data-row"><span class="data-label">Generation Time</span><span class="data-value">${(l.generation_time || 0).toFixed(3)}s</span></div>
                    <div class="data-row"><span class="data-label">Vision Scribe Time</span><span class="data-value">${(l.vision_scribe_time || 0).toFixed(3)}s</span></div>
                    <div class="data-row"><span class="data-label">Audio Synth Time</span><span class="data-value">${(l.tts_synth_time || 0).toFixed(3)}s</span></div>
                    <div class="data-row" style="border-top: 1px solid rgba(255,255,255,0.05); margin-top: 6px; padding-top: 6px;">
                        <span class="data-label" style="color: #fff;">Total Pipeline</span>
                        <span class="data-value" style="color: #fff;">${(l.total_pipeline_time || 0).toFixed(3)}s</span>
                    </div>
                </div>

                <!-- LLM & ML METRICS -->
                <div class="data-panel">
                    <div class="data-panel-title">Machine Learning Load</div>
                    <div class="data-row"><span class="data-label">Velocity (TPS)</span><span class="data-value" style="color: #a6e3a1;">${m.tps || 0} t/s</span></div>
                    <div class="data-row"><span class="data-label">Tokens (Prompt / Comp)</span><span class="data-value">${m.prompt_tokens || 0} / ${m.completion_tokens || 0}</span></div>
                    <div class="data-row"><span class="data-label">KV Cache Util</span><span class="data-value">${m.kv_cache_utilization || 0}%</span></div>
                    <div class="data-row"><span class="data-label">Prediction Entropy</span><span class="data-value">${m.prediction_entropy || 0}</span></div>
                    <div class="data-row"><span class="data-label">TTS Real-Time Factor</span><span class="data-value">${(t.tts_rtf || 0).toFixed(4)}</span></div>
                </div>
            </div>

            <!-- I/O TEXT DUMPS -->
            <div style="margin-top: 15px;">
                <div class="data-panel-title">Input Prompt</div>
                <div class="text-dump">${p.prompt_text || "No prompt recorded."}</div>
            </div>
            
            <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <div class="data-panel-title">Target Mock (Ground Truth)</div>
                    <div class="text-dump" style="border-color: rgba(16, 185, 129, 0.2);">${p.target_mock || "N/A"}</div>
                </div>
                <div>
                    <div class="data-panel-title">Actual Model Output</div>
                    <div class="text-dump" style="border-color: rgba(59, 130, 246, 0.2);">${txt.model_response || "No response generated."}</div>
                </div>
            </div>
        </div>`;
    });

    body.innerHTML = htmlContent;
    modal.style.display = 'flex';
}

function closeDiagnosticsModal() {
    document.getElementById('diagnostics-modal').style.display = 'none';
}