// --- AUTO-INITIALIZE ACTUAL VARIABLES ON BOOT ---
window.addEventListener("DOMContentLoaded", () => {
    eel.ui_get_current_settings()(function(data) {
        document.getElementById("sl-minp").value = data.LLM_MIN_P;
        document.getElementById("val-minp").innerText = data.LLM_MIN_P;

        document.getElementById("sl-topp").value = data.LLM_TOP_P;
        document.getElementById("val-topp").innerText = data.LLM_TOP_P;

        document.getElementById("sl-speed").value = data.TTS_SPEED;
        document.getElementById("val-speed").innerText = data.TTS_SPEED + "x";

        document.getElementById("sl-steps").value = data.TTS_QUALITY_STEPS;
        document.getElementById("val-steps").innerText = data.TTS_QUALITY_STEPS;
    });
});

// --- TAB NAVIGATION & AUTO-FETCHING ---
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    
    // Remap logs request target to handle tab string swap safely
    let targetId = tabId === 'transcript' ? 'tab-transcript' : 'tab-' + tabId;
    document.getElementById(targetId).classList.add('active');
    event.currentTarget.classList.add('active');

    if (tabId === 'memory') eel.ui_fetch_memory()(renderMemory);
    if (tabId === 'transcript') eel.ui_fetch_logs()(renderLogs);
    if (tabId === 'vts') eel.ui_fetch_vts_state()(renderVTS);
}

// --- RENDER CALLBACKS ---
function renderMemory(data) {
    document.getElementById("mem-short").innerText = data.short_term || "Context window empty.";
    document.getElementById("mem-long").innerText = data.long_term || "No disk memories found.";
}

let currentRawTranscript = "";

function renderLogs(logText) {
    currentRawTranscript = logText; // Save the raw JSONL for the export button
    let viewer = document.getElementById("logs-viewer");
    viewer.innerHTML = "";
    
    let lines = logText.split('\n');
    let displayHtml = "";
    
    lines.forEach(line => {
        if (!line.trim()) return;
        try {
            let data = JSON.parse(line);
            if (data.messages) {
                // Create a container block for each full conversation session
                displayHtml += "<div style='margin-bottom: 15px; padding: 12px; background: #151522; border-radius: 8px; border: 1px solid #242936;'>";
                
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

// Triggers a native browser file download of the raw JSONL dataset
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

// --- CONFIGURATION LOGIC ---
// Fires instantly when a slider is dragged
function updateTempSetting(configKey, value, displayId, suffix = "") {
    document.getElementById(displayId).innerText = value + suffix;
    
    // Cast appropriately: Steps and Tokens are ints, Temp/Speed are floats
    let parsedValue = value.includes('.') ? parseFloat(value) : parseInt(value);
    
    // Send to python to modify the current runtime memory ONLY
    eel.ui_update_temp_config(configKey, parsedValue);
}

// Gathers all slider values and tells python to write them to the file
function saveConfigToDisk() {
    let settings = {
        "LLM_MIN_P": parseFloat(document.getElementById("sl-minp").value),
        "LLM_TOP_P": parseFloat(document.getElementById("sl-topp").value),
        "TTS_SPEED": parseFloat(document.getElementById("sl-speed").value),
        "TTS_QUALITY_STEPS": parseInt(document.getElementById("sl-steps").value)
    };
    
    eel.ui_save_config_to_disk(settings)(function(success) {
        if(success) alert("Config saved successfully! Your hardcoded settings are updated.");
        else alert("Error saving config to disk.");
    });
}

// --- CHAT & TELEMETRY ---
eel.expose(update_chat);
function update_chat(role, text, isNewTurn = true) {
    let chatBox = document.getElementById("chat-box");
    if (isNewTurn === "false" || isNewTurn === 0) isNewTurn = false;
    if (isNewTurn === "true" || isNewTurn === 1) isNewTurn = true;

    if (!isNewTurn && role === "ada") {
        let bubbles = document.querySelectorAll('.msg-ada');
        if (bubbles.length > 0) {
            let lastBubble = bubbles[bubbles.length - 1];
            lastBubble.innerText += " " + text;
            chatBox.scrollTop = chatBox.scrollHeight;
            return;
        }
    }
    
    let div = document.createElement("div");
    if (role === "ada") div.className = "bubble msg-ada";
    else if (role === "user") div.className = "bubble msg-user";
    else div.className = "bubble msg-system";
    
    div.innerText = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Replaces the old update_telemetry with detailed stat targeting
eel.expose(update_telemetry_detailed);
function update_telemetry_detailed(engineName, statusText, colorHex, statsText) {
    document.getElementById("dot-" + engineName).style.backgroundColor = colorHex;
    
    let statusEl = document.getElementById("txt-" + engineName);
    statusEl.innerText = statusText;
    statusEl.style.color = colorHex;

    // Only update the stats readout if data was explicitly provided
    if (statsText !== undefined && statsText !== "") {
        document.getElementById("stats-" + engineName).innerText = statsText;
    }
}

eel.expose(update_vision_card);
function update_vision_card(summaryText) {
    document.getElementById("vision-summary").innerText = summaryText;
}

// --- SECURE PIPELINE SYSTEM TEARDOWN ---
function triggerSystemCoreShutdown() {
    // Structural confirmation window prevent tracking accidental thread terminations
    let verify = confirm("WARNING: This will immediately close the user interface dashboard AND stop all underlying Docker engine container nodes. Proceed with full system shutdown?");
    
    if (verify) {
        document.body.innerHTML = "<div style='display:flex; height:100vh; align-items:center; justify-content:center; font-family:sans-serif; color:#7f849c; font-weight:600;'>System Core Terminating Nodes... You can close this window now.</div>";
        eel.ui_hard_shutdown_pipeline();
    }
}