/**
 * Sarcasm & Hate Speech Neural Studio — Frontend Controller
 * Handles interactive classification, live language preview, animated diagnostics,
 * batch evaluation suite, and query history persistence.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Navigation & System
    const deviceStatusBadge = document.getElementById("deviceStatusBadge");
    const deviceStatusText = document.getElementById("deviceStatusText");
    const btnOpenArchitecture = document.getElementById("btnOpenArchitecture");
    const architectureModal = document.getElementById("architectureModal");
    const btnCloseModal = document.getElementById("btnCloseModal");
    const btnDismissModal = document.getElementById("btnDismissModal");

    // View Tabs
    const tabSingle = document.getElementById("tabSingle");
    const tabBatch = document.getElementById("tabBatch");
    const singleView = document.getElementById("singleView");
    const batchView = document.getElementById("batchView");

    // Studio Input Elements
    const inputText = document.getElementById("inputText");
    const charCount = document.getElementById("charCount");
    const liveScriptIndicator = document.getElementById("liveScriptIndicator");
    const liveScriptText = document.getElementById("liveScriptText");
    const btnClearText = document.getElementById("btnClearText");
    const btnPasteText = document.getElementById("btnPasteText");
    const btnAnalyze = document.getElementById("btnAnalyze");
    const sampleChipsContainer = document.getElementById("sampleChipsContainer");

    // Studio Output Elements
    const emptyState = document.getElementById("emptyState");
    const activeResult = document.getElementById("activeResult");
    const verdictBanner = document.getElementById("verdictBanner");
    const verdictCategory = document.getElementById("verdictCategory");
    const verdictLanguage = document.getElementById("verdictLanguage");
    const verdictSeverity = document.getElementById("verdictSeverity");
    const verdictIconContainer = document.getElementById("verdictIconContainer");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictDescription = document.getElementById("verdictDescription");
    const verdictNoteBox = document.getElementById("verdictNoteBox");
    const verdictNoteText = document.getElementById("verdictNoteText");

    // Flow Pipeline Nodes
    const nodeSarcasm = document.getElementById("nodeSarcasm");
    const nodeHate = document.getElementById("nodeHate");

    // Diagnostics - Sarcasm
    const sarcasmModelName = document.getElementById("sarcasmModelName");
    const sarcasmStatusBadge = document.getElementById("sarcasmStatusBadge");
    const sarcasmScoreVal = document.getElementById("sarcasmScoreVal");
    const sarcasmMeterFill = document.getElementById("sarcasmMeterFill");
    const sarcasmNeedle = document.getElementById("sarcasmNeedle");
    const sarcasmDecisionText = document.getElementById("sarcasmDecisionText");

    // Diagnostics - Hate & Offensive
    const hateStatusBadge = document.getElementById("hateStatusBadge");
    const hateProbaContainer = document.getElementById("hateProbaContainer");
    const hateUnsupportedBox = document.getElementById("hateUnsupportedBox");
    const barHate = document.getElementById("barHate");
    const barOffensive = document.getElementById("barOffensive");
    const barNeither = document.getElementById("barNeither");
    const probaHatePct = document.getElementById("probaHatePct");
    const probaOffensivePct = document.getElementById("probaOffensivePct");
    const probaNeitherPct = document.getElementById("probaNeitherPct");

    // Linguistic Markers
    const markersCard = document.getElementById("markersCard");
    const markersList = document.getElementById("markersList");

    // Session History Table
    const historyTableBody = document.getElementById("historyTableBody");
    const historyCountBadge = document.getElementById("historyCountBadge");
    const btnClearHistory = document.getElementById("btnClearHistory");
    const btnExportHistory = document.getElementById("btnExportHistory");

    // Batch Studio Elements
    const batchInputText = document.getElementById("batchInputText");
    const batchLineCount = document.getElementById("batchLineCount");
    const btnLoadBenchmarkSuite = document.getElementById("btnLoadBenchmarkSuite");
    const btnRunBatch = document.getElementById("btnRunBatch");
    const batchResultsContainer = document.getElementById("batchResultsContainer");
    const batchTableBody = document.getElementById("batchTableBody");
    const statTotalCount = document.getElementById("statTotalCount");
    const statHateCount = document.getElementById("statHateCount");
    const statSarcasmCount = document.getElementById("statSarcasmCount");
    const statNeutralCount = document.getElementById("statNeutralCount");

    // State Variables
    let sessionHistory = [];
    let presetSamples = [];
    const DEVANAGARI_REGEX = /[\u0900-\u097F]/;
    const HINGLISH_MARKERS = new Set([
        "hai", "hain", "kya", "nahi", "nhi", "kyun", "kyu", "yaar", "yr",
        "krte", "krta", "krna", "bhi", "toh", "aur", "mein", "wala",
        "wali", "bhai", "bahut", "acha", "accha", "theek", "matlab",
        "chahiye", "karo", "karta", "karti", "hoga", "hogi", "tha", "thi",
        "koi", "kuch", "sab", "log", "ghar", "paisa", "kaam", "chutiya",
        "yee", "pah", "hona", "aisa", "aaisa", "tumhare", "sath", "uski",
        "iske", "hota", "hoti", "isko", "aaj", "kal", "abhi", "waha",
        "yaha", "kahan", "kaise", "itna", "itni", "sala"
    ]);

    // =========================================================================
    // INITIALIZATION & HEALTH CHECK
    // =========================================================================

    async function checkSystemHealth() {
        try {
            const res = await fetch("/api/health");
            const data = await res.json();
            if (data.status === "ready") {
                deviceStatusBadge.className = "status-pill status-ready";
                const devText = data.gpu_name ? `Active (${data.gpu_name})` : `Active (${data.device})`;
                deviceStatusText.textContent = devText;
            } else {
                deviceStatusBadge.className = "status-pill status-error";
                deviceStatusText.textContent = "Pipeline Error";
            }
        } catch (err) {
            deviceStatusBadge.className = "status-pill status-error";
            deviceStatusText.textContent = "Offline / Connection Failed";
        }
    }

    async function loadPresetSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            presetSamples = data.samples || [];
            renderSampleChips(presetSamples);
        } catch (err) {
            console.error("Failed to load preset samples:", err);
        }
    }

    function renderSampleChips(samples) {
        sampleChipsContainer.innerHTML = "";
        samples.forEach(sample => {
            const chip = document.createElement("button");
            chip.className = "sample-chip";
            chip.setAttribute("type", "button");
            chip.innerHTML = `
                <span class="sample-chip-tag">${sample.lang}</span>
                <span>${sample.label}</span>
            `;
            chip.addEventListener("click", () => {
                inputText.value = sample.text;
                handleInputChange();
                runSingleClassification();
            });
            sampleChipsContainer.appendChild(chip);
        });
    }

    // =========================================================================
    // REAL-TIME INPUT & SCRIPT HEURISTIC
    // =========================================================================

    function detectScriptHeuristic(text) {
        if (!text.trim()) return "en";
        if (DEVANAGARI_REGEX.test(text)) return "devanagari";

        const words = text.toLowerCase().match(/[a-zA-Z]+/g) || [];
        if (!words.length) return "en";

        let hits = 0;
        for (const w of words) {
            if (HINGLISH_MARKERS.has(w)) hits++;
        }

        if (hits >= 1 && (hits / words.length) >= 0.12) {
            return "hinglish";
        }
        return "en";
    }

    function handleInputChange() {
        const text = inputText.value;
        charCount.textContent = `${text.length} chars`;

        const detected = detectScriptHeuristic(text);
        liveScriptIndicator.className = `script-badge script-${detected}`;
        
        const labels = {
            "en": "EN (Latin)",
            "hinglish": "Hinglish (Romanized)",
            "devanagari": "Hindi (Devanagari)"
        };
        liveScriptText.textContent = labels[detected] || "EN";
    }

    inputText.addEventListener("input", handleInputChange);

    btnClearText.addEventListener("click", () => {
        inputText.value = "";
        handleInputChange();
        inputText.focus();
    });

    btnPasteText.addEventListener("click", async () => {
        try {
            const text = await navigator.clipboard.readText();
            inputText.value = text;
            handleInputChange();
            inputText.focus();
        } catch (err) {
            alert("Clipboard access was blocked or is unavailable.");
        }
    });

    // Keyboard Shortcut (Ctrl + Enter)
    inputText.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            runSingleClassification();
        }
    });

    // =========================================================================
    // SINGLE TEXT CLASSIFICATION
    // =========================================================================

    async function runSingleClassification() {
        const text = inputText.value.trim();
        if (!text) {
            inputText.focus();
            return;
        }

        // Set Loading state
        btnAnalyze.disabled = true;
        btnAnalyze.classList.add("loading");
        const originalLabel = btnAnalyze.querySelector(".btn-label").textContent;
        btnAnalyze.querySelector(".btn-label").textContent = "Inferencing Dual Engine...";

        try {
            const res = await fetch("/api/classify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || "Server inference error");
            }

            const data = await res.json();
            renderClassificationResults(data);
            recordSessionHistory(data);

        } catch (err) {
            alert(`Classification failed: ${err.message}`);
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.classList.remove("loading");
            btnAnalyze.querySelector(".btn-label").textContent = originalLabel;
        }
    }

    btnAnalyze.addEventListener("click", runSingleClassification);

    // =========================================================================
    // RENDER RESULTS & DIAGNOSTICS
    // =========================================================================

    function renderClassificationResults(data) {
        // Toggle view from empty state to active result
        emptyState.classList.add("hidden");
        activeResult.classList.remove("hidden");

        const bucket = data.bucket_info || {};
        const color = bucket.color || "#8b5cf6";
        const glow = bucket.glow || "rgba(139, 92, 246, 0.4)";

        // 1. Verdict Hero Card
        verdictBanner.style.background = `linear-gradient(135deg, ${color}1a, rgba(14, 18, 28, 0.95))`;
        verdictBanner.style.borderColor = `${color}66`;
        verdictBanner.style.boxShadow = `0 12px 35px -8px ${glow}`;

        verdictCategory.textContent = bucket.category || "Classification";
        verdictLanguage.textContent = data.language_display || "English";
        
        verdictSeverity.textContent = bucket.severity ? `${bucket.severity} severity` : "Analysis";
        verdictSeverity.style.backgroundColor = `${color}25`;
        verdictSeverity.style.borderColor = `${color}55`;
        verdictSeverity.style.color = color;

        verdictTitle.textContent = bucket.title || data.bucket.toUpperCase();
        verdictTitle.style.color = "#ffffff";
        verdictDescription.textContent = bucket.description || "Classified using chained model pipeline.";

        // Verdict Icon
        verdictIconContainer.style.backgroundColor = `${color}25`;
        verdictIconContainer.style.borderColor = `${color}60`;
        verdictIconContainer.style.boxShadow = `0 0 25px ${glow}`;
        verdictIconContainer.innerHTML = getVerdictIconSvg(data.bucket);

        // Note Banner
        if (data.note) {
            verdictNoteBox.classList.remove("hidden");
            verdictNoteText.textContent = data.note;
        } else {
            verdictNoteBox.classList.add("hidden");
        }

        // 2. Chained Pipeline Flow Highlights
        nodeSarcasm.className = data.sarcasm.evaluated ? "flow-node active" : "flow-node";
        nodeHate.className = data.hate.evaluated ? "flow-node active" : "flow-node";

        // 3. Sarcasm Engine Diagnostics
        sarcasmModelName.textContent = data.sarcasm.model_name || "LinearSVC";
        if (data.sarcasm.evaluated) {
            const isSarcastic = data.sarcasm.is_sarcastic;
            const score = typeof data.sarcasm.score === "number" ? data.sarcasm.score : 0;
            
            sarcasmStatusBadge.textContent = isSarcastic ? "Sarcastic" : "Not Sarcastic";
            sarcasmStatusBadge.style.backgroundColor = isSarcastic ? "rgba(168, 85, 247, 0.2)" : "rgba(100, 116, 139, 0.2)";
            sarcasmStatusBadge.style.color = isSarcastic ? "#d8b4fe" : "#94a3b8";
            sarcasmStatusBadge.style.borderColor = isSarcastic ? "rgba(168, 85, 247, 0.4)" : "rgba(100, 116, 139, 0.3)";

            sarcasmScoreVal.textContent = (score >= 0 ? `+${score.toFixed(3)}` : score.toFixed(3));
            sarcasmDecisionText.textContent = isSarcastic 
                ? (score > 0.8 ? "Strong Sarcasm Signal" : "Moderate Sarcasm Signal")
                : "No Sarcasm Detected";

            // Map score to meter percentage:
            // Score ranges roughly from -2.0 to +2.0. Center (0.0) is 50%.
            const clamped = Math.max(-2, Math.min(2, score));
            const pct = ((clamped + 2) / 4) * 100;
            sarcasmNeedle.style.left = `${pct}%`;

            if (pct >= 50) {
                sarcasmMeterFill.style.left = "50%";
                sarcasmMeterFill.style.width = `${pct - 50}%`;
                sarcasmMeterFill.style.background = "linear-gradient(90deg, #8b5cf6, #c084fc)";
            } else {
                sarcasmMeterFill.style.left = `${pct}%`;
                sarcasmMeterFill.style.width = `${50 - pct}%`;
                sarcasmMeterFill.style.background = "linear-gradient(90deg, #64748b, #8b5cf6)";
            }
        } else {
            sarcasmStatusBadge.textContent = "Bypassed";
            sarcasmScoreVal.textContent = "N/A";
            sarcasmNeedle.style.left = "50%";
            sarcasmMeterFill.style.width = "0%";
            sarcasmDecisionText.textContent = "Script not evaluated for sarcasm";
        }

        // 4. Hate & Offensive Engine Diagnostics
        if (data.hate.evaluated && data.hate.probabilities) {
            hateProbaContainer.classList.remove("hidden");
            hateUnsupportedBox.classList.add("hidden");

            const probs = data.hate.probabilities;
            const hateVal = (probs.hate || 0) * 100;
            const offVal = (probs.offensive || 0) * 100;
            const neitherVal = (probs.neither || 0) * 100;

            barHate.style.width = `${hateVal.toFixed(1)}%`;
            probaHatePct.textContent = `${hateVal.toFixed(1)}%`;

            barOffensive.style.width = `${offVal.toFixed(1)}%`;
            probaOffensivePct.textContent = `${offVal.toFixed(1)}%`;

            barNeither.style.width = `${neitherVal.toFixed(1)}%`;
            probaNeitherPct.textContent = `${neitherVal.toFixed(1)}%`;

            const labelMap = {
                "hate": "Hate Speech",
                "offensive": "Offensive Language",
                "neither": "Neither (Clean)"
            };
            hateStatusBadge.textContent = labelMap[data.hate.label] || data.hate.label;
            
            if (data.hate.label === "hate") {
                hateStatusBadge.className = "badge-hate";
                hateStatusBadge.style.color = "#fca5a5";
                hateStatusBadge.style.backgroundColor = "rgba(244, 63, 94, 0.2)";
            } else if (data.hate.label === "offensive") {
                hateStatusBadge.className = "badge-hate";
                hateStatusBadge.style.color = "#fde68a";
                hateStatusBadge.style.backgroundColor = "rgba(245, 158, 11, 0.2)";
            } else {
                hateStatusBadge.className = "badge-hate";
                hateStatusBadge.style.color = "#a7f3d0";
                hateStatusBadge.style.backgroundColor = "rgba(16, 185, 129, 0.2)";
            }

        } else {
            hateProbaContainer.classList.add("hidden");
            hateUnsupportedBox.classList.remove("hidden");
            hateStatusBadge.textContent = "English Only";
            hateStatusBadge.style.backgroundColor = "rgba(255, 255, 255, 0.05)";
            hateStatusBadge.style.color = "var(--text-muted)";
        }

        // 5. Matched Markers
        if (data.matched_markers && data.matched_markers.length > 0) {
            markersCard.classList.remove("hidden");
            markersList.innerHTML = "";
            data.matched_markers.forEach(token => {
                const span = document.createElement("span");
                span.className = "marker-token";
                span.textContent = token;
                markersList.appendChild(span);
            });
        } else {
            markersCard.classList.add("hidden");
        }

        // Refresh Lucide icons inside dynamically rendered nodes
        lucide.createIcons();
    }

    function getVerdictIconSvg(bucket) {
        const iconMap = {
            "hate_sarcastic": `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m15.5 14 2 2"/><path d="m12 12-2 2"/><path d="M8.5 8.5 7 10"/><path d="M12 2C8 6 6 8.5 6 12a6 6 0 0 0 12 0c0-1.5-.5-3-1.5-4.5"/><path d="m2 2 20 20"/></svg>`,
            "hate": `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/></svg>`,
            "offensive_sarcastic": `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
            "offensive": `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
            "sarcastic": `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>`,
            "neutral": `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>`,
            "unsupported": `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
        };
        return iconMap[bucket] || iconMap["neutral"];
    }

    // =========================================================================
    // SESSION QUERY HISTORY
    // =========================================================================

    function recordSessionHistory(data) {
        sessionHistory.unshift({
            id: Date.now(),
            text: data.text,
            language: data.language_display || data.language,
            bucket: data.bucket,
            bucket_badge: (data.bucket_info && data.bucket_info.badge) || data.bucket,
            color: (data.bucket_info && data.bucket_info.color) || "#8b5cf6",
            is_sarcastic: data.sarcasm.is_sarcastic
        });

        renderHistoryTable();
    }

    function renderHistoryTable() {
        historyCountBadge.textContent = `${sessionHistory.length} entries`;

        if (!sessionHistory.length) {
            historyTableBody.innerHTML = `
                <tr class="empty-history-row">
                    <td colspan="5">No queries analyzed in this session yet. Run a classification above to track results.</td>
                </tr>
            `;
            return;
        }

        historyTableBody.innerHTML = "";
        sessionHistory.forEach(item => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>
                    <div class="table-text-preview" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</div>
                </td>
                <td>
                    <span class="script-badge">${item.language}</span>
                </td>
                <td>
                    <span class="bucket-pill" style="background-color: ${item.color}25; color: ${item.color}; border: 1px solid ${item.color}45;">
                        ${item.bucket_badge}
                    </span>
                </td>
                <td>
                    ${item.is_sarcastic === true ? '<span style="color:#d8b4fe; font-weight:600;">Yes</span>' : 
                      item.is_sarcastic === false ? '<span style="color:#94a3b8;">No</span>' : '<span style="color:#64748b;">—</span>'}
                </td>
                <td>
                    <button class="tool-btn btn-reuse" data-text="${escapeHtml(item.text)}" title="Load into Studio">
                        <i data-lucide="arrow-up-right" class="icon-xs"></i>
                        <span>Load</span>
                    </button>
                </td>
            `;

            tr.querySelector(".btn-reuse").addEventListener("click", () => {
                inputText.value = item.text;
                handleInputChange();
                window.scrollTo({ top: 0, behavior: "smooth" });
                runSingleClassification();
            });

            historyTableBody.appendChild(tr);
        });

        lucide.createIcons();
    }

    btnClearHistory.addEventListener("click", () => {
        sessionHistory = [];
        renderHistoryTable();
    });

    btnExportHistory.addEventListener("click", () => {
        if (!sessionHistory.length) {
            alert("No history available to export.");
            return;
        }
        const blob = new Blob([JSON.stringify(sessionHistory, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `sarcasm_hate_analysis_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // =========================================================================
    // BATCH EVALUATION SUITE
    // =========================================================================

    function updateBatchLineCount() {
        const lines = batchInputText.value
            .split("\n")
            .map(l => l.trim())
            .filter(l => l.length > 0);
        batchLineCount.textContent = `${lines.length} sentences`;
    }

    batchInputText.addEventListener("input", updateBatchLineCount);

    btnLoadBenchmarkSuite.addEventListener("click", () => {
        const suite = presetSamples.map(s => s.text).join("\n");
        batchInputText.value = suite;
        updateBatchLineCount();
    });

    btnRunBatch.addEventListener("click", async () => {
        const lines = batchInputText.value
            .split("\n")
            .map(l => l.trim())
            .filter(l => l.length > 0);

        if (!lines.length) {
            alert("Please enter at least one sentence to evaluate in batch.");
            batchInputText.focus();
            return;
        }

        btnRunBatch.disabled = true;
        btnRunBatch.innerHTML = `<span>Inferencing Batch (${lines.length})...</span>`;

        try {
            const res = await fetch("/api/batch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ texts: lines })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || "Batch processing error");
            }

            const data = await res.json();
            renderBatchResults(data.results || []);

        } catch (err) {
            alert(`Batch evaluation failed: ${err.message}`);
        } finally {
            btnRunBatch.disabled = false;
            btnRunBatch.innerHTML = `
                <i data-lucide="play" class="icon-sm"></i>
                <span>Run Batch Classifier</span>
            `;
            lucide.createIcons();
        }
    });

    function renderBatchResults(results) {
        batchResultsContainer.classList.remove("hidden");
        batchTableBody.innerHTML = "";

        let hateCount = 0;
        let sarcasmCount = 0;
        let neutralCount = 0;

        results.forEach(r => {
            if (r.bucket.includes("hate") || r.bucket.includes("offensive")) hateCount++;
            if (r.is_sarcastic) sarcasmCount++;
            if (r.bucket === "neutral") neutralCount++;

            const cfg = r.bucket_info || {};
            const color = cfg.color || "#8b5cf6";

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>
                    <div class="table-text-preview" title="${escapeHtml(r.text)}">${escapeHtml(r.text)}</div>
                </td>
                <td>
                    <span class="script-badge">${r.language}</span>
                </td>
                <td>
                    <span class="bucket-pill" style="background-color: ${color}25; color: ${color}; border: 1px solid ${color}45;">
                        ${cfg.badge || r.bucket}
                    </span>
                </td>
                <td>
                    <span style="font-family: var(--font-mono); font-size: 0.78rem;">
                        ${r.hate_label ? `Hate: <strong>${r.hate_label}</strong> ` : ''}
                        ${r.is_sarcastic !== null ? `| Sarcastic: <strong>${r.is_sarcastic ? 'Yes' : 'No'}</strong>` : ''}
                    </span>
                </td>
            `;
            batchTableBody.appendChild(tr);
        });

        statTotalCount.textContent = `Total: ${results.length}`;
        statHateCount.textContent = `Hate/Offensive: ${hateCount}`;
        statSarcasmCount.textContent = `Sarcastic: ${sarcasmCount}`;
        statNeutralCount.textContent = `Neutral: ${neutralCount}`;

        lucide.createIcons();
    }

    // =========================================================================
    // VIEW SWITCHING (SINGLE VS BATCH)
    // =========================================================================

    tabSingle.addEventListener("click", () => {
        tabSingle.classList.add("active");
        tabSingle.setAttribute("aria-selected", "true");
        tabBatch.classList.remove("active");
        tabBatch.setAttribute("aria-selected", "false");

        singleView.classList.add("active");
        batchView.classList.remove("active");
    });

    tabBatch.addEventListener("click", () => {
        tabBatch.classList.add("active");
        tabBatch.setAttribute("aria-selected", "true");
        tabSingle.classList.remove("active");
        tabSingle.setAttribute("aria-selected", "false");

        batchView.classList.add("active");
        singleView.classList.remove("active");
        
        if (!batchInputText.value) {
            btnLoadBenchmarkSuite.click();
        }
    });

    // =========================================================================
    // MODAL INTERACTION
    // =========================================================================

    function openModal() {
        architectureModal.classList.remove("hidden");
    }

    function closeModal() {
        architectureModal.classList.add("hidden");
    }

    btnOpenArchitecture.addEventListener("click", openModal);
    btnCloseModal.addEventListener("click", closeModal);
    btnDismissModal.addEventListener("click", closeModal);

    architectureModal.addEventListener("click", (e) => {
        if (e.target === architectureModal) closeModal();
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !architectureModal.classList.contains("hidden")) {
            closeModal();
        }
    });

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Run system checks and preset loaders
    checkSystemHealth();
    loadPresetSamples();
});
