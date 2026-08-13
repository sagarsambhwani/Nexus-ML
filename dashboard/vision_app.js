document.addEventListener("DOMContentLoaded", () => {
  const visionListContainer = document.getElementById("visionListContainer");
  const visionSearchInput = document.getElementById("visionSearchInput");
  const visionTitleHeader = document.getElementById("visionTitleHeader");
  const visionCategoryBadge = document.getElementById("visionCategoryBadge");
  const visionFileBadge = document.getElementById("visionFileBadge");
  const visionLatencyBadge = document.getElementById("visionLatencyBadge");
  const sectionNavTitle = document.getElementById("sectionNavTitle");
  
  const btnTabRoadmap = document.getElementById("btnTabRoadmap");
  const btnTabImplementations = document.getElementById("btnTabImplementations");
  
  const projectModeTabs = document.getElementById("projectModeTabs");
  const tabModePlayground = document.getElementById("tabModePlayground");
  const tabModeTheory = document.getElementById("tabModeTheory");
  const tabModeCode = document.getElementById("tabModeCode");

  const viewPlayground = document.getElementById("viewPlayground");
  const viewReader = document.getElementById("viewReader");
  const visionContentBody = document.getElementById("visionContentBody");
  const currentReaderTitle = document.getElementById("currentReaderTitle");
  
  const btnPrevModule = document.getElementById("btnPrevModule");
  const btnNextModule = document.getElementById("btnNextModule");
  const btnExecuteModule = document.getElementById("btnExecuteModule");

  const ctrlPreset = document.getElementById("ctrlPreset");
  const ctrlParamGroup = document.getElementById("ctrlParamGroup");
  const imgVisualInput = document.getElementById("imgVisualInput");
  const imgVisualOutput = document.getElementById("imgVisualOutput");
  const imgVisualIntermediate = document.getElementById("imgVisualIntermediate");
  const intermediateCard = document.getElementById("intermediateCard");
  const intermediateCardTitle = document.getElementById("intermediateCardTitle");
  const outputCardTitle = document.getElementById("outputCardTitle");
  const inputShapeBadge = document.getElementById("inputShapeBadge");
  const outputShapeBadge = document.getElementById("outputShapeBadge");
  const telemetrySummary = document.getElementById("telemetrySummary");
  const telemetryJson = document.getElementById("telemetryJson");

  let CURRENT_ITEMS = [];
  const CONTENT_CACHE = {};
  let activeIndex = 0;
  let activeNav = "implementations"; // "implementations" or "roadmap"
  let activeViewMode = "playground"; // "playground", "theory", "code"

  const BASE_URL = window.location.origin;

  // Tab mode switching (Playground vs Theory vs Code)
  function setViewMode(mode) {
    activeViewMode = mode;
    [tabModePlayground, tabModeTheory, tabModeCode].forEach(t => t && t.classList.remove("active"));

    if (mode === "playground") {
      if (tabModePlayground) tabModePlayground.classList.add("active");
      if (viewPlayground) viewPlayground.style.display = "flex";
      if (viewReader) viewReader.style.display = "none";
    } else if (mode === "theory") {
      if (tabModeTheory) tabModeTheory.classList.add("active");
      if (viewPlayground) viewPlayground.style.display = "none";
      if (viewReader) viewReader.style.display = "flex";
      loadTheoryForActiveItem();
    } else if (mode === "code") {
      if (tabModeCode) tabModeCode.classList.add("active");
      if (viewPlayground) viewPlayground.style.display = "none";
      if (viewReader) viewReader.style.display = "flex";
      loadCodeForActiveItem();
    }
  }

  if (tabModePlayground) tabModePlayground.addEventListener("click", () => setViewMode("playground"));
  if (tabModeTheory) tabModeTheory.addEventListener("click", () => setViewMode("theory"));
  if (tabModeCode) tabModeCode.addEventListener("click", () => setViewMode("code"));

  // Switch between 17 Projects and 14 Roadmap Phases
  function switchNav(nav) {
    activeNav = nav;
    if (nav === "implementations") {
      btnTabImplementations.style.background = "#eef2ff";
      btnTabImplementations.style.color = "var(--accent-primary)";
      btnTabRoadmap.style.background = "transparent";
      btnTabRoadmap.style.color = "var(--text-muted)";
      if (sectionNavTitle) sectionNavTitle.textContent = "17 HANDS-ON DELIVERABLES";
      if (projectModeTabs) projectModeTabs.style.display = "flex";
      loadImplementationsList();
    } else {
      btnTabRoadmap.style.background = "#eef2ff";
      btnTabRoadmap.style.color = "var(--accent-primary)";
      btnTabImplementations.style.background = "transparent";
      btnTabImplementations.style.color = "var(--text-muted)";
      if (sectionNavTitle) sectionNavTitle.textContent = "14-PHASE DEEP ROADMAP";
      if (projectModeTabs) projectModeTabs.style.display = "none";
      setViewMode("theory");
      loadRoadmapList();
    }
  }

  if (btnTabImplementations) btnTabImplementations.addEventListener("click", () => switchNav("implementations"));
  if (btnTabRoadmap) btnTabRoadmap.addEventListener("click", () => switchNav("roadmap"));

  async function loadImplementationsList() {
    if (visionListContainer) {
      visionListContainer.innerHTML = `<div class="loading-spinner">Loading 17 Projects...</div>`;
    }
    try {
      const res = await fetch(`${BASE_URL}/api/v1/vision/implementations`);
      if (!res.ok) throw new Error("Failed to load implementations.");
      const data = await res.json();
      CURRENT_ITEMS = data.implementations;
      activeIndex = 0;
      renderItemList(CURRENT_ITEMS);
      if (CURRENT_ITEMS.length > 0) {
        selectItem(0);
      }
    } catch (e) {
      if (visionListContainer) {
        visionListContainer.innerHTML = `<p class="error" style="color:var(--accent-rose); padding:12px;">Error: ${e.message}</p>`;
      }
    }
  }

  async function loadRoadmapList() {
    if (visionListContainer) {
      visionListContainer.innerHTML = `<div class="loading-spinner">Loading 14 Roadmap Phases...</div>`;
    }
    try {
      const res = await fetch(`${BASE_URL}/api/v1/vision/roadmap`);
      if (!res.ok) throw new Error("Failed to load roadmap.");
      const data = await res.json();
      CURRENT_ITEMS = data.phases;
      activeIndex = 0;
      renderItemList(CURRENT_ITEMS);
      if (CURRENT_ITEMS.length > 0) {
        selectItem(0);
      }
    } catch (e) {
      if (visionListContainer) {
        visionListContainer.innerHTML = `<p class="error" style="color:var(--accent-rose); padding:12px;">Error: ${e.message}</p>`;
      }
    }
  }

  function renderItemList(items) {
    if (!visionListContainer) return;
    visionListContainer.innerHTML = "";

    const activeKey = CURRENT_ITEMS[activeIndex]?.key;
    items.forEach((itemObj) => {
      const isActive = itemObj.key === activeKey;
      const btn = document.createElement("button");
      btn.className = `nav-item ${isActive ? "active" : ""}`;
      btn.style.width = "100%";
      btn.style.textAlign = "left";
      btn.style.marginBottom = "4px";
      btn.style.padding = "8px 12px";
      btn.style.fontSize = "13px";
      btn.style.borderRadius = "8px";
      btn.style.border = isActive ? "1px solid #c7d2fe" : "1px solid transparent";
      btn.style.cursor = "pointer";
      btn.style.background = isActive ? "#eef2ff" : "transparent";
      btn.style.color = isActive ? "#4338ca" : "var(--text-muted)";
      
      const icon = activeNav === "implementations" ? "💻" : "📘";
      btn.innerHTML = `<span class="icon">${icon}</span> ${itemObj.title}`;
      btn.title = itemObj.title;

      btn.addEventListener("click", () => {
        const idx = CURRENT_ITEMS.findIndex(i => i.key === itemObj.key);
        if (idx !== -1) selectItem(idx);
      });

      visionListContainer.appendChild(btn);
    });
  }

  if (visionSearchInput) {
    visionSearchInput.addEventListener("input", (e) => {
      const query = e.target.value.toLowerCase().trim();
      const filtered = CURRENT_ITEMS.filter(i => 
        i.title.toLowerCase().includes(query) || 
        i.key.toLowerCase().includes(query)
      );
      renderItemList(filtered);
    });
  }

  function selectItem(index) {
    if (index < 0 || index >= CURRENT_ITEMS.length) return;
    activeIndex = index;
    const item = CURRENT_ITEMS[index];

    renderItemList(CURRENT_ITEMS);

    if (visionTitleHeader) visionTitleHeader.textContent = item.title;
    if (visionCategoryBadge) visionCategoryBadge.textContent = item.category;
    if (visionFileBadge) visionFileBadge.textContent = `nexus_vision / ${item.filename || item.key}`;

    if (activeNav === "implementations") {
      updateDynamicControls(item.key);
      if (activeViewMode === "playground") {
        executeActiveModule();
      } else if (activeViewMode === "theory") {
        loadTheoryForActiveItem();
      } else if (activeViewMode === "code") {
        loadCodeForActiveItem();
      }
    } else {
      setViewMode("theory");
      loadRoadmapContent(item.key);
    }
  }

  function updateDynamicControls(key) {
    if (!ctrlParamGroup) return;
    ctrlParamGroup.innerHTML = "";

    const k = key.toLowerCase();
    if (k.includes("convolution") || k.includes("01")) {
      ctrlParamGroup.innerHTML = `
        <div>
          <label style="font-size: 11px; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 4px;">FILTER KERNEL</label>
          <select id="paramKernel" style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); background: #ffffff; font-family: inherit; font-size: 13px;">
            <option value="sobel_v">Sobel Vertical (d/dx)</option>
            <option value="sobel_h">Sobel Horizontal (d/dy)</option>
            <option value="gaussian_3x3">Gaussian 3x3 Smoothing</option>
            <option value="sharpen">Sharpening Kernel</option>
            <option value="laplacian">Laplacian Edge</option>
          </select>
        </div>
      `;
    } else if (k.includes("canny") || k.includes("02")) {
      ctrlParamGroup.innerHTML = `
        <div>
          <label style="font-size: 11px; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 4px;">LOW THRESH: <span id="valLow">30</span></label>
          <input type="range" id="paramLow" min="5" max="100" value="30" style="width: 90px;">
        </div>
        <div>
          <label style="font-size: 11px; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 4px;">HIGH THRESH: <span id="valHigh">80</span></label>
          <input type="range" id="paramHigh" min="40" max="200" value="80" style="width: 90px;">
        </div>
      `;
      const pLow = document.getElementById("paramLow");
      const pHigh = document.getElementById("paramHigh");
      if (pLow) pLow.addEventListener("input", (e) => document.getElementById("valLow").textContent = e.target.value);
      if (pHigh) pHigh.addEventListener("input", (e) => document.getElementById("valHigh").textContent = e.target.value);
    } else if (k.includes("iou") || k.includes("nms") || k.includes("07")) {
      ctrlParamGroup.innerHTML = `
        <div>
          <label style="font-size: 11px; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 4px;">IoU OVERLAP THRESH: <span id="valIou">0.5</span></label>
          <input type="range" id="paramIou" min="0.1" max="0.9" step="0.05" value="0.5" style="width: 120px;">
        </div>
      `;
      const pIou = document.getElementById("paramIou");
      if (pIou) pIou.addEventListener("input", (e) => document.getElementById("valIou").textContent = e.target.value);
    } else if (k.includes("backprop") || k.includes("cnn") || k.includes("05")) {
      ctrlParamGroup.innerHTML = `
        <div>
          <label style="font-size: 11px; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 4px;">EPOCHS</label>
          <select id="paramEpochs" style="padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border-color); background: #ffffff; font-family: inherit; font-size: 13px;">
            <option value="10">10 Epochs</option>
            <option value="20" selected>20 Epochs</option>
            <option value="50">50 Epochs</option>
          </select>
        </div>
      `;
    }
  }

  async function executeActiveModule() {
    if (CURRENT_ITEMS.length === 0) return;
    const item = CURRENT_ITEMS[activeIndex];
    
    if (btnExecuteModule) {
      btnExecuteModule.disabled = true;
      btnExecuteModule.innerHTML = `⏳ Running...`;
    }

    const preset = ctrlPreset ? ctrlPreset.value : "shapes";
    const payload = { preset };

    const pKernel = document.getElementById("paramKernel");
    if (pKernel) payload.kernel_name = pKernel.value;

    const pLow = document.getElementById("paramLow");
    const pHigh = document.getElementById("paramHigh");
    if (pLow && pHigh) {
      payload.low_threshold = parseFloat(pLow.value);
      payload.high_threshold = parseFloat(pHigh.value);
    }

    const pIou = document.getElementById("paramIou");
    if (pIou) payload.iou_threshold = parseFloat(pIou.value);

    const pEpochs = document.getElementById("paramEpochs");
    if (pEpochs) payload.epochs = parseInt(pEpochs.value);

    try {
      const res = await fetch(`${BASE_URL}/api/v1/vision/execute/${item.key}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Execution failed.");
      }
      const data = await res.json();
      renderExecutionResults(data);
    } catch (e) {
      if (telemetrySummary) {
        telemetrySummary.innerHTML = `<span style="color:var(--accent-rose); font-weight:600;">Execution Error:</span> ${e.message}`;
      }
    } finally {
      if (btnExecuteModule) {
        btnExecuteModule.disabled = false;
        btnExecuteModule.innerHTML = `🚀 Run Live Evaluation`;
      }
    }
  }

  function renderExecutionResults(data) {
    if (visionLatencyBadge) {
      visionLatencyBadge.style.display = "inline-flex";
      visionLatencyBadge.textContent = `⚡ ${data.latency_ms} ms`;
    }

    if (telemetrySummary && data.summary) {
      telemetrySummary.innerHTML = `<strong>Diagnostic Result:</strong> ${data.summary}`;
    }

    if (telemetryJson) {
      const cleanData = { ...data };
      delete cleanData.input_image_b64;
      delete cleanData.output_image_b64;
      delete cleanData.blurred_image_b64;
      delete cleanData.gradient_image_b64;
      delete cleanData.nms_image_b64;
      delete cleanData.visualization_b64;
      delete cleanData.side_by_side_b64;
      telemetryJson.textContent = JSON.stringify(cleanData, null, 2);
    }

    // Input image
    if (imgVisualInput) {
      if (data.input_image_b64) imgVisualInput.src = data.input_image_b64;
      else if (data.left_image_b64) imgVisualInput.src = data.left_image_b64;
      else if (data.sample_input_b64) imgVisualInput.src = data.sample_input_b64;
      else if (data.input_sample_b64) imgVisualInput.src = data.input_sample_b64;
      else if (data.visualization_b64) imgVisualInput.src = data.visualization_b64;
    }

    // Primary Output
    if (imgVisualOutput) {
      if (data.output_image_b64) imgVisualOutput.src = data.output_image_b64;
      else if (data.depth_map_b64) imgVisualOutput.src = data.depth_map_b64;
      else if (data.side_by_side_b64) imgVisualOutput.src = data.side_by_side_b64;
      else if (data.visualization_b64) imgVisualOutput.src = data.visualization_b64;
    }

    // Intermediate
    if (intermediateCard && imgVisualIntermediate) {
      if (data.nms_image_b64) {
        intermediateCard.style.display = "flex";
        intermediateCardTitle.textContent = "🔍 Stage 3: NMS Thinning";
        imgVisualIntermediate.src = data.nms_image_b64;
      } else if (data.gradient_image_b64) {
        intermediateCard.style.display = "flex";
        intermediateCardTitle.textContent = "🔍 Stage 2: Sobel Gradient";
        imgVisualIntermediate.src = data.gradient_image_b64;
      } else if (data.disparity_map_b64) {
        intermediateCard.style.display = "flex";
        intermediateCardTitle.textContent = "🔍 Stereo Disparity Map";
        imgVisualIntermediate.src = data.disparity_map_b64;
      } else {
        intermediateCard.style.display = "none";
      }
    }

    if (inputShapeBadge && data.input_shape) {
      inputShapeBadge.textContent = data.input_shape.join("x");
    }
    if (outputShapeBadge && data.output_shape) {
      outputShapeBadge.textContent = data.output_shape.join("x");
    }
  }

  if (btnExecuteModule) btnExecuteModule.addEventListener("click", executeActiveModule);
  if (ctrlPreset) ctrlPreset.addEventListener("change", executeActiveModule);

  async function loadTheoryForActiveItem() {
    if (CURRENT_ITEMS.length === 0) return;
    const item = CURRENT_ITEMS[activeIndex];
    if (currentReaderTitle) currentReaderTitle.textContent = `Theory: ${item.title}`;
    
    // Map project to corresponding roadmap phase
    let phaseKey = "roadmap/phase_01_classical_image_processing.md";
    const k = item.key.toLowerCase();
    if (k.includes("01") || k.includes("convolution")) phaseKey = "roadmap/phase_00_foundations_math.md";
    else if (k.includes("02") || k.includes("canny")) phaseKey = "roadmap/phase_01_classical_image_processing.md";
    else if (k.includes("03") || k.includes("04") || k.includes("hog") || k.includes("sift")) phaseKey = "roadmap/phase_02_feature_engineering.md";
    else if (k.includes("05") || k.includes("06") || k.includes("cnn") || k.includes("resnet")) phaseKey = "roadmap/phase_03_cnns_first_principles.md";
    else if (k.includes("07") || k.includes("08") || k.includes("iou") || k.includes("detector")) phaseKey = "roadmap/phase_04_object_detection.md";
    else if (k.includes("09") || k.includes("unet")) phaseKey = "roadmap/phase_05_segmentation.md";
    else if (k.includes("10") || k.includes("stereo")) phaseKey = "roadmap/phase_06_geometry_3d_vision.md";
    else if (k.includes("11") || k.includes("simclr")) phaseKey = "roadmap/phase_07_representation_learning.md";
    else if (k.includes("12") || k.includes("vit")) phaseKey = "roadmap/phase_08_vision_transformers.md";
    else if (k.includes("13") || k.includes("clip")) phaseKey = "roadmap/phase_09_vision_language_multimodal.md";
    else if (k.includes("14") || k.includes("diffusion")) phaseKey = "roadmap/phase_11_generative_vision.md";
    else if (k.includes("15") || k.includes("production")) phaseKey = "roadmap/phase_13_production_cv.md";

    loadRoadmapContent(phaseKey);
  }

  async function loadCodeForActiveItem() {
    if (CURRENT_ITEMS.length === 0) return;
    const item = CURRENT_ITEMS[activeIndex];
    if (currentReaderTitle) currentReaderTitle.textContent = `Python Source: ${item.filename || item.key}`;

    if (visionContentBody) {
      visionContentBody.innerHTML = `<div class="loading-spinner">Loading Python code...</div>`;
    }

    try {
      const res = await fetch(`${BASE_URL}/api/v1/vision/implementations/${item.key}`);
      if (!res.ok) throw new Error("Failed to load source code.");
      const data = await res.json();
      const mdCode = "```python\n" + data.code + "\n```";
      if (window.marked) {
        visionContentBody.innerHTML = marked.parse(mdCode);
      } else {
        visionContentBody.innerHTML = `<pre><code>${data.code}</code></pre>`;
      }
      visionContentBody.scrollTop = 0;
    } catch (e) {
      if (visionContentBody) {
        visionContentBody.innerHTML = `<p class="error" style="color:var(--accent-rose); padding:16px;">Error: ${e.message}</p>`;
      }
    }
  }

  async function loadRoadmapContent(phaseKey) {
    if (visionContentBody) {
      visionContentBody.innerHTML = `<div class="loading-spinner">Loading study guide...</div>`;
    }
    try {
      const res = await fetch(`${BASE_URL}/api/v1/vision/roadmap/${phaseKey}`);
      if (!res.ok) throw new Error("Failed to load roadmap phase.");
      const data = await res.json();
      if (window.marked) {
        visionContentBody.innerHTML = marked.parse(data.content);
      } else {
        visionContentBody.textContent = data.content;
      }
      visionContentBody.scrollTop = 0;
    } catch (e) {
      if (visionContentBody) {
        visionContentBody.innerHTML = `<p class="error" style="color:var(--accent-rose); padding:16px;">Error: ${e.message}</p>`;
      }
    }
  }

  if (btnPrevModule) {
    btnPrevModule.addEventListener("click", () => {
      if (activeIndex > 0) selectItem(activeIndex - 1);
    });
  }

  if (btnNextModule) {
    btnNextModule.addEventListener("click", () => {
      if (activeIndex < CURRENT_ITEMS.length - 1) selectItem(activeIndex + 1);
    });
  }

  // Initial load
  switchNav("implementations");
});
