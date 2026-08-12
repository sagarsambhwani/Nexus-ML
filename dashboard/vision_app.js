document.addEventListener("DOMContentLoaded", () => {
  const visionListContainer = document.getElementById("visionListContainer");
  const visionSearchInput = document.getElementById("visionSearchInput");
  const visionTitleHeader = document.getElementById("visionTitleHeader");
  const currentModuleName = document.getElementById("currentModuleName");
  const visionCategoryBadge = document.getElementById("visionCategoryBadge");
  const visionFileBadge = document.getElementById("visionFileBadge");
  const visionContentBody = document.getElementById("visionContentBody");
  const btnPrevModule = document.getElementById("btnPrevModule");
  const btnNextModule = document.getElementById("btnNextModule");
  const btnTabRoadmap = document.getElementById("btnTabRoadmap");
  const btnTabImplementations = document.getElementById("btnTabImplementations");
  const sectionNavTitle = document.getElementById("sectionNavTitle");

  let CURRENT_ITEMS = [];
  const CONTENT_CACHE = {};
  let activeIndex = 0;
  let activeMode = "roadmap"; // "roadmap" or "implementations"

  const BASE_URL = window.location.origin;

  function switchTab(mode) {
    activeMode = mode;
    if (mode === "roadmap") {
      btnTabRoadmap.style.background = "#eef2ff";
      btnTabRoadmap.style.color = "var(--accent-primary)";
      btnTabImplementations.style.background = "transparent";
      btnTabImplementations.style.color = "var(--text-muted)";
      if (sectionNavTitle) sectionNavTitle.textContent = "14-PHASE DEEP ROADMAP";
      loadRoadmapList();
    } else {
      btnTabImplementations.style.background = "#eef2ff";
      btnTabImplementations.style.color = "var(--accent-primary)";
      btnTabRoadmap.style.background = "transparent";
      btnTabRoadmap.style.color = "var(--text-muted)";
      if (sectionNavTitle) sectionNavTitle.textContent = "17 CORE DELIVERABLES";
      loadImplementationsList();
    }
  }

  if (btnTabRoadmap) btnTabRoadmap.addEventListener("click", () => switchTab("roadmap"));
  if (btnTabImplementations) btnTabImplementations.addEventListener("click", () => switchTab("implementations"));

  async function loadRoadmapList() {
    if (visionListContainer) {
      visionListContainer.innerHTML = `<div class="loading-spinner">Loading 14 Roadmap Phases...</div>`;
    }
    try {
      const res = await fetch(`${BASE_URL}/api/v1/vision/roadmap`);
      if (!res.ok) throw new Error("Failed to load vision roadmap list.");
      const data = await res.json();
      CURRENT_ITEMS = data.phases;
      activeIndex = 0;
      renderItemList(CURRENT_ITEMS);
      if (CURRENT_ITEMS.length > 0) {
        loadItemByIndex(0);
      }
    } catch (e) {
      if (visionListContainer) {
        visionListContainer.innerHTML = `<p class="error" style="color:var(--accent-rose); padding:12px;">Error: ${e.message}</p>`;
      }
    }
  }

  async function loadImplementationsList() {
    if (visionListContainer) {
      visionListContainer.innerHTML = `<div class="loading-spinner">Loading 17 Core Implementations...</div>`;
    }
    try {
      const res = await fetch(`${BASE_URL}/api/v1/vision/implementations`);
      if (!res.ok) throw new Error("Failed to load vision implementations list.");
      const data = await res.json();
      CURRENT_ITEMS = data.implementations;
      activeIndex = 0;
      renderItemList(CURRENT_ITEMS);
      if (CURRENT_ITEMS.length > 0) {
        loadItemByIndex(0);
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
      
      const icon = activeMode === "roadmap" ? "📘" : "💻";
      btn.innerHTML = `<span class="icon">${icon}</span> ${itemObj.title}`;
      btn.title = itemObj.title;

      btn.addEventListener("click", () => {
        const idx = CURRENT_ITEMS.findIndex(i => i.key === itemObj.key);
        if (idx !== -1) loadItemByIndex(idx);
      });

      visionListContainer.appendChild(btn);
    });
  }

  if (visionSearchInput) {
    visionSearchInput.addEventListener("input", (e) => {
      const query = e.target.value.toLowerCase().trim();
      const filtered = CURRENT_ITEMS.filter(i => 
        i.title.toLowerCase().includes(query) || 
        i.key.toLowerCase().includes(query) ||
        (i.docstring && i.docstring.toLowerCase().includes(query))
      );
      renderItemList(filtered);
    });
  }

  async function loadItemByIndex(index) {
    if (index < 0 || index >= CURRENT_ITEMS.length) return;
    activeIndex = index;
    const item = CURRENT_ITEMS[index];

    renderItemList(CURRENT_ITEMS);

    if (currentModuleName) currentModuleName.textContent = item.title;
    if (visionTitleHeader) visionTitleHeader.textContent = item.title;
    if (visionCategoryBadge) visionCategoryBadge.textContent = item.category;
    if (visionFileBadge) visionFileBadge.textContent = `nexus_vision / ${item.filename || item.key}`;

    const cacheKey = `${activeMode}_${item.key}`;
    if (CONTENT_CACHE[cacheKey]) {
      displayContent(CONTENT_CACHE[cacheKey]);
      return;
    }

    if (visionContentBody) {
      visionContentBody.innerHTML = `<div class="loading-spinner">Loading ${activeMode === 'roadmap' ? 'study guide' : 'code deliverable'}...</div>`;
    }

    try {
      let endpoint = "";
      if (activeMode === "roadmap") {
        endpoint = `${BASE_URL}/api/v1/vision/roadmap/${item.key}`;
      } else {
        endpoint = `${BASE_URL}/api/v1/vision/implementations/${item.key}`;
      }

      const res = await fetch(endpoint);
      if (!res.ok) throw new Error("Failed to fetch content from server.");
      const data = await res.json();
      
      const payload = activeMode === "roadmap" ? data.content : data.code;
      CONTENT_CACHE[cacheKey] = payload;
      displayContent(payload);
    } catch (e) {
      if (visionContentBody) {
        visionContentBody.innerHTML = `<p class="error" style="color:var(--accent-rose); padding:16px;">Error loading content: ${e.message}</p>`;
      }
    }
  }

  function displayContent(rawText) {
    if (!visionContentBody) return;
    if (activeMode === "roadmap") {
      if (window.marked) {
        visionContentBody.innerHTML = marked.parse(rawText);
      } else {
        visionContentBody.textContent = rawText;
      }
    } else {
      // Format Python implementation code nicely
      const mdCode = "```python\n" + rawText + "\n```";
      if (window.marked) {
        visionContentBody.innerHTML = marked.parse(mdCode);
      } else {
        visionContentBody.innerHTML = `<pre><code>${rawText}</code></pre>`;
      }
    }
    visionContentBody.scrollTop = 0;
  }

  if (btnPrevModule) {
    btnPrevModule.addEventListener("click", () => {
      if (activeIndex > 0) loadItemByIndex(activeIndex - 1);
    });
  }

  if (btnNextModule) {
    btnNextModule.addEventListener("click", () => {
      if (activeIndex < CURRENT_ITEMS.length - 1) loadItemByIndex(activeIndex + 1);
    });
  }

  // Check URL params for deep linking (e.g. ?mode=implementations)
  const urlParams = new URLSearchParams(window.location.search);
  const modeParam = urlParams.get("mode");
  if (modeParam === "implementations" || modeParam === "code" || modeParam === "projects") {
    switchTab("implementations");
  } else {
    switchTab("roadmap");
  }
});
