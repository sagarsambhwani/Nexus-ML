document.addEventListener("DOMContentLoaded", () => {
  const courseListContainer = document.getElementById("courseListContainer");
  const courseSearchInput = document.getElementById("courseSearchInput");
  const courseTitleHeader = document.getElementById("courseTitleHeader");
  const currentChapterName = document.getElementById("currentChapterName");
  const courseFileBadge = document.getElementById("courseFileBadge");
  const chapterCountBadge = document.getElementById("chapterCountBadge");
  const courseMarkdownContent = document.getElementById("courseMarkdownContent");
  const btnPrevChapter = document.getElementById("btnPrevChapter");
  const btnNextChapter = document.getElementById("btnNextChapter");
  const btnVersionV1 = document.getElementById("btnVersionV1");
  const btnVersionV2 = document.getElementById("btnVersionV2");
  const btnVersionVision = document.getElementById("btnVersionVision");

  let ALL_COURSES = [];
  const COURSE_CACHE = {};
  let activeIndex = 0;
  let currentVersion = "v1";

  const COURSE_BASE_URL = window.location.origin;

  async function fetchCourseList(version = "v1") {
    currentVersion = version;
    try {
      const res = await fetch(`${COURSE_BASE_URL}/api/v1/courses?version=${version}`);
      if (!res.ok) return;
      const data = await res.json();
      ALL_COURSES = data.courses;
      activeIndex = 0;
      if (chapterCountBadge) {
        chapterCountBadge.textContent = `${data.total_courses} Total Modules`;
      }
      renderCourseList(ALL_COURSES);
      
      if (ALL_COURSES.length > 0) {
        loadChapterByIndex(0);
      }
    } catch (e) {
      if (courseListContainer) {
        courseListContainer.innerHTML = `<p class="error" style="color:var(--accent-rose); padding:12px;">Failed to load curriculum: ${e.message}</p>`;
      }
    }
  }

  function updateActiveTab(activeBtn) {
    [btnVersionV1, btnVersionV2, btnVersionVision].forEach(btn => {
      if (btn) {
        if (btn === activeBtn) {
          btn.style.background = "#eef2ff";
          btn.style.color = "var(--accent-primary)";
        } else {
          btn.style.background = "transparent";
          btn.style.color = "var(--text-muted)";
        }
      }
    });
  }

  if (btnVersionV1) {
    btnVersionV1.addEventListener("click", () => {
      updateActiveTab(btnVersionV1);
      fetchCourseList("v1");
    });
  }

  if (btnVersionV2) {
    btnVersionV2.addEventListener("click", () => {
      updateActiveTab(btnVersionV2);
      fetchCourseList("v2");
    });
  }

  if (btnVersionVision) {
    btnVersionVision.addEventListener("click", () => {
      updateActiveTab(btnVersionVision);
      fetchCourseList("vision");
    });
  }

  function renderCourseList(courses) {
    if (!courseListContainer) return;
    courseListContainer.innerHTML = "";

    const activeKey = ALL_COURSES[activeIndex]?.key;
    courses.forEach((c) => {
      const isActive = c.key === activeKey;
      const item = document.createElement("button");
      item.className = `nav-item ${isActive ? "active" : ""}`;
      item.style.width = "100%";
      item.style.textAlign = "left";
      item.style.marginBottom = "4px";
      item.style.padding = "8px 12px";
      item.style.fontSize = "13px";
      item.style.borderRadius = "8px";
      item.style.border = isActive ? "1px solid #c7d2fe" : "1px solid transparent";
      item.style.cursor = "pointer";
      item.style.background = isActive ? "#eef2ff" : "transparent";
      item.style.color = isActive ? "#4338ca" : "var(--text-muted)";
      
      let icon = '📖';
      if (currentVersion === 'vision') icon = '👁️';
      else if (currentVersion === 'v2') icon = '🚀';

      item.innerHTML = `<span class="icon">${icon}</span> ${c.title}`;
      item.title = c.title;

      item.addEventListener("click", () => {
        loadChapterByIndex(ALL_COURSES.findIndex(item => item.key === c.key));
      });

      courseListContainer.appendChild(item);
    });
  }

  if (courseSearchInput) {
    courseSearchInput.addEventListener("input", (e) => {
      const query = e.target.value.toLowerCase().trim();
      const filtered = ALL_COURSES.filter(c => c.title.toLowerCase().includes(query) || c.key.toLowerCase().includes(query));
      renderCourseList(filtered);
    });
  }

  async function loadChapterByIndex(index) {
    if (index < 0 || index >= ALL_COURSES.length) return;
    activeIndex = index;
    const chapter = ALL_COURSES[index];

    renderCourseList(ALL_COURSES);

    if (currentChapterName) currentChapterName.textContent = chapter.title;
    if (courseTitleHeader) courseTitleHeader.textContent = chapter.title;
    if (courseFileBadge) courseFileBadge.textContent = `${chapter.part} / ${chapter.filename}`;

    const cacheKey = `${currentVersion}_${chapter.key}`;
    if (COURSE_CACHE[cacheKey]) {
      renderMarkdown(COURSE_CACHE[cacheKey]);
      return;
    }

    if (courseMarkdownContent) {
      courseMarkdownContent.innerHTML = `<div class="loading-spinner">Loading chapter contents...</div>`;
    }

    try {
      const res = await fetch(`${COURSE_BASE_URL}/api/v1/courses/${chapter.key}?version=${currentVersion}`);
      if (!res.ok) throw new Error("Failed to load chapter content.");
      const data = await res.json();
      COURSE_CACHE[cacheKey] = data.content;
      renderMarkdown(data.content);
    } catch (e) {
      if (courseMarkdownContent) {
        courseMarkdownContent.innerHTML = `<p class="error" style="color:var(--accent-rose); padding:16px;">Error loading content: ${e.message}</p>`;
      }
    }
  }

  function renderMarkdown(md) {
    if (!courseMarkdownContent) return;
    if (window.marked) {
      courseMarkdownContent.innerHTML = marked.parse(md);
    } else {
      courseMarkdownContent.textContent = md;
    }
    courseMarkdownContent.scrollTop = 0;
  }

  if (btnPrevChapter) {
    btnPrevChapter.addEventListener("click", () => {
      if (activeIndex > 0) {
        loadChapterByIndex(activeIndex - 1);
      }
    });
  }

  if (btnNextChapter) {
    btnNextChapter.addEventListener("click", () => {
      if (activeIndex < ALL_COURSES.length - 1) {
        loadChapterByIndex(activeIndex + 1);
      }
    });
  }

  // Initial load checking URL parameters
  const urlParams = new URLSearchParams(window.location.search);
  const trackParam = urlParams.get("track") || urlParams.get("version");
  if (trackParam === "vision" || trackParam === "cv") {
    updateActiveTab(btnVersionVision);
    fetchCourseList("vision");
  } else if (trackParam === "v2" || trackParam === "2") {
    updateActiveTab(btnVersionV2);
    fetchCourseList("v2");
  } else {
    updateActiveTab(btnVersionV1);
    fetchCourseList("v1");
  }
});
