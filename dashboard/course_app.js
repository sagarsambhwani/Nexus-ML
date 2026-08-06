document.addEventListener("DOMContentLoaded", () => {
  const courseListContainer = document.getElementById("courseListContainer");
  const courseSearchInput = document.getElementById("courseSearchInput");
  const courseTitleHeader = document.getElementById("courseTitleHeader");
  const currentChapterName = document.getElementById("currentChapterName");
  const courseFileBadge = document.getElementById("courseFileBadge");
  const courseMarkdownContent = document.getElementById("courseMarkdownContent");
  const btnPrevChapter = document.getElementById("btnPrevChapter");
  const btnNextChapter = document.getElementById("btnNextChapter");
  const btnVersionV1 = document.getElementById("btnVersionV1");
  const btnVersionV2 = document.getElementById("btnVersionV2");

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
      renderCourseList(ALL_COURSES);
      
      if (ALL_COURSES.length > 0) {
        loadChapterByIndex(0);
      }
    } catch (e) {
      if (courseListContainer) {
        courseListContainer.innerHTML = `<p class="error" style="color:#fca5a5; padding:12px;">Failed to load course list: ${e.message}</p>`;
      }
    }
  }

  if (btnVersionV1) {
    btnVersionV1.addEventListener("click", () => {
      btnVersionV1.style.background = "rgba(216, 180, 254, 0.2)";
      btnVersionV1.style.color = "#ffffff";
      if (btnVersionV2) {
        btnVersionV2.style.background = "transparent";
        btnVersionV2.style.color = "var(--text-muted)";
      }
      fetchCourseList("v1");
    });
  }

  if (btnVersionV2) {
    btnVersionV2.addEventListener("click", () => {
      btnVersionV2.style.background = "rgba(216, 180, 254, 0.2)";
      btnVersionV2.style.color = "#ffffff";
      if (btnVersionV1) {
        btnVersionV1.style.background = "transparent";
        btnVersionV1.style.color = "var(--text-muted)";
      }
      fetchCourseList("v2");
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
      item.style.border = "1px solid transparent";
      item.style.cursor = "pointer";
      item.style.background = isActive ? "rgba(216, 180, 254, 0.2)" : "transparent";
      item.style.color = isActive ? "#ffffff" : "var(--text-muted)";
      item.innerHTML = `<span class="icon">${currentVersion === 'v2' ? '🚀' : '📖'}</span> ${c.title}`;
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

  function loadChapterByIndex(index) {
    if (index < 0 || index >= ALL_COURSES.length) return;
    activeIndex = index;
    const course = ALL_COURSES[index];
    renderCourseList(ALL_COURSES);
    fetchCourseContent(course.key, course.title);
  }

  async function fetchCourseContent(courseKey, title) {
    if (courseTitleHeader) courseTitleHeader.textContent = title || "Course Content";
    if (currentChapterName) currentChapterName.textContent = title || "Chapter Content";
    const folderName = currentVersion === "v2" ? "course_v2" : "course";
    if (courseFileBadge) courseFileBadge.textContent = `Course Microservice (:8001) / ${folderName}/${courseKey}`;

    const cacheKey = `${currentVersion}:${courseKey}`;
    if (COURSE_CACHE[cacheKey]) {
      renderMarkdown(COURSE_CACHE[cacheKey]);
      return;
    }

    if (courseMarkdownContent) {
      courseMarkdownContent.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">⏳</span>
          <p>Loading course module content from Course Microservice...</p>
        </div>
      `;
    }

    try {
      const res = await fetch(`${COURSE_BASE_URL}/api/v1/courses/${courseKey}?version=${currentVersion}`);
      if (!res.ok) {
        if (courseMarkdownContent) courseMarkdownContent.innerHTML = `<p class="error">Failed to load chapter content.</p>`;
        return;
      }
      const data = await res.json();
      COURSE_CACHE[cacheKey] = data.content;
      renderMarkdown(data.content);
    } catch (e) {
      if (courseMarkdownContent) courseMarkdownContent.innerHTML = `<p class="error">Error loading course: ${e.message}</p>`;
    }
  }

  function renderMarkdown(markdownText) {
    if (!courseMarkdownContent) return;
    try {
      if (typeof window.marked === 'function') {
        courseMarkdownContent.innerHTML = window.marked(markdownText);
        return;
      } else if (window.marked && typeof window.marked.parse === 'function') {
        courseMarkdownContent.innerHTML = window.marked.parse(markdownText);
        return;
      }
    } catch (e) {
      console.warn("Marked parser fallback:", e);
    }
    courseMarkdownContent.innerText = markdownText;
  }

  if (btnPrevChapter) {
    btnPrevChapter.addEventListener("click", () => {
      if (activeIndex > 0) loadChapterByIndex(activeIndex - 1);
    });
  }

  if (btnNextChapter) {
    btnNextChapter.addEventListener("click", () => {
      if (activeIndex < ALL_COURSES.length - 1) loadChapterByIndex(activeIndex + 1);
    });
  }

  // Initialize with V1
  fetchCourseList("v1");
});
