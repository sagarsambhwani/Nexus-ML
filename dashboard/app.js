document.addEventListener("DOMContentLoaded", () => {
  // Global Pipeline Configuration & Form Schemas
  const PIPELINE_CONFIGS = {
    fraud_detection: {
      title: "Fraud Detection",
      subtitle: "Credit card transaction anomaly scoring & classification",
      taskType: "Classification",
      endpoint: "/api/v1/predict/fraud-detection",
      fields: [
        { name: "amount", label: "Amount ($)", type: "number", step: "0.01", default: 850.00 },
        { name: "time_hour", label: "Transaction Hour (0-23)", type: "number", default: 3 },
        { name: "velocity_1h", label: "1hr Velocity (Tx Count)", type: "number", default: 6 },
        { name: "location_risk", label: "Location Risk (0-1)", type: "number", step: "0.01", default: 0.85 },
        { name: "v1", label: "PCA Feature V1", type: "number", step: "0.1", default: 2.1 },
        { name: "v2", label: "PCA Feature V2", type: "number", step: "0.1", default: -1.2 },
        { name: "v3", label: "PCA Feature V3", type: "number", step: "0.1", default: 0.5 },
        { name: "v4", label: "PCA Feature V4", type: "number", step: "0.1", default: 1.8 }
      ],
      presets: {
        highRisk: { amount: 1450.00, time_hour: 2, velocity_1h: 8, location_risk: 0.92, v1: 2.8, v2: -2.1, v3: 1.1, v4: 2.4 },
        normal: { amount: 45.20, time_hour: 14, velocity_1h: 1, location_risk: 0.10, v1: 0.1, v2: 0.2, v3: -0.1, v4: 0.0 }
      }
    },
    credit_risk: {
      title: "Credit Risk Prediction",
      subtitle: "Loan default risk underwriting & credit scorecard evaluation",
      taskType: "Classification",
      endpoint: "/api/v1/predict/credit-risk",
      fields: [
        { name: "credit_score", label: "Credit Score (300-850)", type: "number", default: 620 },
        { name: "annual_income", label: "Annual Income ($)", type: "number", default: 48000 },
        { name: "dti_ratio", label: "Debt-to-Income Ratio (0-1)", type: "number", step: "0.01", default: 0.45 },
        { name: "loan_amount", label: "Loan Amount Requested ($)", type: "number", default: 30000 },
        { name: "delinquencies_2yr", label: "2yr Delinquency Count", type: "number", default: 2 },
        { name: "employment_years", label: "Employment Length (Yrs)", type: "number", default: 2 }
      ],
      presets: {
        highRisk: { credit_score: 540, annual_income: 32000, dti_ratio: 0.55, loan_amount: 35000, delinquencies_2yr: 3, employment_years: 1 },
        normal: { credit_score: 760, annual_income: 95000, dti_ratio: 0.18, loan_amount: 20000, delinquencies_2yr: 0, employment_years: 8 }
      }
    },
    customer_churn: {
      title: "Customer Churn Prediction",
      subtitle: "Telecom & SaaS subscriber retention & churn driver analysis",
      taskType: "Classification",
      endpoint: "/api/v1/predict/customer-churn",
      fields: [
        { name: "tenure", label: "Tenure (Months)", type: "number", default: 4 },
        { name: "monthly_charges", label: "Monthly Charges ($)", type: "number", step: "0.1", default: 95.00 },
        { name: "total_charges", label: "Total Lifetime Charges ($)", type: "number", default: 380.00 },
        { name: "contract_type", label: "Contract Type (0=Month-to-month, 1=1yr, 2=2yr)", type: "number", default: 0 },
        { name: "support_tickets", label: "Support Ticket Count", type: "number", default: 5 },
        { name: "paperless_billing", label: "Paperless Billing (0/1)", type: "number", default: 1 }
      ],
      presets: {
        highRisk: { tenure: 3, monthly_charges: 105.00, total_charges: 315.00, contract_type: 0, support_tickets: 6, paperless_billing: 1 },
        normal: { tenure: 42, monthly_charges: 55.00, total_charges: 2310.00, contract_type: 2, support_tickets: 0, paperless_billing: 0 }
      }
    },
    house_prices: {
      title: "House Price Prediction",
      subtitle: "Real estate property automated valuation & price bounds",
      taskType: "Regression",
      endpoint: "/api/v1/predict/house-prices",
      fields: [
        { name: "sqft", label: "Square Footage", type: "number", default: 2400 },
        { name: "bedrooms", label: "Bedrooms", type: "number", default: 4 },
        { name: "bathrooms", label: "Bathrooms", type: "number", step: "0.5", default: 3.0 },
        { name: "location_score", label: "Location Score (1-10)", type: "number", step: "0.1", default: 8.5 },
        { name: "house_age", label: "House Age (Years)", type: "number", default: 8 },
        { name: "garage_cars", label: "Garage Car Capacity", type: "number", default: 2 },
        { name: "dist_city_km", label: "Distance to City Center (km)", type: "number", step: "0.1", default: 5.0 }
      ],
      presets: {
        highRisk: { sqft: 3800, bedrooms: 5, bathrooms: 4.0, location_score: 9.5, house_age: 2, garage_cars: 3, dist_city_km: 2.5 },
        normal: { sqft: 1400, bedrooms: 2, bathrooms: 1.5, location_score: 4.5, house_age: 35, garage_cars: 1, dist_city_km: 18.0 }
      }
    },
    recommendation: {
      title: "Recommendation System",
      subtitle: "Hybrid collaborative-content product recommendation engine",
      taskType: "Recommendation",
      endpoint: "/api/v1/predict/recommendation",
      fields: [
        { name: "user_id", label: "User ID", type: "text", default: "USER_012" },
        { name: "category", label: "Category Filter", type: "select", options: ["All", "Electronics", "Books", "Fashion", "Sports"], default: "All" },
        { name: "top_n", label: "Top-K Recommendations", type: "number", default: 5 }
      ],
      presets: {
        highRisk: { user_id: "USER_012", category: "Electronics", top_n: 5 },
        normal: { user_id: "USER_045", category: "All", top_n: 5 }
      }
    },
    demand_forecasting: {
      title: "Demand Forecasting",
      subtitle: "Retail time-series sales demand multi-day forecast horizon",
      taskType: "Time Series",
      endpoint: "/api/v1/predict/demand-forecasting",
      fields: [
        { name: "store_id", label: "Store Identifier", type: "text", default: "STORE_101" },
        { name: "horizon_days", label: "Forecast Horizon (Days)", type: "number", default: 7 },
        { name: "is_promo", label: "Promotion Active (0/1)", type: "number", default: 1 }
      ],
      presets: {
        highRisk: { store_id: "STORE_PROMO_SPECIAL", horizon_days: 10, is_promo: 1 },
        normal: { store_id: "STORE_REGULAR", horizon_days: 7, is_promo: 0 }
      }
    },
    predictive_maintenance: {
      title: "Predictive Maintenance",
      subtitle: "Industrial IoT telemetry failure prediction & RUL estimation",
      taskType: "Telemetry",
      endpoint: "/api/v1/predict/predictive-maintenance",
      fields: [
        { name: "vibration_hz", label: "Vibration Frequency (Hz)", type: "number", step: "0.1", default: 78.4 },
        { name: "temperature_c", label: "Temperature (°C)", type: "number", step: "0.1", default: 98.2 },
        { name: "pressure_psi", label: "Pressure (PSI)", type: "number", step: "0.1", default: 85.0 },
        { name: "rpm", label: "RPM", type: "number", default: 3100 },
        { name: "sensor_noise_std", label: "Sensor Noise Std Dev", type: "number", step: "0.1", default: 4.1 },
        { name: "operating_hours", label: "Total Operating Hours", type: "number", default: 7800 }
      ],
      presets: {
        highRisk: { vibration_hz: 86.5, temperature_c: 106.0, pressure_psi: 94.0, rpm: 3400, sensor_noise_std: 4.8, operating_hours: 9200 },
        normal: { vibration_hz: 22.0, temperature_c: 55.0, pressure_psi: 42.0, rpm: 1500, sensor_noise_std: 0.5, operating_hours: 1200 }
      }
    },
    medical_diagnosis: {
      title: "Medical Diagnosis Support",
      subtitle: "Clinical biomarker evaluation & health risk stratification",
      taskType: "Healthcare ML",
      endpoint: "/api/v1/predict/medical-diagnosis",
      fields: [
        { name: "age", label: "Patient Age", type: "number", default: 58 },
        { name: "glucose", label: "Fasting Glucose (mg/dL)", type: "number", step: "0.1", default: 155.0 },
        { name: "blood_pressure", label: "Systolic BP (mmHg)", type: "number", step: "0.1", default: 142.0 },
        { name: "bmi", label: "Body Mass Index (BMI)", type: "number", step: "0.1", default: 34.2 },
        { name: "hba1c", label: "HbA1c Level (%)", type: "number", step: "0.1", default: 7.2 },
        { name: "family_history", label: "Family History (0/1)", type: "number", default: 1 },
        { name: "smoker", label: "Smoker (0/1)", type: "number", default: 1 }
      ],
      presets: {
        highRisk: { age: 62, glucose: 180.0, blood_pressure: 150.0, bmi: 36.5, hba1c: 8.4, family_history: 1, smoker: 1 },
        normal: { age: 28, glucose: 88.0, blood_pressure: 112.0, bmi: 21.5, hba1c: 5.1, family_history: 0, smoker: 0 }
      }
    },
    sentiment_analysis: {
      title: "Sentiment Analysis",
      subtitle: "NLP customer review sentiment & composite emotion scoring",
      taskType: "NLP",
      endpoint: "/api/v1/predict/sentiment-analysis",
      fields: [
        { name: "text", label: "Input Text Review", type: "textarea", fullWidth: true, default: "This product is fantastic! Exceeded all my expectations and arrived super fast." }
      ],
      presets: {
        highRisk: { text: "Terrible purchase! Defective item, complete waste of money and horrible customer service." },
        normal: { text: "Outstanding quality! Super sleek build, easy to set up and worth every single penny. Highly recommend!" }
      }
    },
    document_classification: {
      title: "Document Classification",
      subtitle: "NLP resume skill parsing & corporate document categorizer",
      taskType: "NLP",
      endpoint: "/api/v1/predict/document-classification",
      fields: [
        { name: "text", label: "Document Content / Text Snippet", type: "textarea", fullWidth: true, default: "Senior Software Engineer with 7 years experience in Python, FastAPI, Docker, Kubernetes, microservices, and distributed cloud systems." }
      ],
      presets: {
        highRisk: { text: "Master Service Agreement: The receiving party agrees to indemnify and hold harmless the disclosing party against liability, breach of contract, or intellectual property infringement under state jurisdiction." },
        normal: { text: "Senior Software Engineer resume with expertise in Python, PyTorch, Machine Learning Pipelines, FastAPI, Docker, CI/CD, and Cloud Architecture." }
      }
    },
    defect_detection: {
      title: "Defect Detection",
      subtitle: "Computer vision image surface inspection & fracture grading",
      taskType: "Computer Vision",
      endpoint: "/api/v1/predict/defect-detection",
      fields: [
        { name: "mean_intensity", label: "Mean Pixel Intensity (0-255)", type: "number", step: "0.1", default: 105.0 },
        { name: "std_intensity", label: "Pixel Intensity Std Dev", type: "number", step: "0.1", default: 38.5 },
        { name: "edge_pixel_density", label: "Edge Density Ratio", type: "number", step: "0.01", default: 0.25 },
        { name: "contrast_ratio", label: "Contrast Ratio", type: "number", step: "0.1", default: 5.8 },
        { name: "surface_roughness", label: "Surface Roughness Metric", type: "number", step: "0.1", default: 7.2 },
        { name: "anomaly_patch_max", label: "Max Patch Anomaly Score", type: "number", step: "0.01", default: 0.88 }
      ],
      presets: {
        highRisk: { mean_intensity: 85.0, std_intensity: 42.0, edge_pixel_density: 0.32, contrast_ratio: 7.1, surface_roughness: 8.5, anomaly_patch_max: 0.94 },
        normal: { mean_intensity: 140.0, std_intensity: 12.0, edge_pixel_density: 0.04, contrast_ratio: 1.8, surface_roughness: 1.2, anomaly_patch_max: 0.15 }
      }
    },
    customer_segmentation: {
      title: "Customer Segmentation",
      subtitle: "Unsupervised K-Means clustering, PCA projection & persona analysis",
      taskType: "Clustering",
      endpoint: "/api/v1/predict/customer-segmentation",
      fields: [
        { name: "annual_income_k", label: "Annual Income ($k)", type: "number", default: 115.0 },
        { name: "spending_score", label: "Spending Score (1-100)", type: "number", default: 90.0 },
        { name: "frequency_purchases", label: "Purchases / Year", type: "number", default: 28.0 },
        { name: "recency_days", label: "Recency (Days since order)", type: "number", default: 8.0 }
      ],
      presets: {
        highRisk: { annual_income_k: 120.0, spending_score: 92.0, frequency_purchases: 32.0, recency_days: 5.0 },
        normal: { annual_income_k: 35.0, spending_score: 18.0, frequency_purchases: 3.0, recency_days: 110.0 }
      }
    }
  };

  let activePipelineKey = "fraud_detection";
  const README_CACHE = {};

  // Elements
  const navMenu = document.getElementById("navMenu");
  const titleEl = document.getElementById("currentPipelineTitle");
  const subtitleEl = document.getElementById("currentPipelineSubtitle");
  const taskTypeBadge = document.getElementById("taskTypeBadge");
  const pipelineForm = document.getElementById("pipelineForm");
  const btnRun = document.getElementById("btnRunInference");
  const btnPresetHigh = document.getElementById("btnLoadPreset");
  const btnPresetNormal = document.getElementById("btnLoadNormalPreset");
  const resultWidget = document.getElementById("resultWidget");
  const jsonOutput = document.getElementById("jsonOutput");
  const execTime = document.getElementById("execTime");
  const metricsContainer = document.getElementById("metricsPillsContainer");
  const btnCopyJson = document.getElementById("btnCopyJson");

  // Course Elements
  const tabCourse = document.getElementById("tabCourse");
  const courseViewContainer = document.getElementById("courseViewContainer");
  const courseListContainer = document.getElementById("courseListContainer");
  const courseSearchInput = document.getElementById("courseSearchInput");
  const courseTitleHeader = document.getElementById("courseTitleHeader");
  const courseFileBadge = document.getElementById("courseFileBadge");
  const courseMarkdownContent = document.getElementById("courseMarkdownContent");
  const courseTotalPill = document.getElementById("courseTotalPill");

  let ALL_COURSES = [];
  const COURSE_CACHE = {};
  let activeCourseKey = "README.md";

  const COURSE_MICROSERVICE_BASE = window.location.origin;

  async function fetchCourseList() {
    try {
      const res = await fetch("/api/v1/courses");
      if (!res.ok) return;
      const data = await res.json();
      ALL_COURSES = data.courses;
      if (courseTotalPill) courseTotalPill.textContent = `${ALL_COURSES.length - 1} Chapters`;
      renderCourseList(ALL_COURSES);
      
      // Load curriculum README or active chapter
      fetchCourseContent(activeCourseKey, "Course Curriculum & Overview");
    } catch (e) {
      if (courseListContainer) {
        courseListContainer.innerHTML = `<p class="error">Failed to load courses: ${e.message}</p>`;
      }
    }
  }

  function renderCourseList(courses) {
    if (!courseListContainer) return;
    courseListContainer.innerHTML = "";

    courses.forEach(c => {
      const item = document.createElement("button");
      item.className = `course-item ${c.key === activeCourseKey ? "active" : ""}`;
      item.textContent = c.title;
      item.title = c.title;
      item.dataset.key = c.key;

      item.addEventListener("click", () => {
        document.querySelectorAll(".course-item").forEach(el => el.classList.remove("active"));
        item.classList.add("active");
        activeCourseKey = c.key;
        fetchCourseContent(c.key, c.title);
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

  async function fetchCourseContent(courseKey, title) {
    if (courseTitleHeader) courseTitleHeader.textContent = title || "Course Content";
    if (courseFileBadge) courseFileBadge.textContent = `Course Service / course/${courseKey}`;

    if (COURSE_CACHE[courseKey]) {
      renderCourseMarkdown(COURSE_CACHE[courseKey]);
      return;
    }

    if (courseMarkdownContent) {
      courseMarkdownContent.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">⏳</span>
          <p>Loading course module from Course Microservice...</p>
        </div>
      `;
    }

    try {
      let res;
      try {
        res = await fetch(`${COURSE_MICROSERVICE_BASE}/api/v1/courses/${courseKey}`);
      } catch (err) {
        res = await fetch(`/api/v1/courses/${courseKey}`);
      }
      if (!res.ok) {
        if (courseMarkdownContent) courseMarkdownContent.innerHTML = `<p class="error">Failed to load chapter content.</p>`;
        return;
      }
      const data = await res.json();
      COURSE_CACHE[courseKey] = data.content;
      renderCourseMarkdown(data.content);
    } catch (e) {
      if (courseMarkdownContent) courseMarkdownContent.innerHTML = `<p class="error">Error loading course: ${e.message}</p>`;
    }
  }

  const readmeViewContainer = document.getElementById("readmeViewContainer");
  const readmeContent = document.getElementById("readmeContent");
  const readmeTitle = document.getElementById("readmeTitle");
  const readmeFileBadge = document.getElementById("readmeFileBadge");

  async function fetchPipelineReadme(pipelineKey) {
    const config = PIPELINE_CONFIGS[pipelineKey];
    if (readmeTitle && config) readmeTitle.textContent = `${config.title} Documentation`;
    if (readmeFileBadge) readmeFileBadge.textContent = `src/${pipelineKey}/README.md`;

    if (README_CACHE[pipelineKey]) {
      renderReadmeMarkdown(README_CACHE[pipelineKey]);
      return;
    }

    if (readmeContent) {
      readmeContent.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">⏳</span>
          <p>Loading README documentation for ${config ? config.title : pipelineKey}...</p>
        </div>
      `;
    }

    try {
      const res = await fetch(`/api/v1/pipelines/${pipelineKey}/readme`);
      if (!res.ok) {
        if (readmeContent) readmeContent.innerHTML = `<p class="error">Failed to load documentation for ${pipelineKey}.</p>`;
        return;
      }
      const data = await res.json();
      README_CACHE[pipelineKey] = data.readme;
      renderReadmeMarkdown(data.readme);
    } catch (e) {
      if (readmeContent) readmeContent.innerHTML = `<p class="error">Error loading README: ${e.message}</p>`;
    }
  }

  function renderReadmeMarkdown(markdownText) {
    if (!readmeContent) return;
    try {
      if (typeof window.marked === 'function') {
        readmeContent.innerHTML = window.marked(markdownText);
        return;
      } else if (window.marked && typeof window.marked.parse === 'function') {
        readmeContent.innerHTML = window.marked.parse(markdownText);
        return;
      }
    } catch (e) {
      console.warn("Marked parser failed, using fallback:", e);
    }
    readmeContent.innerHTML = parseMarkdownFallback(markdownText);
  }

  function renderCourseMarkdown(markdownText) {
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
      console.warn("Marked parser failed, using fallback:", e);
    }
    courseMarkdownContent.innerHTML = parseMarkdownFallback(markdownText);
  }

  // View Mode Navigation
  function switchMainView(mode) {
    const headerActions = document.getElementById("headerActions");
    if (mode === "course") {
      tabWorkspace.classList.remove("active");
      tabReadme.classList.remove("active");
      if (tabCourse) tabCourse.classList.add("active");

      workspaceGrid.classList.add("hidden");
      readmeViewContainer.classList.add("hidden");
      if (courseViewContainer) courseViewContainer.classList.remove("hidden");
      if (headerActions) headerActions.classList.add("hidden");

      if (ALL_COURSES.length === 0) {
        fetchCourseList();
      }
    } else if (mode === "readme") {
      tabWorkspace.classList.remove("active");
      if (tabCourse) tabCourse.classList.remove("active");
      tabReadme.classList.add("active");

      workspaceGrid.classList.remove("hidden");
      readmeViewContainer.classList.remove("hidden");
      if (courseViewContainer) courseViewContainer.classList.add("hidden");
      if (headerActions) headerActions.classList.remove("hidden");

      if (readmeViewContainer) {
        readmeViewContainer.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      tabReadme.classList.remove("active");
      if (tabCourse) tabCourse.classList.remove("active");
      tabWorkspace.classList.add("active");

      workspaceGrid.classList.remove("hidden");
      readmeViewContainer.classList.remove("hidden");
      if (courseViewContainer) courseViewContainer.classList.add("hidden");
      if (headerActions) headerActions.classList.remove("hidden");

      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  if (tabWorkspace) tabWorkspace.addEventListener("click", () => switchMainView("workspace"));
  if (tabReadme) tabReadme.addEventListener("click", () => switchMainView("readme"));
  if (tabCourse) tabCourse.addEventListener("click", () => switchMainView("course"));

  // Render Form for Active Pipeline
  function renderForm(pipelineKey) {
    activePipelineKey = pipelineKey;
    const config = PIPELINE_CONFIGS[pipelineKey];

    titleEl.textContent = config.title;
    subtitleEl.textContent = config.subtitle;
    taskTypeBadge.textContent = config.taskType;

    // Load README for this pipeline
    fetchPipelineReadme(pipelineKey);

    pipelineForm.innerHTML = "";

    config.fields.forEach(field => {
      const group = document.createElement("div");
      group.className = `form-group ${field.fullWidth ? "full-width" : ""}`;

      const label = document.createElement("label");
      label.textContent = field.label;
      group.appendChild(label);

      let inputElement;
      if (field.type === "textarea") {
        inputElement = document.createElement("textarea");
        inputElement.value = field.default || "";
      } else if (field.type === "select") {
        inputElement = document.createElement("select");
        field.options.forEach(opt => {
          const optEl = document.createElement("option");
          optEl.value = opt;
          optEl.textContent = opt;
          if (opt === field.default) optEl.selected = true;
          inputElement.appendChild(optEl);
        });
      } else {
        inputElement = document.createElement("input");
        inputElement.type = field.type;
        if (field.step) inputElement.step = field.step;
        inputElement.value = field.default !== undefined ? field.default : "";
      }

      inputElement.name = field.name;
      group.appendChild(inputElement);
      pipelineForm.appendChild(group);
    });

    // Reset results widget
    resultWidget.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">🔮</span>
        <p>Ready to evaluate <strong>${config.title}</strong> model.</p>
      </div>
    `;
    jsonOutput.textContent = "{}";
    execTime.textContent = "Ready";
  }

  // Populate Preset Values
  function loadPreset(presetType) {
    const config = PIPELINE_CONFIGS[activePipelineKey];
    const presetData = config.presets[presetType];
    if (!presetData) return;

    Object.keys(presetData).forEach(key => {
      const el = pipelineForm.elements[key];
      if (el) el.value = presetData[key];
    });

    runInference();
  }

  // Collect Form Payload
  function getFormPayload() {
    const formData = new FormData(pipelineForm);
    const payload = {};
    const config = PIPELINE_CONFIGS[activePipelineKey];

    config.fields.forEach(field => {
      const rawVal = formData.get(field.name);
      if (field.type === "number") {
        payload[field.name] = rawVal.includes(".") ? parseFloat(rawVal) : parseInt(rawVal, 10);
      } else {
        payload[field.name] = rawVal;
      }
    });
    return payload;
  }

  // Execute API Inference
  async function runInference() {
    const config = PIPELINE_CONFIGS[activePipelineKey];
    const payload = getFormPayload();

    resultWidget.innerHTML = `<div class="empty-state"><span class="empty-icon">⚡</span><p>Executing model inference...</p></div>`;
    btnRun.disabled = true;
    const startTime = performance.now();

    try {
      const response = await fetch(config.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const elapsed = Math.round(performance.now() - startTime);
      execTime.textContent = `⚡ ${elapsed} ms`;

      if (!response.ok) {
        throw new Error(`API Error ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      jsonOutput.textContent = JSON.stringify(data, null, 2);
      renderVisualResult(data.result);

    } catch (err) {
      resultWidget.innerHTML = `
        <div class="empty-state" style="color: var(--accent-rose);">
          <span class="empty-icon">⚠️</span>
          <p>Inference Failed: ${err.message}</p>
        </div>
      `;
      jsonOutput.textContent = JSON.stringify({ error: err.message }, null, 2);
    } finally {
      btnRun.disabled = false;
    }
  }

  // Render Styled Result Widget
  function renderVisualResult(res) {
    let html = `<div class="widget-grid">`;

    if (res.fraud_probability !== undefined) {
      const probPct = Math.round(res.fraud_probability * 100);
      const barColor = res.fraud_probability > 0.5 ? "var(--accent-rose)" : "var(--accent-emerald)";
      html += `
        <div class="stat-row">
          <span class="stat-label">Alert Status</span>
          <span class="pill ${res.fraud_probability > 0.5 ? 'red' : 'green'}">${res.alert_status}</span>
        </div>
        <div class="gauge-container">
          <div class="stat-row">
            <span class="stat-label">Fraud Probability</span>
            <span class="stat-value" style="color: ${barColor}">${probPct}%</span>
          </div>
          <div class="gauge-bar-bg"><div class="gauge-bar-fill" style="width: ${probPct}%; background: ${barColor}"></div></div>
        </div>
        <div class="stat-row"><span class="stat-label">Top Risk Factors</span><span class="stat-value">${(res.top_risk_factors || []).join(", ")}</span></div>
      `;
    } else if (res.default_probability !== undefined) {
      const probPct = Math.round(res.default_probability * 100);
      const barColor = res.default_probability > 0.3 ? "var(--accent-rose)" : "var(--accent-emerald)";
      html += `
        <div class="stat-row"><span class="stat-label">Decision</span><span class="pill ${res.underwriting_decision === 'APPROVED' ? 'green' : 'red'}">${res.underwriting_decision}</span></div>
        <div class="stat-row"><span class="stat-label">Risk Tier</span><span class="stat-value">${res.risk_tier}</span></div>
        <div class="gauge-container">
          <div class="stat-row"><span class="stat-label">Default Probability</span><span class="stat-value" style="color: ${barColor}">${probPct}%</span></div>
          <div class="gauge-bar-bg"><div class="gauge-bar-fill" style="width: ${probPct}%; background: ${barColor}"></div></div>
        </div>
      `;
    } else if (res.predicted_price !== undefined) {
      html += `
        <div class="stat-row"><span class="stat-label">Predicted Valuation</span><span class="stat-value" style="color: var(--accent-cyan); font-size: 22px;">$${res.predicted_price.toLocaleString()}</span></div>
        <div class="stat-row"><span class="stat-label">Price per SqFt</span><span class="stat-value">$${res.price_per_sqft}</span></div>
        <div class="stat-row"><span class="stat-label">Valuation Bounds</span><span class="stat-value">$${res.valuation_range.lower_bound.toLocaleString()} - $${res.valuation_range.upper_bound.toLocaleString()}</span></div>
      `;
    } else if (res.recommendations !== undefined) {
      html += `<div style="display: flex; flex-direction: column; gap: 8px;">`;
      res.recommendations.forEach(item => {
        html += `
          <div class="stat-row">
            <div><strong>${item.title}</strong> <span class="pill blue">${item.category}</span></div>
            <span class="stat-value" style="color: var(--accent-emerald)">Match ${(item.match_score * 100).toFixed(0)}%</span>
          </div>
        `;
      });
      html += `</div>`;
    } else if (res.daily_forecast !== undefined) {
      html += `
        <div class="stat-row"><span class="stat-label">Total Forecasted Demand</span><span class="stat-value" style="color: var(--accent-cyan); font-size: 20px;">${res.total_predicted_units} Units</span></div>
        <div class="stat-row"><span class="stat-label">Forecast Horizon</span><span class="stat-value">${res.forecast_horizon_days} Days</span></div>
      `;
    } else if (res.sentiment !== undefined) {
      const color = res.sentiment === "POSITIVE" ? "green" : (res.sentiment === "NEGATIVE" ? "red" : "blue");
      html += `
        <div class="stat-row"><span class="stat-label">Predicted Sentiment</span><span class="pill ${color}">${res.sentiment}</span></div>
        <div class="stat-row"><span class="stat-label">Confidence Score</span><span class="stat-value">${(res.confidence * 100).toFixed(1)}%</span></div>
        <div class="stat-row"><span class="stat-label">Composite Sentiment Score</span><span class="stat-value">${res.composite_sentiment_score}</span></div>
      `;
    } else if (res.predicted_category !== undefined) {
      html += `
        <div class="stat-row"><span class="stat-label">Document Category</span><span class="pill blue">${res.predicted_category}</span></div>
        <div class="stat-row"><span class="stat-label">Confidence</span><span class="stat-value">${(res.confidence * 100).toFixed(1)}%</span></div>
        <div class="stat-row"><span class="stat-label">Key Terms Extracted</span><span class="stat-value">${(res.top_keywords_detected || []).join(", ")}</span></div>
      `;
    } else if (res.defect_type !== undefined) {
      const isClean = res.quality_control_passed;
      html += `
        <div class="stat-row"><span class="stat-label">QC Status</span><span class="pill ${isClean ? 'green' : 'red'}">${isClean ? 'PASSED' : 'DEFECT DETECTED'}</span></div>
        <div class="stat-row"><span class="stat-label">Defect Type</span><span class="stat-value">${res.defect_type}</span></div>
        <div class="stat-row"><span class="stat-label">Severity Grade</span><span class="stat-value">${res.severity_grade}</span></div>
      `;
    } else if (res.persona_name !== undefined) {
      html += `
        <div class="stat-row"><span class="stat-label">Assigned Cluster</span><span class="pill blue">Cluster #${res.cluster_id}</span></div>
        <div class="stat-row"><span class="stat-label">Customer Persona</span><span class="stat-value" style="color: var(--accent-amber);">${res.persona_name}</span></div>
        <div class="stat-row"><span class="stat-label">Target Strategy</span><span class="stat-value">${res.marketing_strategy}</span></div>
      `;
    } else {
      // Fallback for general status
      Object.keys(res).forEach(k => {
        html += `<div class="stat-row"><span class="stat-label">${k}</span><span class="stat-value">${JSON.stringify(res[k])}</span></div>`;
      });
    }

    html += `</div>`;
    resultWidget.innerHTML = html;
  }

  // Fetch Pipeline Metrics & Metadata and Check Both Microservices
  async function fetchPipelineStatus() {
    try {
      const res = await fetch("/api/v1/pipelines/status");
      if (res.ok) {
        const data = await res.json();
        metricsContainer.innerHTML = "";
        data.pipelines.forEach(p => {
          const tag = document.createElement("div");
          tag.className = "metric-tag";
          const metricsStr = Object.entries(p.metrics).map(([k, v]) => `${k}: ${v}`).join(" | ");
          tag.innerHTML = `<strong>${p.name}:</strong> ${metricsStr || 'Trained'}`;
          metricsContainer.appendChild(tag);
        });

        const mlStatusEl = document.getElementById("apiStatusText");
        if (mlStatusEl) mlStatusEl.textContent = "Online (12 Models Ready)";
      }
    } catch (e) {
      const mlStatusEl = document.getElementById("apiStatusText");
      if (mlStatusEl) mlStatusEl.textContent = "Offline / Disconnected";
    }

    // Check Unified API status
    try {
      const courseRes = await fetch("/health");
      if (courseRes.ok) {
        const courseStatusEl = document.getElementById("courseStatusText");
        if (courseStatusEl) courseStatusEl.textContent = "Online (101 Chapters Ready)";
      }
    } catch (e) {
      const courseStatusEl = document.getElementById("courseStatusText");
      if (courseStatusEl) courseStatusEl.textContent = "Offline / Disconnected";
    }
  }

  // Event Listeners
  navMenu.addEventListener("click", (e) => {
    const button = e.target.closest(".nav-item");
    if (!button) return;

    document.querySelectorAll(".nav-item").forEach(btn => btn.classList.remove("active"));
    button.classList.add("active");

    const pipelineKey = button.dataset.pipeline;
    renderForm(pipelineKey);
  });

  btnRun.addEventListener("click", runInference);
  btnPresetHigh.addEventListener("click", () => loadPreset("highRisk"));
  btnPresetNormal.addEventListener("click", () => loadPreset("normal"));

  btnCopyJson.addEventListener("click", () => {
    navigator.clipboard.writeText(jsonOutput.textContent);
    btnCopyJson.textContent = "✓ Copied!";
    setTimeout(() => { btnCopyJson.textContent = "📋 Copy JSON"; }, 2000);
  });

  // Initialize
  renderForm("fraud_detection");
  fetchPipelineReadme("fraud_detection");
  fetchPipelineStatus();
});
