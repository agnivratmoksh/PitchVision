const API = "/api";

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const browseBtn = document.getElementById("browseBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const selectedFileBox = document.getElementById("selectedFile");
const removeFileBtn = document.getElementById("removeFile");
const progressBox = document.getElementById("progress");
const errorBox = document.getElementById("errorBox");
const resultsSection = document.getElementById("results");

let selectedFile = null;
let pollTimer = null;


// ==================================================
// FILE SELECTION
// ==================================================

browseBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});

dropzone.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  setFile(fileInput.files?.[0]);
});


// ==================================================
// DRAG AND DROP
// ==================================================

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
  });
});

dropzone.addEventListener("drop", (event) => {
  setFile(event.dataTransfer.files?.[0]);
});


// ==================================================
// REMOVE FILE
// ==================================================

removeFileBtn.addEventListener("click", clearFile);


// ==================================================
// NEW ANALYSIS
// ==================================================

document.getElementById("newAnalysis").addEventListener("click", () => {
  clearFile();
  resultsSection.classList.add("hidden");
  document.getElementById("analyze").scrollIntoView({ behavior: "smooth" });
});


// ==================================================
// ANALYZE BUTTON
// ==================================================

analyzeBtn.addEventListener("click", startAnalysis);


// ==================================================
// SET FILE
// ==================================================

function setFile(file) {
  if (!file) return;

  const allowed = file.type.startsWith("image/") || file.type.startsWith("video/");

  if (!allowed) {
    showError("Please select an image or video.");
    return;
  }

  selectedFile = file;

  const extension = file.name.includes(".")
    ? file.name.split(".").pop().toUpperCase()
    : "FILE";

  document.getElementById("fileType").textContent = extension;
  document.getElementById("fileName").textContent = file.name;
  document.getElementById("fileMeta").textContent =
    `${file.type || "file"} · ${formatBytes(file.size)}`;

  selectedFileBox.classList.remove("hidden");
  analyzeBtn.disabled = false;
  hideError();
}


// ==================================================
// CLEAR FILE
// ==================================================

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  selectedFileBox.classList.add("hidden");
  analyzeBtn.disabled = true;
  progressBox.classList.add("hidden");
  hideError();

  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}


// ==================================================
// START ANALYSIS
// ==================================================

async function startAnalysis() {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  progressBox.classList.remove("hidden");
  hideError();
  setProgress(5, "Uploading footage...", "Browser → Node API");

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const response = await fetch(`${API}/analyze`, {
      method: "POST",
      body: form
    });

    const data = await readJson(response);

    if (!response.ok) {
      throw new Error(
        data.detail || data.error || "The server rejected the upload."
      );
    }

    await pollJob(data.job_id);
  } catch (error) {
    handleFailure(error);
  }
}


// ==================================================
// POLL JOB
// ==================================================

async function pollJob(jobId) {
  try {
    const response = await fetch(
      `${API}/jobs/${encodeURIComponent(jobId)}`
    );

    const job = await readJson(response);

    if (!response.ok) {
      throw new Error(job.error || "Could not read analysis status.");
    }

    setProgress(
      Number(job.progress || 0),
      job.message || "Processing...",
      job.stage || "Python inference"
    );

    if (job.status === "completed") {
      renderResults(job.result);
      return;
    }

    if (job.status === "failed") {
      throw new Error(
        job.error || job.message || "The Python pipeline failed."
      );
    }

    pollTimer = setTimeout(() => {
      pollJob(jobId);
    }, 900);
  } catch (error) {
    handleFailure(error);
  }
}


// ==================================================
// RENDER RESULTS
// ==================================================

function renderResults(result) {
  if (!result) {
    handleFailure(new Error("The server returned no result."));
    return;
  }

  const resultMedia = document.getElementById("resultMedia");
  resultMedia.innerHTML = "";

  // VIDEO
  if (result.annotated_video) {
    const figure = document.createElement("figure");
    figure.innerHTML = `
      <figcaption>ANNOTATED VIDEO</figcaption>
      <video controls playsinline src="${result.annotated_video}"></video>
    `;
    resultMedia.appendChild(figure);
  }

  // IMAGE
  if (result.annotated_image) {
    const figure = document.createElement("figure");
    figure.innerHTML = `
      <figcaption>ANNOTATED INPUT</figcaption>
      <img src="${result.annotated_image}" alt="Annotated input">
    `;
    resultMedia.appendChild(figure);
  }

  // PITCH
  if (result.pitch_view) {
    const figure = document.createElement("figure");
    figure.innerHTML = `
      <figcaption>TOP-DOWN PITCH</figcaption>
      <img src="${result.pitch_view}" alt="Top-down pitch">
    `;
    resultMedia.appendChild(figure);
  }

  document.getElementById("formationImage").src = result.formations || "";
  document.getElementById("heatmapImage").src = result.heatmap || "";

  renderMetrics(result.metrics || {});

  document.getElementById("downloadOutput").href = result.output_file || "#";

  setProgress(
    100,
    "Analysis complete.",
    "Detection → Classification → Mapping → Analytics"
  );

  resultsSection.classList.remove("hidden");

  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}


// ==================================================
// METRICS
// ==================================================

function renderMetrics(metrics) {
  const items = [
    ["Players", metrics.players_detected ?? "—"],
    [
      "Team A width",
      metrics.team_a_width != null
        ? `${Number(metrics.team_a_width).toFixed(1)} m`
        : "—"
    ],
    [
      "Team B width",
      metrics.team_b_width != null
        ? `${Number(metrics.team_b_width).toFixed(1)} m`
        : "—"
    ],
    [
      "Ball confidence",
      metrics.ball_confidence != null
        ? `${(Number(metrics.ball_confidence) * 100).toFixed(0)}%`
        : "—"
    ]
  ];

  document.getElementById("metricsGrid").innerHTML = items
    .map(
      ([label, value]) => `
        <div class="metric">
          <small>${label.toUpperCase()}</small>
          <strong>${value}</strong>
        </div>
      `
    )
    .join("");
}


// ==================================================
// PROGRESS
// ==================================================

function setProgress(percent, text, stage) {
  const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));

  document.getElementById("statusText").textContent = text;
  document.getElementById("statusPercent").textContent = `${Math.round(safePercent)}%`;
  document.getElementById("progressFill").style.width = `${safePercent}%`;
  document.getElementById("pipelineStatus").textContent = stage;
}


// ==================================================
// ERRORS
// ==================================================

function handleFailure(error) {
  console.error(error);
  setProgress(0, "Analysis failed.", "Check the Python server and model.");
  showError(error?.message || "Something went wrong during analysis.");
  analyzeBtn.disabled = false;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}


// ==================================================
// JSON HELPER
// ==================================================

async function readJson(response) {
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    return { error: text || `HTTP ${response.status}` };
  }
}


// ==================================================
// FILE SIZE
// ==================================================

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "unknown size";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}