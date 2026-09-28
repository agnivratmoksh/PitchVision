import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT = path.resolve(__dirname, "..");

const FRONTEND = path.join(ROOT, "frontend");
const STORAGE = path.join(ROOT, "storage");
const UPLOADS = path.join(STORAGE, "uploads");
const OUTPUTS = path.join(STORAGE, "outputs");

fs.mkdirSync(UPLOADS, { recursive: true });
fs.mkdirSync(OUTPUTS, { recursive: true });

const PORT = Number(process.env.PORT || 5050);

const PYTHON_API =
  process.env.PYTHON_API ||
  "http://127.0.0.1:8000";

const ALLOWED_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".mp4",
  ".mov",
  ".avi",
  ".mkv"
]);

const app = express();

app.disable("x-powered-by");
app.use(cors());
app.use(express.json({ limit: "2mb" }));

const upload = multer({
  dest: UPLOADS,
  limits: {
    fileSize: 500 * 1024 * 1024
  }
});

const jobs = new Map();


// ==================================================
// STATIC FILES
// ==================================================

app.use("/files", express.static(STORAGE));
app.use(express.static(FRONTEND));


// ==================================================
// FRONTEND
// ==================================================

app.get("/", (_req, res) => {
  res.sendFile(path.join(FRONTEND, "index.html"));
});


// ==================================================
// HEALTH CHECK
// ==================================================

app.get("/api/health", async (_req, res) => {
  let python = { ok: false };

  try {
    const response = await fetch(`${PYTHON_API}/health`);
    if (response.ok) {
      python = await response.json();
    }
  } catch {
    // Python server may not be running.
  }

  res.json({
    ok: true,
    service: "pitchvision-node",
    port: PORT,
    python: python
  });
});


// ==================================================
// ANALYZE
// ==================================================

app.post("/api/analyze", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      error: "No image or video was uploaded."
    });
  }

  const originalName = req.file.originalname || "input";
  const extension = path.extname(originalName).toLowerCase();

  if (!ALLOWED_EXTENSIONS.has(extension)) {
    try {
      fs.unlinkSync(req.file.path);
    } catch {}

    return res.status(400).json({
      error: `Unsupported file type: ${extension || "unknown"}`
    });
  }

  const jobId = crypto.randomUUID();
  const finalPath = path.join(UPLOADS, `${jobId}${extension}`);

  fs.renameSync(req.file.path, finalPath);

  jobs.set(jobId, {
    status: "queued",
    progress: 5,
    message: "File uploaded.",
    stage: "Backend API",
    filename: originalName,
    result: null
  });

  console.log(`New analysis job: ${jobId}`);

  try {
    const response = await fetch(`${PYTHON_API}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        job_id: jobId,
        file_path: finalPath
      })
    });

    const raw = await response.text();
    let data = {};

    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      data = { raw };
    }

    if (!response.ok) {
      throw new Error(
        data.detail ||
        data.error ||
        raw ||
        `Python service returned HTTP ${response.status}`
      );
    }

    jobs.set(jobId, {
      ...jobs.get(jobId),
      status: "processing",
      progress: 8,
      message: "Analysis started.",
      stage: "Python pipeline"
    });

    return res.status(202).json({
      job_id: jobId
    });
  } catch (error) {
    console.error("Python service error:", error);

    jobs.set(jobId, {
      ...jobs.get(jobId),
      status: "failed",
      progress: 0,
      message: "Could not start analysis.",
      stage: "Python service",
      error: error.message
    });

    return res.status(503).json({
      error: "Python inference service is not available.",
      detail: error.message,
      hint: "Start FastAPI on http://127.0.0.1:8000."
    });
  }
});


// ==================================================
// JOB STATUS
// ==================================================

app.get("/api/jobs/:jobId", async (req, res) => {
  const { jobId } = req.params;
  const localJob = jobs.get(jobId);

  try {
    const response = await fetch(`${PYTHON_API}/jobs/${jobId}`);

    if (response.ok) {
      const remoteJob = await response.json();
      const merged = {
        ...(localJob || {}),
        ...remoteJob
      };

      jobs.set(jobId, merged);
      return res.json(merged);
    }
  } catch {
    // Fall back to local job state.
  }

  if (localJob) {
    return res.json(localJob);
  }

  return res.status(404).json({
    error: "Job not found."
  });
});


// ==================================================
// 404
// ==================================================

app.use((req, res) => {
  res.status(404).json({
    error: "Route not found",
    path: req.originalUrl
  });
});


// ==================================================
// START SERVER
// ==================================================

app.listen(PORT, "0.0.0.0", () => {
  console.log("");
  console.log("==============================================");
  console.log("              PITCHVISION");
  console.log("==============================================");
  console.log(`Website : http://localhost:${PORT}`);
  console.log(`Health  : http://localhost:${PORT}/api/health`);
  console.log(`Python  : ${PYTHON_API}`);
  console.log(`Frontend: ${FRONTEND}`);
  console.log("==============================================");
  console.log("");
});