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

const UPLOADS = path.join(ROOT, "storage", "uploads");
const PYTHON_RESULTS = path.join(ROOT, "storage", "python_results");

const PORT = Number(process.env.PORT || 5050);
const PYTHON_URL =
  process.env.PYTHON_URL || "http://127.0.0.1:8000";

// -------------------------------------------------------
// DIRECTORIES
// -------------------------------------------------------

for (const directory of [UPLOADS, PYTHON_RESULTS]) {
  fs.mkdirSync(directory, { recursive: true });
}

// -------------------------------------------------------
// EXPRESS
// -------------------------------------------------------

const app = express();

app.use(cors());

app.use(
  express.json({
    limit: "2mb",
  })
);

// -------------------------------------------------------
// MULTER
// -------------------------------------------------------

const upload = multer({
  dest: UPLOADS,

  limits: {
    fileSize: 1024 * 1024 * 1024, // 1 GB
  },

  fileFilter: (_req, file, cb) => {
    const allowed =
      /\.(jpg|jpeg|png|bmp|webp|mp4|mov|avi|mkv|webm)$/i.test(
        file.originalname
      );

    if (!allowed) {
      return cb(new Error("Unsupported file type"));
    }

    cb(null, true);
  },
});

// -------------------------------------------------------
// URL REWRITER
// -------------------------------------------------------

/*
  Python returns URLs such as:

      /results/<job>/file.png

  The browser should instead request:

      /python-results/<job>/file.png

  Node then proxies that request to Python.
*/

function rewriteUrls(value) {
  // Arrays
  if (Array.isArray(value)) {
    return value.map((item) => rewriteUrls(item));
  }

  // Objects
  if (value && typeof value === "object") {
    const out = {};

    for (const [key, item] of Object.entries(value)) {
      out[key] = rewriteUrls(item);
    }

    return out;
  }

  // Strings containing Python result URLs
  if (
    typeof value === "string" &&
    value.startsWith("/results/")
  ) {
    return (
      "/python-results/" +
      value.replace(/^\/results\//, "")
    );
  }

  return value;
}

// -------------------------------------------------------
// HEALTH
// -------------------------------------------------------

app.get("/api/health", async (_req, res) => {
  try {
    const response = await fetch(
      `${PYTHON_URL}/health`
    );

    const data = await response.json();

    res.json({
      ok: true,
      node: true,
      python: data,
    });
  } catch (error) {
    console.error(
      "[health] Python service unavailable:",
      error
    );

    res.status(503).json({
      ok: false,
      node: true,
      python: false,
      error: String(error),
    });
  }
});

// -------------------------------------------------------
// ANALYSIS
// -------------------------------------------------------

app.post(
  "/api/analyze",
  upload.single("file"),
  async (req, res) => {
    console.log("\n========================================");
    console.log("[ANALYZE] New analysis request");
    console.log("========================================");

    // ---------------------------------------------------
    // CHECK FILE
    // ---------------------------------------------------

    if (!req.file) {
      console.error("[ANALYZE] No file uploaded");

      return res.status(400).json({
        error: "No file uploaded",
      });
    }

    console.log(
      "[ANALYZE] Original filename:",
      req.file.originalname
    );

    console.log(
      "[ANALYZE] MIME type:",
      req.file.mimetype
    );

    console.log(
      "[ANALYZE] Temporary multer path:",
      req.file.path
    );

    console.log(
      "[ANALYZE] Uploaded size:",
      req.file.size,
      "bytes"
    );

    // ---------------------------------------------------
    // CREATE JOB
    // ---------------------------------------------------

    const job = crypto
      .randomUUID()
      .replaceAll("-", "")
      .slice(0, 12);

    const extension =
      path
        .extname(req.file.originalname || ".bin")
        .toLowerCase();

    const savedFile = path.join(
      UPLOADS,
      `${job}${extension}`
    );

    // Move multer file to permanent job file
    fs.renameSync(
      req.file.path,
      savedFile
    );

    console.log(
      "[ANALYZE] Saved uploaded file:",
      savedFile
    );

    // ---------------------------------------------------
    // VERIFY FILE
    // ---------------------------------------------------

    if (!fs.existsSync(savedFile)) {
      console.error(
        "[ANALYZE] Saved file does not exist!"
      );

      return res.status(500).json({
        error: "Uploaded file could not be saved",
      });
    }

    const stats = fs.statSync(savedFile);

    console.log(
      "[ANALYZE] Verified file size:",
      stats.size,
      "bytes"
    );

    // ---------------------------------------------------
    // FORM DATA FOR PYTHON
    // ---------------------------------------------------

    try {
      const body = new FormData();

      const bytes = fs.readFileSync(
        savedFile
      );

      console.log(
        "[ANALYZE] Read file into memory:",
        bytes.length,
        "bytes"
      );

      // IMPORTANT:
      // This is the actual image/video being sent
      // from Node → Python.
      body.append(
        "file",
        new Blob(
          [bytes],
          {
            type:
              req.file.mimetype ||
              "application/octet-stream",
          }
        ),
        req.file.originalname
      );

      // ---------------------------------------------------
      // FORWARD ANALYSIS PARAMETERS
      // ---------------------------------------------------

      const fields = [
        "passer_team",
        "passer_index",
        "attack_red",
        "attack_blue",
        "ball_bbox",
        "detect_every",
        "calibration_every",
      ];

      console.log(
        "[ANALYZE] Forwarding parameters:"
      );

      for (const field of fields) {
        if (
          req.body[field] !== undefined &&
          req.body[field] !== ""
        ) {
          const value = String(
            req.body[field]
          );

          body.append(
            field,
            value
          );

          console.log(
            `  ${field}:`,
            value
          );
        }
      }

      // ---------------------------------------------------
      // BALL BOX DEBUGGING
      // ---------------------------------------------------

      if (req.body.ball_bbox) {
        console.log(
          "[BALL] Manual ball bounding box received:",
          req.body.ball_bbox
        );
      } else {
        console.log(
          "[BALL] No manual ball bounding box supplied"
        );
      }

      // ---------------------------------------------------
      // SEND TO PYTHON
      // ---------------------------------------------------

      console.log(
        "[ANALYZE] Sending image to Python:"
      );

      console.log(
        `  ${PYTHON_URL}/analyze`
      );

      const response = await fetch(
        `${PYTHON_URL}/analyze`,
        {
          method: "POST",
          body,
        }
      );

      console.log(
        "[ANALYZE] Python response status:",
        response.status
      );

      // ---------------------------------------------------
      // READ PYTHON RESPONSE
      // ---------------------------------------------------

      const text =
        await response.text();

      console.log(
        "[ANALYZE] Python response length:",
        text.length
      );

      let data;

      try {
        data = JSON.parse(text);
      } catch (parseError) {
        console.error(
          "[ANALYZE] Python returned non-JSON response:"
        );

        console.error(text);

        data = {
          error: text,
        };
      }

      // ---------------------------------------------------
      // PYTHON ERROR
      // ---------------------------------------------------

      if (!response.ok) {
        console.error(
          "[ANALYZE] Python analysis failed:"
        );

        console.error(data);

        return res
          .status(response.status)
          .json(data);
      }

      // ---------------------------------------------------
      // REWRITE RESULT URLS
      // ---------------------------------------------------

      const rewritten =
        rewriteUrls(data);

      console.log(
        "[ANALYZE] Analysis completed successfully"
      );

      console.log(
        "[ANALYZE] Sending result to frontend"
      );

      // ---------------------------------------------------
      // RESPONSE TO FRONTEND
      // ---------------------------------------------------

      return res.json(
        rewritten
      );

    } catch (error) {
      console.error(
        "[ANALYZE] Python analysis request failed:"
      );

      console.error(error);

      return res.status(500).json({
        error:
          "Python analysis request failed",

        detail:
          String(error),
      });
    }
  }
);

// -------------------------------------------------------
// PYTHON RESULT FILE PROXY
// -------------------------------------------------------

/*
  Browser requests:

      /python-results/<job>/file.png

  Node requests internally:

      http://127.0.0.1:8000/results/<job>/file.png
*/

app.use(
  "/python-results",
  async (req, res) => {
    const upstream =
      `${PYTHON_URL}/results${req.path}`;

    console.log(
      "[RESULT PROXY]",
      req.path,
      "→",
      upstream
    );

    try {
      const upstreamResponse =
        await fetch(upstream);

      if (!upstreamResponse.ok) {
        console.error(
          "[RESULT PROXY] Python returned:",
          upstreamResponse.status
        );

        return res
          .status(upstreamResponse.status)
          .json({
            error:
              `Python result file not found: ${req.path}`,
          });
      }

      const contentType =
        upstreamResponse.headers.get(
          "content-type"
        ) ||
        "application/octet-stream";

      res.setHeader(
        "content-type",
        contentType
      );

      const buffer =
        await upstreamResponse.arrayBuffer();

      res.end(
        Buffer.from(buffer)
      );

    } catch (proxyError) {
      console.error(
        "[RESULT PROXY] Error:",
        proxyError
      );

      return res.status(502).json({
        error:
          "Could not fetch result file from Python service",

        detail:
          String(proxyError),
      });
    }
  }
);

// -------------------------------------------------------
// FRONTEND
// -------------------------------------------------------

app.use(
  express.static(FRONTEND)
);

// Express 5 compatible fallback
app.use(
  (_req, res) => {
    res.sendFile(
      path.join(
        FRONTEND,
        "index.html"
      )
    );
  }
);

// -------------------------------------------------------
// ERROR HANDLER
// -------------------------------------------------------

app.use(
  (
    error,
    _req,
    res,
    _next
  ) => {
    console.error(
      "[EXPRESS ERROR]",
      error
    );

    res.status(400).json({
      error:
        error.message ||
        "Request failed",
    });
  }
);

// -------------------------------------------------------
// START SERVER
// -------------------------------------------------------

app.listen(
  PORT,
  "0.0.0.0",
  () => {
    console.log(
      "========================================"
    );

    console.log(
      "       PITCHVISION DINO SERVER"
    );

    console.log(
      "========================================"
    );

    console.log(
      `Website : http://localhost:${PORT}`
    );

    console.log(
      `Python  : ${PYTHON_URL}`
    );

    console.log(
      "Pitch   : Roboflow football-field detection"
    );

    console.log(
      "Ball    : DINO + optional user box/tracking"
    );

    console.log(
      "========================================"
    );
  }
);