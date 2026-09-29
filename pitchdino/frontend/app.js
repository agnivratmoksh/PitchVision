"use strict";

/* ======================================================
   DOM HELPERS
====================================================== */

const $ = (id) => document.getElementById(id);


/* ======================================================
   DOM ELEMENTS
====================================================== */

const fileInput = $("fileInput");
const dropZone = $("dropZone");
const browseBtn = $("browseBtn");
const heroUploadBtn = $("heroUploadBtn");

const previewCard = $("previewCard");
const imagePreview = $("imagePreview");
const videoPreview = $("videoPreview");

const canvas = $("overlay");
const ctx = canvas?.getContext("2d");
const stage = $("stage");
const mediaLoading = $("mediaLoading");

const markBallBtn = $("markBallBtn");
const clearBallBtn = $("clearBallBtn");
const clearBtn = $("clearBtn");

const analyzeBtn = $("analyzeBtn");

const statusBox = $("status");
const results = $("results");

const markingBanner = $("markingBanner");
const ballStatus = $("ballStatus");


/* ======================================================
   STATE
====================================================== */

let file = null;
let objectUrl = null;

let marking = false;
let start = null;

/*
 * Ball box is ALWAYS stored in the original media
 * coordinate system:
 *
 * [x1, y1, x2, y2]
 *
 * Example for a 1920x1080 image:
 * [850, 420, 890, 460]
 */
let ballBox = null;

let resizeFrame = 0;


/* ======================================================
   FILE TYPES
====================================================== */

const IMAGE_EXTENSIONS =
  /\.(jpg|jpeg|png|bmp|webp)$/i;

const VIDEO_EXTENSIONS =
  /\.(mp4|mov|avi|mkv|webm)$/i;


/* ======================================================
   INITIALIZATION CHECK
====================================================== */

if (
  !fileInput ||
  !dropZone ||
  !canvas ||
  !ctx ||
  !stage ||
  !imagePreview ||
  !videoPreview
) {
  console.error(
    "PitchVision: required frontend elements are missing."
  );
}


/* ======================================================
   FILE PICKER
====================================================== */

/*
 * IMPORTANT:
 *
 * The latest HTML uses:
 *
 * <label for="fileInput">Choose file</label>
 *
 * Therefore we DO NOT programmatically call:
 *
 * fileInput.click()
 *
 * from the main Choose File button.
 *
 * The browser handles the native picker through
 * the label/input relationship.
 */


/*
 * Hero upload button
 *
 * If the homepage contains a "Start analysis" or
 * similar button with id="heroUploadBtn", scroll to
 * the workspace.
 */
heroUploadBtn?.addEventListener("click", () => {

  $("workspace")?.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });

});


/*
 * The browse button is a <label>, not a normal button.
 *
 * Therefore we intentionally do nothing here.
 */
browseBtn?.addEventListener("click", (event) => {

  /*
   * Allow the browser's native label → input behavior.
   */
});


/* ======================================================
   FILE INPUT
====================================================== */

fileInput?.addEventListener("change", () => {

  const selectedFile =
    fileInput.files?.[0];

  if (!selectedFile) {
    return;
  }

  loadFile(selectedFile);

});


/* ======================================================
   DRAG AND DROP
====================================================== */

["dragenter", "dragover"].forEach(
  (eventName) => {

    dropZone?.addEventListener(
      eventName,
      (event) => {

        event.preventDefault();
        event.stopPropagation();

        dropZone.classList.add("drag");

      }
    );

  }
);


["dragleave", "drop"].forEach(
  (eventName) => {

    dropZone?.addEventListener(
      eventName,
      (event) => {

        event.preventDefault();
        event.stopPropagation();

        dropZone.classList.remove("drag");

      }
    );

  }
);


dropZone?.addEventListener(
  "drop",
  (event) => {

    const droppedFile =
      event.dataTransfer?.files?.[0];

    if (!droppedFile) {
      return;
    }

    loadFile(droppedFile);

  }
);


/* ======================================================
   RESET / CHANGE FILE
====================================================== */

/*
 * The Change File control in the latest HTML is a
 * <label for="fileInput">.
 *
 * We reset the current state before the browser
 * opens the file picker.
 */
clearBtn?.addEventListener(
  "click",
  () => {

    resetApplication();

  }
);


function resetApplication() {

  cancelMarking();

  /*
   * Release previous object URL.
   */
  if (objectUrl) {

    URL.revokeObjectURL(objectUrl);

  }

  objectUrl = null;

  file = null;

  ballBox = null;

  start = null;


  /*
   * Remove old media sources.
   */
  imagePreview.onload = null;
  imagePreview.onerror = null;

  videoPreview.removeAttribute("src");

  imagePreview.removeAttribute("src");

  videoPreview.load();


  /*
   * Reset canvas.
   */
  clearCanvas();


  /*
   * Reset file input.
   *
   * This is important because it allows selecting
   * the SAME file again.
   */
  fileInput.value = "";


  /*
   * Reset UI.
   */
  previewCard?.classList.add("hidden");

  dropZone?.classList.remove("hidden");

  results?.classList.add("hidden");

  statusBox?.classList.add("hidden");


  if ($("warnings")) {
    $("warnings").innerHTML = "";
  }

  if ($("visuals")) {
    $("visuals").innerHTML = "";
  }

  if ($("json")) {
    $("json").textContent = "";
  }

  if ($("fileName")) {
    $("fileName").textContent = "—";
  }

  if ($("fileMeta")) {
    $("fileMeta").textContent =
      "Ready for analysis";
  }


  hideMediaLoading();


  setBallStatus(
    "auto",
    "Automatic ball detection enabled"
  );

}


/* ======================================================
   LOAD FILE
====================================================== */

function loadFile(selectedFile) {

  if (!selectedFile) {
    return;
  }


  const isImage =
    IMAGE_EXTENSIONS.test(
      selectedFile.name
    );

  const isVideo =
    VIDEO_EXTENSIONS.test(
      selectedFile.name
    );


  /*
   * Validate extension.
   */
  if (!isImage && !isVideo) {

    setStatus(
      "Unsupported file type. Use JPG, PNG, WEBP, BMP, MP4, MOV, AVI, MKV or WEBM.",
      true
    );

    return;

  }


  /*
   * Release previous object URL.
   */
  if (objectUrl) {

    URL.revokeObjectURL(objectUrl);

  }


  /*
   * Save selected file.
   */
  file = selectedFile;

  objectUrl =
    URL.createObjectURL(
      selectedFile
    );


  /*
   * Reset ball selection.
   */
  ballBox = null;

  cancelMarking();

  clearCanvas();


  /*
   * Update UI.
   */
  dropZone?.classList.add("hidden");

  previewCard?.classList.remove("hidden");

  results?.classList.add("hidden");

  statusBox?.classList.add("hidden");


  if ($("fileName")) {

    $("fileName").textContent =
      selectedFile.name;

  }


  if ($("fileMeta")) {

    $("fileMeta").textContent =
      `${formatBytes(selectedFile.size)} · ${
        isVideo ? "Video" : "Image"
      }`;

  }


  /*
   * Hide both media elements first.
   */
  imagePreview.classList.add("hidden");

  videoPreview.classList.add("hidden");


  /*
   * Load correct media type.
   */
  if (isVideo) {

    loadVideo(selectedFile);

  } else {

    loadImage(selectedFile);

  }


  setBallStatus(
    "auto",
    "Automatic ball detection enabled"
  );

}


/* ======================================================
   IMAGE LOADING
====================================================== */

function loadImage(selectedFile) {

  showMediaLoading();


  imagePreview.classList.remove(
    "hidden"
  );


  imagePreview.src =
    objectUrl;


  imagePreview.onload = () => {

    hideMediaLoading();

    scheduleResize();

  };


  imagePreview.onerror = () => {

    hideMediaLoading();

    setStatus(
      "The selected image could not be loaded by the browser.",
      true
    );

  };

}


/* ======================================================
   VIDEO LOADING
====================================================== */

function loadVideo(selectedFile) {

  showMediaLoading();


  videoPreview.classList.remove(
    "hidden"
  );


  videoPreview.src =
    objectUrl;


  videoPreview.load();


  const ready = () => {

    hideMediaLoading();

    scheduleResize();

  };


  videoPreview.addEventListener(
    "loadedmetadata",
    ready,
    { once: true }
  );


  videoPreview.addEventListener(
    "loadeddata",
    ready,
    { once: true }
  );


  videoPreview.addEventListener(
    "error",
    () => {

      hideMediaLoading();

      setStatus(
        "The selected video could not be decoded by this browser.",
        true
      );

    },
    { once: true }
  );

}


/* ======================================================
   LOADING INDICATOR
====================================================== */

function showMediaLoading() {

  mediaLoading?.classList.remove(
    "hidden"
  );

}


function hideMediaLoading() {

  mediaLoading?.classList.add(
    "hidden"
  );

}


/* ======================================================
   BALL MARKING MODE
====================================================== */

markBallBtn?.addEventListener(
  "click",
  () => {

    if (!file) {

      setStatus(
        "Choose an image first.",
        true
      );

      return;

    }


    if (!mediaIsReady()) {

      setStatus(
        "The preview is still loading. Please wait a moment.",
        true
      );

      return;

    }


    /*
     * Enter marking mode.
     */
    marking = true;

    start = null;


    /*
     * Update UI.
     */
    stage.classList.add(
      "marking-active"
    );


    canvas.classList.remove(
      "not-marking"
    );


    canvas.classList.add(
      "marking"
    );


    markingBanner?.classList.remove(
      "hidden"
    );


    markBallBtn.classList.add(
      "active"
    );


    /*
     * Make sure canvas is perfectly aligned
     * before drawing.
     */
    resizeCanvas();


    setBallStatus(
      "drawing",
      "Drag a rectangle around the ball…"
    );

  }
);


/* ======================================================
   CLEAR BALL BOX
====================================================== */

clearBallBtn?.addEventListener(
  "click",
  () => {

    ballBox = null;

    cancelMarking();

    clearCanvas();


    setBallStatus(
      "auto",
      "Automatic ball detection enabled"
    );

  }
);


/* ======================================================
   CANCEL MARKING MODE
====================================================== */

function cancelMarking() {

  marking = false;

  start = null;


  stage?.classList.remove(
    "marking-active"
  );


  canvas?.classList.remove(
    "marking"
  );


  canvas?.classList.add(
    "not-marking"
  );


  markingBanner?.classList.add(
    "hidden"
  );


  markBallBtn?.classList.remove(
    "active"
  );

}


/* ======================================================
   BALL STATUS
====================================================== */

function setBallStatus(
  mode,
  text
) {

  if (!ballStatus) {
    return;
  }


  ballStatus.className =
    `ball-status-badge mode-${mode}`;


  ballStatus.innerHTML =
    `<span class="ball-status-dot"></span>${escapeHtml(text)}`;

}


/* ======================================================
   ACTIVE MEDIA
====================================================== */

function activeMedia() {

  if (
    imagePreview &&
    !imagePreview.classList.contains(
      "hidden"
    )
  ) {

    return imagePreview;

  }


  return videoPreview;

}


/* ======================================================
   NATURAL MEDIA SIZE
====================================================== */

function getNaturalSize(media) {

  return {

    width:
      media.naturalWidth ||
      media.videoWidth ||
      0,

    height:
      media.naturalHeight ||
      media.videoHeight ||
      0

  };

}


/* ======================================================
   CHECK MEDIA READY
====================================================== */

function mediaIsReady() {

  const media =
    activeMedia();


  const natural =
    getNaturalSize(media);


  const rect =
    media.getBoundingClientRect();


  return (
    natural.width > 0 &&
    natural.height > 0 &&
    rect.width > 0 &&
    rect.height > 0
  );

}


/* ======================================================
   CANVAS RESIZE SCHEDULING
====================================================== */

function scheduleResize() {

  cancelAnimationFrame(
    resizeFrame
  );


  resizeFrame =
    requestAnimationFrame(
      () => {

        resizeCanvas();


        /*
         * Second frame handles cases where the
         * browser has not finished layout yet.
         */
        requestAnimationFrame(
          resizeCanvas
        );

      }
    );

}


/* ======================================================
   RESIZE CANVAS
====================================================== */

/*
 * VERY IMPORTANT:
 *
 * The canvas is NOT placed over the entire stage.
 *
 * It is placed exactly over the visible image/video.
 *
 * This prevents the "dead area" problem where the
 * pointer stops working in some sections.
 */

function resizeCanvas() {

  const media =
    activeMedia();


  if (
    !media ||
    media.classList.contains(
      "hidden"
    )
  ) {

    return;

  }


  const natural =
    getNaturalSize(media);


  const stageRect =
    stage.getBoundingClientRect();


  const mediaRect =
    media.getBoundingClientRect();


  if (
    !natural.width ||
    !natural.height ||
    !mediaRect.width ||
    !mediaRect.height ||
    !stageRect.width ||
    !stageRect.height
  ) {

    return;

  }


  /*
   * Canvas internal drawing resolution equals
   * the displayed media dimensions.
   */
  canvas.width =
    Math.max(
      1,
      Math.round(
        mediaRect.width
      )
    );


  canvas.height =
    Math.max(
      1,
      Math.round(
        mediaRect.height
      )
    );


  /*
   * Position canvas relative to stage.
   */
  const left =
    mediaRect.left -
    stageRect.left;


  const top =
    mediaRect.top -
    stageRect.top;


  canvas.style.left =
    `${left}px`;


  canvas.style.top =
    `${top}px`;


  canvas.style.width =
    `${mediaRect.width}px`;


  canvas.style.height =
    `${mediaRect.height}px`;


  /*
   * Redraw previously selected box.
   */
  drawBox();

}


/* ======================================================
   RESIZE / ORIENTATION
====================================================== */

window.addEventListener(
  "resize",
  scheduleResize
);


window.addEventListener(
  "orientationchange",
  scheduleResize
);


videoPreview?.addEventListener(
  "loadedmetadata",
  scheduleResize
);


videoPreview?.addEventListener(
  "loadeddata",
  scheduleResize
);


videoPreview?.addEventListener(
  "resize",
  scheduleResize
);


/*
 * ResizeObserver makes the overlay follow
 * changes to the stage/media size.
 */
if (
  typeof ResizeObserver !==
  "undefined"
) {

  const observer =
    new ResizeObserver(
      () => {

        scheduleResize();

      }
    );


  observer.observe(stage);

  observer.observe(imagePreview);

  observer.observe(videoPreview);

}


/* ======================================================
   POINTER POSITION
====================================================== */

/*
 * Convert browser pointer coordinates into
 * canvas coordinates.
 *
 * Since the canvas itself is positioned directly
 * over the media, this is now reliable even when
 * the image is centered inside a larger stage.
 */

function pointerPos(event) {

  const rect =
    canvas.getBoundingClientRect();


  if (
    !rect.width ||
    !rect.height
  ) {

    return {
      x: 0,
      y: 0
    };

  }


  const scaleX =
    canvas.width /
    rect.width;


  const scaleY =
    canvas.height /
    rect.height;


  return {

    x: clamp(
      (
        event.clientX -
        rect.left
      ) * scaleX,

      0,
      canvas.width
    ),


    y: clamp(
      (
        event.clientY -
        rect.top
      ) * scaleY,

      0,
      canvas.height
    )

  };

}


/* ======================================================
   POINTER DOWN
====================================================== */

canvas?.addEventListener(
  "pointerdown",
  (event) => {

    if (!marking) {
      return;
    }


    event.preventDefault();


    start =
      pointerPos(event);


    try {

      canvas.setPointerCapture(
        event.pointerId
      );

    } catch (_) {}

  }
);


/* ======================================================
   POINTER MOVE
====================================================== */

canvas?.addEventListener(
  "pointermove",
  (event) => {

    if (
      !marking ||
      !start
    ) {

      return;

    }


    event.preventDefault();


    const point =
      pointerPos(event);


    drawBox([

      start.x,
      start.y,

      point.x,
      point.y

    ]);

  }
);


/* ======================================================
   POINTER UP
====================================================== */

canvas?.addEventListener(
  "pointerup",
  (event) => {

    if (!start) {
      return;
    }


    event.preventDefault();


    const point =
      pointerPos(event);


    /*
     * Normalize drag direction.
     */
    const x1 =
      Math.min(
        start.x,
        point.x
      );


    const y1 =
      Math.min(
        start.y,
        point.y
      );


    const x2 =
      Math.max(
        start.x,
        point.x
      );


    const y2 =
      Math.max(
        start.y,
        point.y
      );


    /*
     * Ignore accidental tiny clicks.
     */
    if (
      x2 - x1 > 4 &&
      y2 - y1 > 4
    ) {

      const media =
        activeMedia();


      const natural =
        getNaturalSize(media);


      if (
        natural.width &&
        natural.height &&
        canvas.width &&
        canvas.height
      ) {

        /*
         * Convert DISPLAY coordinates
         * back to ORIGINAL image coordinates.
         */
        ballBox = [

          (
            x1 /
            canvas.width
          ) * natural.width,

          (
            y1 /
            canvas.height
          ) * natural.height,

          (
            x2 /
            canvas.width
          ) * natural.width,

          (
            y2 /
            canvas.height
          ) * natural.height

        ].map(
          (value) =>
            Math.round(value)
        );


        setBallStatus(
          "manual",
          `Manual ball box: ${ballBox.join(", ")}`
        );


        /*
         * Leave drawing mode.
         */
        cancelMarking();


        /*
         * Draw confirmed box.
         */
        drawBox();

      }

    } else {

      setBallStatus(
        "drawing",
        "Box too small — drag a larger rectangle around the ball."
      );

    }


    start = null;


    try {

      canvas.releasePointerCapture(
        event.pointerId
      );

    } catch (_) {}

  }
);


/* ======================================================
   POINTER CANCEL
====================================================== */

canvas?.addEventListener(
  "pointercancel",
  (event) => {

    start = null;


    try {

      canvas.releasePointerCapture(
        event.pointerId
      );

    } catch (_) {}


    drawBox();

  }
);


/* ======================================================
   DRAW BALL BOX
====================================================== */

function drawBox(
  temp = null
) {

  ctx.clearRect(
    0,
    0,
    canvas.width,
    canvas.height
  );


  let box =
    temp;


  const isTemp =
    temp !== null;


  /*
   * If there is no temporary drag,
   * convert saved ORIGINAL coordinates
   * back into DISPLAY coordinates.
   */
  if (
    !box &&
    ballBox
  ) {

    const natural =
      getNaturalSize(
        activeMedia()
      );


    if (
      !natural.width ||
      !natural.height
    ) {

      return;

    }


    box = [

      (
        ballBox[0] /
        natural.width
      ) * canvas.width,

      (
        ballBox[1] /
        natural.height
      ) * canvas.height,

      (
        ballBox[2] /
        natural.width
      ) * canvas.width,

      (
        ballBox[3] /
        natural.height
      ) * canvas.height

    ];

  }


  if (!box) {
    return;
  }


  let [
    x1,
    y1,
    x2,
    y2
  ] = box;


  /*
   * Normalize and clamp.
   */
  const normalizedX1 =
    clamp(
      Math.min(x1, x2),
      0,
      canvas.width
    );


  const normalizedY1 =
    clamp(
      Math.min(y1, y2),
      0,
      canvas.height
    );


  const normalizedX2 =
    clamp(
      Math.max(x1, x2),
      0,
      canvas.width
    );


  const normalizedY2 =
    clamp(
      Math.max(y1, y2),
      0,
      canvas.height
    );


  x1 = normalizedX1;
  y1 = normalizedY1;
  x2 = normalizedX2;
  y2 = normalizedY2;


  /* --------------------------------------------------
     BOX
  -------------------------------------------------- */

  ctx.save();


  /*
   * Temporary drawing:
   * orange dashed.
   *
   * Confirmed drawing:
   * white solid.
   */
  ctx.strokeStyle =
    isTemp
      ? "#f97316"
      : "#ffffff";


  ctx.lineWidth =
    isTemp
      ? 2
      : 3;


  ctx.setLineDash(
    isTemp
      ? [6, 4]
      : []
  );


  ctx.strokeRect(
    x1,
    y1,
    x2 - x1,
    y2 - y1
  );


  ctx.setLineDash([]);


  /*
   * Fill.
   */
  ctx.fillStyle =
    isTemp
      ? "rgba(249,115,22,0.18)"
      : "rgba(255,255,255,0.14)";


  ctx.fillRect(
    x1,
    y1,
    x2 - x1,
    y2 - y1
  );


  /*
   * Corner handles.
   */
  if (!isTemp) {

    const size = 8;


    ctx.fillStyle =
      "#f97316";


    const corners = [

      [x1, y1],

      [x2, y1],

      [x1, y2],

      [x2, y2]

    ];


    corners.forEach(
      ([cx, cy]) => {

        ctx.fillRect(

          cx -
            size / 2,

          cy -
            size / 2,

          size,

          size

        );

      }
    );

  }


  ctx.restore();

}


/* ======================================================
   CLEAR CANVAS
====================================================== */

function clearCanvas() {

  if (!ctx) {
    return;
  }


  ctx.clearRect(
    0,
    0,
    canvas.width,
    canvas.height
  );

}


/* ======================================================
   ANALYZE
====================================================== */

analyzeBtn?.addEventListener(
  "click",
  async () => {

    /*
     * Make sure a file exists.
     */
    if (!file) {

      setStatus(
        "Choose an image or video first.",
        true
      );

      return;

    }


    /*
     * Disable button during request.
     */
    const originalText =
      analyzeBtn.textContent;


    analyzeBtn.disabled =
      true;


    analyzeBtn.textContent =
      "Running analysis…";


    setStatus(
      "Uploading and running DINO + Roboflow…"
    );


    results?.classList.add(
      "hidden"
    );


    /*
     * Build multipart request.
     */
    const form =
      new FormData();


    form.append(
      "file",
      file,
      file.name
    );


    /*
     * Manual ball box.
     */
    if (ballBox) {

      form.append(
        "ball_bbox",
        ballBox.join(",")
      );

    }


    /*
     * Detection interval.
     */
    const detectInput =
      $("detectEvery");


    const detectEvery =
      Math.max(
        1,
        Number.parseInt(
          detectInput?.value,
          10
        ) || 1
      );


    /*
     * Calibration interval.
     */
    const calibrationInput =
      $("calibrationEvery");


    const calibrationEvery =
      Math.max(
        1,
        Number.parseInt(
          calibrationInput?.value,
          10
        ) || 30
      );


    form.append(
      "detect_every",
      String(detectEvery)
    );


    form.append(
      "calibration_every",
      String(calibrationEvery)
    );


    try {

      /*
       * Send to Node.js.
       *
       * Node.js should proxy this to
       * FastAPI on port 8000.
       */
      const response =
        await fetch(
          "/api/analyze",
          {
            method: "POST",
            body: form
          }
        );


      /*
       * Handle both JSON and plain-text
       * server responses.
       */
      const contentType =
        response.headers.get(
          "content-type"
        ) || "";


      let data;


      if (
        contentType.includes(
          "application/json"
        )
      ) {

        data =
          await response.json();

      } else {

        data = {
          error:
            await response.text()
        };

      }


      /*
       * HTTP error.
       */
      if (!response.ok) {

        throw new Error(
          data.detail ||
          data.error ||
          `Analysis failed (${response.status})`
        );

      }


      /*
       * Render successful result.
       */
      renderResults(data);

    } catch (error) {

      console.error(
        "PitchVision analysis error:",
        error
      );


      setStatus(
        error?.message ||
        "Analysis failed.",
        true
      );

    } finally {

      /*
       * Restore button.
       */
      analyzeBtn.disabled =
        false;


      analyzeBtn.textContent =
        originalText;

    }

  }
);


/* ======================================================
   STATUS
====================================================== */

function setStatus(
  message,
  error = false
) {

  if (!statusBox) {
    return;
  }


  statusBox.textContent =
    message;


  statusBox.classList.remove(
    "hidden"
  );


  statusBox.classList.toggle(
    "error",
    error
  );

}


/* ======================================================
   RENDER RESULTS
====================================================== */

function renderResults(data) {

  setStatus(
    "Analysis complete."
  );


  results?.classList.remove(
    "hidden"
  );


  /*
   * Video response may contain:
   *
   * data.final_analysis
   *
   * Image response is directly
   * represented by data.
   */
  const analysis =
    data.mode === "video"
      ? (
          data.final_analysis ||
          {}
        )
      : data;


  const calibration =
    analysis.calibration ||
    data.calibration ||
    {};


  const detections =
    Array.isArray(
      analysis.detections
    )
      ? analysis.detections
      : [];


  /*
   * Count players.
   */
  const playerCount =
    detections.filter(
      (item) =>
        item?.class === "player"
    ).length;


  /*
   * Determine ball state.
   */
  const ballDetected =
    Boolean(
      data.ball_input
        ?.final_ball_available
      ??
      analysis.ball?.detected
    );


  /*
   * Metrics.
   */
  if ($("metrics")) {

    const metrics = [

      [
        "Mode",
        data.mode ||
        "image"
      ],

      [
        "Pitch",
        `${calibration.method || "—"} · ${
          calibration.quality || "—"
        }`
      ],

      [
        "Players",
        playerCount
      ],

      [
        "Ball",
        ballDetected
          ? "tracked"
          : "not found"
      ]

    ];


    $("metrics").innerHTML =
      metrics
        .map(
          ([label, value]) => `

            <div class="metric">

              <b>
                ${escapeHtml(value)}
              </b>

              <small>
                ${escapeHtml(label)}
              </small>

            </div>

          `
        )
        .join("");

  }


  /* ==================================================
     WARNINGS
  ================================================== */

  const warnings = [

    ...(Array.isArray(
      analysis.warnings
    )
      ? analysis.warnings
      : []),

    ...(Array.isArray(
      data.final_analysis?.warnings
    )
      ? data.final_analysis.warnings
      : [])

  ];


  if ($("warnings")) {

    $("warnings").innerHTML =
      [
        ...new Set(warnings)
      ]
        .map(
          (warning) =>
            `
              <div class="warning">
                ${escapeHtml(warning)}
              </div>
            `
        )
        .join("");

  }


  /* ==================================================
     VISUAL RESULT FILES
  ================================================== */

  const files =
    analysis.files ||
    {};


  const images = [];


  function walk(value) {

    /*
     * Direct image path.
     */
    if (
      typeof value ===
        "string" &&
      /\.(png|jpg|jpeg|webp)$/i.test(
        value
      )
    ) {

      images.push(value);

      return;

    }


    /*
     * Nested object/array.
     */
    if (
      value &&
      typeof value ===
        "object"
    ) {

      Object.values(value)
        .forEach(walk);

    }

  }


  walk(files);


  if ($("visuals")) {

    $("visuals").innerHTML =
      images
        .slice(0, 10)
        .map(
          (url) => {

            const safeUrl =
              escapeAttribute(
                resolveUrl(url)
              );


            return `

              <img
                src="${safeUrl}"
                loading="lazy"
                alt="PitchVision analysis result"
              />

            `;

          }
        )
        .join("");

  }


  /* ==================================================
     RAW JSON
  ================================================== */

  if ($("json")) {

    $("json").textContent =
      JSON.stringify(
        data,
        null,
        2
      );

  }


  /*
   * Scroll to results.
   */
  window.setTimeout(
    () => {

      results?.scrollIntoView({
        behavior: "smooth",
        block: "start"
      });

    },
    100
  );

}


/* ======================================================
   RESOLVE RESULT URL
====================================================== */

function resolveUrl(url) {

  if (!url) {
    return "";
  }


  /*
   * Absolute HTTP/HTTPS URL.
   */
  if (
    /^https?:\/\//i.test(
      url
    )
  ) {

    return url;

  }


  /*
   * Already absolute local path.
   */
  if (
    url.startsWith("/")
  ) {

    return url;

  }


  /*
   * Python may return:
   *
   * results/file.png
   *
   * or:
   *
   * /results/file.png
   */
  return `/${String(url).replace(
    /^\/+/,
    ""
  )}`;

}


/* ======================================================
   CLAMP
====================================================== */

function clamp(
  value,
  min,
  max
) {

  return Math.max(
    min,
    Math.min(
      max,
      value
    )
  );

}


/* ======================================================
   FORMAT FILE SIZE
====================================================== */

function formatBytes(
  bytes
) {

  if (
    !Number.isFinite(bytes) ||
    bytes <= 0
  ) {

    return "0 B";

  }


  const units = [
    "B",
    "KB",
    "MB",
    "GB"
  ];


  const index =
    Math.min(
      Math.floor(
        Math.log(bytes) /
        Math.log(1024)
      ),
      units.length - 1
    );


  const value =
    bytes /
    Math.pow(
      1024,
      index
    );


  return `${value.toFixed(
    index === 0
      ? 0
      : 1
  )} ${units[index]}`;

}


/* ======================================================
   ESCAPE HTML
====================================================== */

function escapeHtml(
  value
) {

  return String(
    value ?? ""
  ).replace(
    /[&<>'"]/g,
    (char) => ({

      "&": "&amp;",

      "<": "&lt;",

      ">": "&gt;",

      "'": "&#39;",

      '"': "&quot;"

    }[char])
  );

}


/* ======================================================
   ESCAPE ATTRIBUTE
====================================================== */

function escapeAttribute(
  value
) {

  return escapeHtml(
    value
  );

}