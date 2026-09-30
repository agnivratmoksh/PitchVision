"use strict";

/* =========================================================
   PITCHVISION FRONTEND

   Features:
   - Image + video upload
   - Drag/drop
   - Manual ball selection
   - Backend analysis
   - Results carousel
   - Keyboard carousel navigation
========================================================= */


/* =========================================================
   DOM
========================================================= */

const $ = (id) => document.getElementById(id);

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


/* =========================================================
   STATE
========================================================= */

let file = null;
let objectUrl = null;

let marking = false;
let start = null;
let ballBox = null;

let resizeFrame = 0;


/* =========================================================
   RESULT CAROUSEL STATE
========================================================= */

let resultImages = [];
let currentResultIndex = 0;


/* =========================================================
   FILE TYPES
========================================================= */

const IMAGE_EXTENSIONS =
  /\.(jpg|jpeg|png|bmp|webp|gif|avif|tif|tiff)$/i;

const VIDEO_EXTENSIONS =
  /\.(mp4|mov|avi|mkv|webm|m4v|mpeg|mpg)$/i;


/* =========================================================
   INITIALIZATION
========================================================= */

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


/* =========================================================
   HERO BUTTON
========================================================= */

heroUploadBtn?.addEventListener("click", () => {

  $("workspace")?.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });

});


/* =========================================================
   FILE PICKER
========================================================= */

browseBtn?.addEventListener("click", (event) => {

  /*
   * browseBtn is normally a label connected
   * to the file input.
   */

  event.stopPropagation();

});


fileInput?.addEventListener("change", () => {

  const selectedFile =
    fileInput.files?.[0];

  if (selectedFile) {
    loadFile(selectedFile);
  }

});


/* =========================================================
   DRAG & DROP
========================================================= */

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

    if (droppedFile) {
      loadFile(droppedFile);
    }

  }
);


/* =========================================================
   FILE RESET
========================================================= */

clearBtn?.addEventListener(
  "click",
  () => {
    resetApplication();
  }
);


function resetApplication() {

  cancelMarking();

  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
  }

  objectUrl = null;

  file = null;
  ballBox = null;
  start = null;

  resultImages = [];
  currentResultIndex = 0;

  imagePreview.onload = null;
  imagePreview.onerror = null;

  imagePreview.removeAttribute("src");

  videoPreview.removeAttribute("src");
  videoPreview.load();

  clearCanvas();

  if (fileInput) {
    fileInput.value = "";
  }

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


/* =========================================================
   FILE TYPE DETECTION
========================================================= */

function detectFileKind(selectedFile) {

  const name =
    String(selectedFile.name || "");

  const type =
    String(selectedFile.type || "")
      .toLowerCase();


  const imageByMime =
    type.startsWith("image/");

  const videoByMime =
    type.startsWith("video/");


  const imageByExtension =
    IMAGE_EXTENSIONS.test(name);

  const videoByExtension =
    VIDEO_EXTENSIONS.test(name);


  if (
    imageByMime ||
    imageByExtension
  ) {
    return "image";
  }


  if (
    videoByMime ||
    videoByExtension
  ) {
    return "video";
  }


  return null;

}


/* =========================================================
   LOAD FILE
========================================================= */

function loadFile(selectedFile) {

  if (!selectedFile) {
    return;
  }


  const kind =
    detectFileKind(selectedFile);


  if (!kind) {

    setStatus(
      `Unsupported media type: ${
        selectedFile.type || "unknown"
      }. Please choose JPG, PNG, WEBP, BMP, MP4, MOV, AVI, MKV or WEBM.`,
      true
    );

    return;
  }


  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
  }


  file = selectedFile;

  objectUrl =
    URL.createObjectURL(selectedFile);

  ballBox = null;

  cancelMarking();
  clearCanvas();


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
        kind === "video"
          ? "Video"
          : "Image"
      }`;

  }


  imagePreview?.classList.add("hidden");
  videoPreview?.classList.add("hidden");


  if (kind === "video") {
    loadVideo();
  } else {
    loadImage();
  }


  setBallStatus(
    "auto",
    "Automatic ball detection enabled"
  );

}


/* =========================================================
   LOAD IMAGE
========================================================= */

function loadImage() {

  showMediaLoading();

  imagePreview?.classList.remove("hidden");


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


  imagePreview.src = objectUrl;

}


/* =========================================================
   LOAD VIDEO
========================================================= */

function loadVideo() {

  showMediaLoading();

  videoPreview?.classList.remove("hidden");

  videoPreview.src = objectUrl;

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
        "The selected video could not be decoded by this browser. You can still upload MP4/WebM files supported by your browser.",
        true
      );

    },
    { once: true }
  );

}


/* =========================================================
   MEDIA LOADING
========================================================= */

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


/* =========================================================
   MEDIA GEOMETRY
========================================================= */

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


function mediaIsReady() {

  const media =
    activeMedia();

  if (!media) {
    return false;
  }


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


/* =========================================================
   RESIZE
========================================================= */

function scheduleResize() {

  cancelAnimationFrame(
    resizeFrame
  );


  resizeFrame =
    requestAnimationFrame(() => {

      resizeCanvas();

      requestAnimationFrame(() => {
        resizeCanvas();
      });

    });

}


function resizeCanvas() {

  const media =
    activeMedia();


  if (
    !media ||
    media.classList.contains("hidden")
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
    !mediaRect.height
  ) {
    return;
  }


  canvas.width =
    Math.max(
      1,
      Math.round(mediaRect.width)
    );


  canvas.height =
    Math.max(
      1,
      Math.round(mediaRect.height)
    );


  canvas.style.left =
    `${mediaRect.left - stageRect.left}px`;


  canvas.style.top =
    `${mediaRect.top - stageRect.top}px`;


  canvas.style.width =
    `${mediaRect.width}px`;


  canvas.style.height =
    `${mediaRect.height}px`;


  drawBox();

}


/* =========================================================
   RESIZE EVENTS
========================================================= */

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


if (
  typeof ResizeObserver !==
  "undefined"
) {

  const observer =
    new ResizeObserver(() => {
      scheduleResize();
    });


  if (stage) {
    observer.observe(stage);
  }

  if (imagePreview) {
    observer.observe(imagePreview);
  }

  if (videoPreview) {
    observer.observe(videoPreview);
  }

}


/* =========================================================
   BALL MARKING
========================================================= */

markBallBtn?.addEventListener(
  "click",
  () => {

    if (!file) {

      setStatus(
        "Choose an image or video first.",
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


    marking = true;
    start = null;

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


    resizeCanvas();


    setBallStatus(
      "drawing",
      "Drag a rectangle around the ball…"
    );

  }
);


/* =========================================================
   CLEAR BALL
========================================================= */

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


/* =========================================================
   CANCEL MARKING
========================================================= */

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


/* =========================================================
   BALL STATUS
========================================================= */

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


/* =========================================================
   POINTER → CANVAS
========================================================= */

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
      (event.clientX - rect.left) *
        scaleX,
      0,
      canvas.width
    ),

    y: clamp(
      (event.clientY - rect.top) *
        scaleY,
      0,
      canvas.height
    )

  };

}


/* =========================================================
   POINTER DOWN
========================================================= */

canvas?.addEventListener(
  "pointerdown",
  (event) => {

    if (!marking) {
      return;
    }


    event.preventDefault();

    start = pointerPos(event);


    try {

      canvas.setPointerCapture(
        event.pointerId
      );

    } catch (_) {}

  }
);


/* =========================================================
   POINTER MOVE
========================================================= */

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


/* =========================================================
   POINTER UP
========================================================= */

canvas?.addEventListener(
  "pointerup",
  (event) => {

    if (!start) {
      return;
    }


    event.preventDefault();


    const point =
      pointerPos(event);


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

        ballBox = [

          (x1 / canvas.width) *
            natural.width,

          (y1 / canvas.height) *
            natural.height,

          (x2 / canvas.width) *
            natural.width,

          (y2 / canvas.height) *
            natural.height

        ].map((value) =>
          Math.round(value)
        );


        setBallStatus(
          "manual",
          `Manual ball box: ${ballBox.join(", ")}`
        );


        cancelMarking();

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


/* =========================================================
   POINTER CANCEL
========================================================= */

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


/* =========================================================
   DRAW BALL BOX
========================================================= */

function drawBox(temp = null) {

  if (!ctx) {
    return;
  }


  ctx.clearRect(
    0,
    0,
    canvas.width,
    canvas.height
  );


  let box = temp;


  const isTemp =
    temp !== null;


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

      (ballBox[0] / natural.width) *
        canvas.width,

      (ballBox[1] / natural.height) *
        canvas.height,

      (ballBox[2] / natural.width) *
        canvas.width,

      (ballBox[3] / natural.height) *
        canvas.height

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


  x1 = clamp(
    Math.min(x1, x2),
    0,
    canvas.width
  );


  y1 = clamp(
    Math.min(y1, y2),
    0,
    canvas.height
  );


  x2 = clamp(
    Math.max(x1, x2),
    0,
    canvas.width
  );


  y2 = clamp(
    Math.max(y1, y2),
    0,
    canvas.height
  );


  ctx.save();


  ctx.strokeStyle =
    isTemp
      ? "#f97316"
      : "#ffffff";


  ctx.lineWidth =
    isTemp ? 2 : 3;


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


  ctx.fillStyle =
    isTemp
      ? "rgba(249,115,22,.18)"
      : "rgba(255,255,255,.14)";


  ctx.fillRect(
    x1,
    y1,
    x2 - x1,
    y2 - y1
  );


  if (!isTemp) {

    const size = 8;

    ctx.fillStyle =
      "#f97316";


    [
      [x1, y1],
      [x2, y1],
      [x1, y2],
      [x2, y2]
    ].forEach(
      ([cx, cy]) => {

        ctx.fillRect(
          cx - size / 2,
          cy - size / 2,
          size,
          size
        );

      }
    );

  }


  ctx.restore();

}


/* =========================================================
   CLEAR CANVAS
========================================================= */

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


/* =========================================================
   ANALYSIS
========================================================= */

analyzeBtn?.addEventListener(
  "click",
  async () => {

    if (!file) {

      setStatus(
        "Choose an image or video first.",
        true
      );

      return;
    }


    const originalText =
      analyzeBtn.textContent;


    analyzeBtn.disabled = true;

    analyzeBtn.textContent =
      "Running analysis…";


    setStatus(
      "Uploading and running DINO + Roboflow…"
    );


    results?.classList.add(
      "hidden"
    );


    const form =
      new FormData();


    form.append(
      "file",
      file,
      file.name
    );


    /*
     * Manual ball selection
     */

    if (ballBox) {

      form.append(
        "ball_bbox",
        ballBox.join(",")
      );

    }


    /*
     * Video detection interval
     */

    const detectEvery =
      Math.max(
        1,
        Number.parseInt(
          $("detectEvery")?.value,
          10
        ) || 1
      );


    /*
     * Video calibration interval
     */

    const calibrationEvery =
      Math.max(
        1,
        Number.parseInt(
          $("calibrationEvery")?.value,
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

      const response =
        await fetch(
          "/api/analyze",
          {
            method: "POST",
            body: form
          }
        );


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


      if (!response.ok) {

        throw new Error(
          data.detail ||
          data.error ||
          `Analysis failed (${response.status})`
        );

      }


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

      analyzeBtn.disabled = false;

      analyzeBtn.textContent =
        originalText;

    }

  }
);


/* =========================================================
   STATUS
========================================================= */

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


/* =========================================================
   RESULTS
========================================================= */

function renderResults(data) {

  setStatus(
    "Analysis complete."
  );


  results?.classList.remove(
    "hidden"
  );


  /*
   * Video responses contain the final
   * analysis inside final_analysis.
   */

  const analysis =
    data.mode === "video"
      ? data.final_analysis || {}
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


  const playerCount =
    detections.filter(
      (item) =>
        item?.class === "player"
    ).length;


  const ballDetected =
    Boolean(
      data.ball_input
        ?.final_ball_available
      ??
      analysis.ball?.detected
    );


  /* =======================================================
     METRICS
  ======================================================= */

  if ($("metrics")) {

    const metrics = [

      [
        "Mode",
        data.mode || "image"
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


  /* =======================================================
     WARNINGS
  ======================================================= */

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
          (warning) => `

            <div class="warning">
              ${escapeHtml(warning)}
            </div>

          `
        )
        .join("");

  }


  /* =======================================================
     COLLECT RESULT IMAGES
  ======================================================= */

  const files =
    analysis.files || {};


  const images = [];


  function walk(value) {

    if (
      typeof value === "string"
    ) {

      /*
       * Backend can return image paths
       * with PNG/JPG/JPEG/WEBP.
       */

      if (
        /\.(png|jpg|jpeg|webp)$/i.test(
          value
        )
      ) {

        images.push(value);

      }


      return;
    }


    if (
      value &&
      typeof value === "object"
    ) {

      Object.values(value)
        .forEach(walk);

    }

  }


  walk(files);


  /*
   * Remove duplicate paths
   * and limit output count.
   */

  resultImages =
    [
      ...new Set(images)
    ].slice(0, 20);


  currentResultIndex = 0;


  renderResultCarousel();


  /* =======================================================
     RAW JSON
  ======================================================= */

  if ($("json")) {

    $("json").textContent =
      JSON.stringify(
        data,
        null,
        2
      );

  }


  /* =======================================================
     SCROLL TO RESULTS
  ======================================================= */

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


/* =========================================================
   RESULT CAROUSEL
========================================================= */

function renderResultCarousel() {

  const visuals =
    $("visuals");


  if (!visuals) {
    return;
  }


  /*
   * No output images
   */

  if (!resultImages.length) {

    visuals.innerHTML = `

      <div class="visual-empty">
        No visual analysis outputs were returned.
      </div>

    `;


    updateCarouselControls();
    updateOutputCounter();

    return;
  }


  /*
   * Keep index valid
   */

  if (
    currentResultIndex < 0 ||
    currentResultIndex >=
      resultImages.length
  ) {

    currentResultIndex = 0;

  }


  /*
   * Resolve image URL
   */

  const currentUrl =
    resolveUrl(
      resultImages[
        currentResultIndex
      ]
    );


  /*
   * Generate title
   */

  const title =
    getResultTitle(
      resultImages[
        currentResultIndex
      ],
      currentResultIndex
    );


  /*
   * Render one output
   */

  visuals.innerHTML = `

    <div class="visual-carousel">

      <!-- PREVIOUS -->

      <button
        id="visualPrev"
        class="visual-nav visual-prev"
        type="button"
        aria-label="Previous analysis result"
      >
        ←
      </button>


      <!-- CURRENT OUTPUT -->

      <div class="visual-frame">

        <img
          id="visualImage"
          src="${escapeAttribute(currentUrl)}"
          alt="${escapeAttribute(title)}"
        />


        <div class="visual-overlay">

          <span
            id="visualTitle"
            class="visual-title"
          >
            ${escapeHtml(title)}
          </span>


          <span
            class="visual-counter"
          >
            ${formatCounter(
              currentResultIndex + 1,
              resultImages.length
            )}
          </span>

        </div>

      </div>


      <!-- NEXT -->

      <button
        id="visualNext"
        class="visual-nav visual-next"
        type="button"
        aria-label="Next analysis result"
      >
        →
      </button>

    </div>

  `;


  /*
   * Attach button events
   */

  $("visualPrev")?.addEventListener(
    "click",
    showPreviousResult
  );


  $("visualNext")?.addEventListener(
    "click",
    showNextResult
  );


  /*
   * Update controls
   */

  updateCarouselControls();
  updateOutputCounter();

}


/* =========================================================
   PREVIOUS OUTPUT
========================================================= */

function showPreviousResult() {

  if (
    resultImages.length <= 1
  ) {
    return;
  }


  currentResultIndex =
    (
      currentResultIndex -
      1 +
      resultImages.length
    ) %
    resultImages.length;


  renderResultCarousel();

}


/* =========================================================
   NEXT OUTPUT
========================================================= */

function showNextResult() {

  if (
    resultImages.length <= 1
  ) {
    return;
  }


  currentResultIndex =
    (
      currentResultIndex +
      1
    ) %
    resultImages.length;


  renderResultCarousel();

}


/* =========================================================
   CAROUSEL BUTTON STATE
========================================================= */

function updateCarouselControls() {

  const previousButton =
    $("visualPrev");

  const nextButton =
    $("visualNext");


  const disabled =
    resultImages.length <= 1;


  if (previousButton) {

    previousButton.disabled =
      disabled;

  }


  if (nextButton) {

    nextButton.disabled =
      disabled;

  }

}


/* =========================================================
   OUTPUT COUNTER
========================================================= */

function updateOutputCounter() {

  /*
   * Counter inside the currently
   * displayed output.
   */

  const counter =
    document.querySelector(
      ".visual-frame .visual-counter"
    );


  if (!counter) {
    return;
  }


  if (!resultImages.length) {

    counter.textContent =
      "00 / 00";

    return;
  }


  counter.textContent =
    formatCounter(
      currentResultIndex + 1,
      resultImages.length
    );

}


/* =========================================================
   KEYBOARD NAVIGATION
========================================================= */

document.addEventListener(
  "keydown",
  (event) => {

    /*
     * Don't capture arrow keys when
     * typing into form elements.
     */

    const tag =
      document.activeElement?.tagName;


    if (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT"
    ) {
      return;
    }


    /*
     * Don't navigate if results
     * are not visible.
     */

    if (
      results?.classList.contains(
        "hidden"
      )
    ) {
      return;
    }


    if (
      event.key === "ArrowLeft"
    ) {

      event.preventDefault();

      showPreviousResult();

    }


    if (
      event.key === "ArrowRight"
    ) {

      event.preventDefault();

      showNextResult();

    }

  }
);


/* =========================================================
   RESULT TITLES
========================================================= */

function getResultTitle(
  url,
  index
) {

  const raw =
    String(url || "")
      .split("/")
      .pop()
      .replace(
        /\.[^.]+$/,
        ""
      );


  if (!raw) {

    return (
      `Analysis result ${index + 1}`
    );

  }


  return raw
    .replace(
      /[\_-]+/g,
      " "
    )
    .replace(
      /\b\w/g,
      (char) =>
        char.toUpperCase()
    );

}


/* =========================================================
   COUNTER FORMAT
========================================================= */

function formatCounter(
  current,
  total
) {

  return (
    String(current).padStart(2, "0") +
    " / " +
    String(total).padStart(2, "0")
  );

}


/* =========================================================
   URL RESOLUTION
========================================================= */

function resolveUrl(url) {

  if (!url) {
    return "";
  }


  const value =
    String(url).trim();


  /*
   * Absolute external URL
   */

  if (
    /^https?:\/\//i.test(value)
  ) {

    return value;

  }


  /*
   * Absolute local URL
   */

  if (
    value.startsWith("/")
  ) {

    return value;

  }


  /*
   * Backend relative path
   */

  return `/${value.replace(
    /^\/+/,
    ""
  )}`;

}


/* =========================================================
   HELPERS
========================================================= */

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


/* =========================================================
   FORMAT FILE SIZE
========================================================= */

function formatBytes(bytes) {

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


  return `${(
    bytes /
    Math.pow(
      1024,
      index
    )
  ).toFixed(
    index === 0
      ? 0
      : 1
  )} ${units[index]}`;

}


/* =========================================================
   ESCAPE HTML
========================================================= */

function escapeHtml(value) {

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


/* =========================================================
   ESCAPE ATTRIBUTE
========================================================= */

function escapeAttribute(value) {

  return escapeHtml(value);

}