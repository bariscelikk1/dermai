/* DermAI — upload → scan → assessment flow */

const $ = (id) => document.getElementById(id);

const dropzone   = $("dropzone");
const fileInput  = $("fileInput");
const uploadErr  = $("uploadError");
const scanStep   = $("scanStep");
const assessment = $("assessment");
const referral   = $("referral");
const frame      = document.querySelector(".scan-frame");
const preview    = $("preview");
const caption    = $("scanCaption");
const pipeline   = [...document.querySelectorAll(".pipeline li")];

const BANDS = {
  low: {
    title: "No malignant indicators",
    line: "The distribution is concentrated in benign categories.",
    note: "A low score is not clearance. Benign-looking lesions can still be " +
          "malignant, and this model has never examined your skin in person. " +
          "If a lesion is changing, itching, bleeding, or simply worries you, " +
          "have it looked at regardless of what this page says.",
    referral: false,
  },
  moderate: {
    title: "Indeterminate — review advised",
    line: "Meaningful probability mass sits in malignant categories.",
    note: "The model is not confident either way. In a clinical setting this " +
          "is exactly the case that gets escalated to a specialist rather " +
          "than dismissed. Arrange an in-person examination.",
    referral: true,
  },
  high: {
    title: "Malignant features detected",
    line: "The distribution is dominated by malignant categories.",
    note: "Treat this as a prompt to book an appointment, not as a diagnosis — " +
          "only a dermatologist, usually with dermoscopy and if needed a " +
          "biopsy, can determine what this lesion actually is. Early-detected " +
          "melanoma is highly treatable; delay is what causes harm.",
    referral: true,
  },
};

const pct = (v) => (v * 100).toFixed(1) + "%";
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/* Downscale before upload. The model only ever sees 224x224, so sending a
   12-megapixel phone photo wastes bandwidth — and some hosts cap request
   bodies well below the size of a modern camera file. EXIF orientation is
   handled server-side, so this only touches dimensions. */
const MAX_EDGE = 1024;
/* The server refuses anything over 1 MB, because a larger upload would be
   spooled to a temporary file and the privacy note promises it is not. A
   small-but-heavy image (an uncompressed PNG, say) needs re-encoding even
   when its dimensions are already fine. */
const MAX_BYTES = 1024 * 1024;

function downscale(file) {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      const { width: w, height: h } = img;
      const scale = Math.min(1, MAX_EDGE / Math.max(w, h));

      // Already small enough, or an image type we would rather not re-encode.
      if (scale === 1 && file.size <= MAX_BYTES) {
        URL.revokeObjectURL(url);
        return resolve(file);
      }

      const canvas = document.createElement("canvas");
      canvas.width = Math.round(w * scale) || w;
      canvas.height = Math.round(h * scale) || h;
      const ctx = canvas.getContext("2d");
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);

      canvas.toBlob(
        (blob) => resolve(blob && blob.size < file.size ? blob : file),
        "image/jpeg",
        0.85
      );
    };

    // If decoding fails, fall back to the original and let the server judge.
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(file);
    };
    img.src = url;
  });
}

/* ── input handling ─────────────────────────────────────── */

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

/* ── main flow ──────────────────────────────────────────── */

async function handleFile(file) {
  uploadErr.hidden = true;

  if (!file.type.startsWith("image/")) {
    return fail("That file is not an image. Please upload a JPEG or PNG.");
  }
  if (file.size > 20 * 1024 * 1024) {
    return fail("That image is larger than 20 MB. Please upload a smaller file.");
  }

  preview.src = URL.createObjectURL(file);
  assessment.hidden = true;
  referral.hidden = true;
  scanStep.hidden = false;
  scanStep.scrollIntoView({ behavior: "smooth", block: "start" });

  pipeline.forEach((li) => li.classList.remove("active", "done"));
  frame.classList.add("scanning");
  caption.textContent = "Analysing…";

  const upload = await downscale(file);

  const body = new FormData();
  body.append("file", upload, "lesion.jpg");

  // Run the real request alongside the staged pipeline display.
  const request = fetch("/api/predict", { method: "POST", body })
    .then(async (r) => {
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || `Server returned ${r.status}.`);
      }
      return r.json();
    });

  const staged = runPipeline();

  try {
    const [result] = await Promise.all([request, staged]);
    frame.classList.remove("scanning");
    caption.textContent = "Analysis complete";
    render(result);
  } catch (err) {
    frame.classList.remove("scanning");
    caption.textContent = "Analysis failed";
    scanStep.hidden = true;
    fail(err.message || "The analysis could not be completed.");
  }
}

async function runPipeline() {
  const timings = [420, 380, 300, 700, 320, 300];
  for (let i = 0; i < pipeline.length; i++) {
    pipeline[i].classList.add("active");
    await wait(timings[i]);
    pipeline[i].classList.remove("active");
    pipeline[i].classList.add("done");
  }
}

function fail(msg) {
  uploadErr.textContent = msg;
  uploadErr.hidden = false;
  uploadErr.scrollIntoView({ behavior: "smooth", block: "center" });
}

/* ── rendering ──────────────────────────────────────────── */

function render(r) {
  const band = BANDS[r.band];
  const box = $("verdict");

  box.classList.remove("low", "moderate", "high");
  box.classList.add(r.band);

  $("verdictBand").textContent = band.title;
  $("verdictLine").textContent = band.line;
  $("verdictNote").textContent = band.note;

  $("mTop").textContent = r.top.name;
  $("mMel").textContent = pct(r.melanoma_probability);
  $("mMal").textContent = pct(r.malignant_score);

  const body = $("distBody");
  body.innerHTML = "";
  r.classes.forEach((c) => {
    const tr = document.createElement("tr");

    const c1 = document.createElement("td");
    c1.innerHTML =
      `<span class="cell-class"></span><span class="cell-code"></span>` +
      `<span class="row-note"></span>`;
    c1.querySelector(".cell-class").textContent = c.name;
    c1.querySelector(".cell-code").textContent = c.code;
    c1.querySelector(".row-note").textContent = c.note;

    const c2 = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = "pill " + (c.malignant ? "pill-mal" : "pill-ben");
    pill.textContent = c.malignant ? "Malignant" : "Benign";
    c2.appendChild(pill);

    const c3 = document.createElement("td");
    c3.className = "prob-cell";
    c3.innerHTML =
      `<div class="prob-row"><div class="prob-track">` +
      `<div class="prob-fill"></div></div><span class="prob-num"></span></div>`;
    c3.querySelector(".prob-num").textContent = pct(c.probability);
    const fill = c3.querySelector(".prob-fill");
    fill.classList.add(c.malignant ? "mal" : "ben");
    requestAnimationFrame(() => {
      fill.style.width = Math.max(c.probability * 100, 1.5) + "%";
    });

    tr.append(c1, c2, c3);
    body.appendChild(tr);
  });

  assessment.hidden = false;
  referral.hidden = !band.referral;
  assessment.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ── geolocation → map links ────────────────────────────── */

$("locBtn").addEventListener("click", () => {
  const status = $("locStatus");

  if (!navigator.geolocation) {
    status.textContent = "This browser does not expose location. Use the general searches below.";
    showMaps(null);
    return;
  }

  status.textContent = "Requesting location…";
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const { latitude: lat, longitude: lon } = pos.coords;
      status.textContent =
        `Location resolved to ${lat.toFixed(3)}, ${lon.toFixed(3)}. ` +
        `These links open in your map app — your coordinates were never sent to DermAI.`;
      showMaps({ lat, lon });
    },
    () => {
      status.textContent =
        "Location permission was declined. The links below fall back to a general search.";
      showMaps(null);
    },
    { timeout: 10000 }
  );
});

function showMaps(coords) {
  const g = (q) =>
    coords
      ? `https://www.google.com/maps/search/${encodeURIComponent(q)}/@${coords.lat},${coords.lon},13z`
      : `https://www.google.com/maps/search/${encodeURIComponent(q + " near me")}`;

  $("gmaps").href = g("dermatologist");
  $("ghosp").href = g("hospital");
  $("amaps").href = coords
    ? `https://maps.apple.com/?q=dermatologist&sll=${coords.lat},${coords.lon}`
    : `https://maps.apple.com/?q=dermatologist`;

  $("mapLinks").hidden = false;
}

/* ── reset ──────────────────────────────────────────────── */

$("resetBtn").addEventListener("click", () => {
  fileInput.value = "";
  scanStep.hidden = true;
  assessment.hidden = true;
  referral.hidden = true;
  $("mapLinks").hidden = true;
  document.getElementById("analysis").scrollIntoView({ behavior: "smooth", block: "start" });
});
