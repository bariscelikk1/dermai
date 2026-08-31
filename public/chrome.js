/* Shared site chrome: header and footer injected on every page,
   so navigation stays consistent across the seven pages. */

const PAGES = [
  { href: "/index.html",      label: "Overview" },
  { href: "/clinical.html",   label: "Clinical Context" },
  { href: "/method.html",     label: "Method" },
  { href: "/analyse.html",    label: "Analysis" },
  { href: "/results.html",    label: "Results" },
  { href: "/ethics.html",     label: "Ethics" },
  { href: "/references.html", label: "References" },
];

(function () {
  const path = location.pathname === "/" ? "/index.html" : location.pathname;

  const nav = PAGES.map(
    (p) =>
      `<a href="${p.href}"${p.href === path ? ' class="is-current" aria-current="page"' : ""}>${p.label}</a>`
  ).join("");

  document.body.insertAdjacentHTML(
    "afterbegin",
    `<header class="site-header">
      <div class="utility-bar">
        <div class="wrap utility-inner">
          <span class="utility-note">Undergraduate research project · Yıldız Teknik Üniversitesi, Department of Computer Engineering</span>
          <nav class="utility-links">
            <a href="/references.html">Bibliography</a>
            <a href="https://github.com/bariscelikk1/dermai" target="_blank" rel="noopener">Source code</a>
          </nav>
        </div>
      </div>
      <div class="wrap masthead">
        <a class="wordmark" href="/index.html">
          <span class="wordmark-mark" aria-hidden="true">
            <svg viewBox="0 0 40 40" width="40" height="40">
              <circle cx="20" cy="20" r="18" fill="none" stroke="currentColor" stroke-width="1.5"/>
              <circle cx="20" cy="20" r="9" fill="none" stroke="currentColor" stroke-width="1.5"/>
              <circle cx="20" cy="20" r="2.5" fill="currentColor"/>
              <path d="M20 2v6M20 32v6M2 20h6M32 20h6" stroke="currentColor" stroke-width="1.5"/>
            </svg>
          </span>
          <span class="wordmark-text">
            <strong>Derm<span class="wordmark-ai">AI</span></strong>
            <small>Dermoscopic Lesion Classification</small>
          </span>
        </a>
        <nav class="main-nav">${nav}</nav>
      </div>
    </header>`
  );

  document.body.insertAdjacentHTML(
    "beforeend",
    `<footer class="site-footer">
      <div class="wrap footer-inner">
        <div class="footer-about">
          <p class="footer-brand">Derm<span>AI</span></p>
          <p class="footer-note">
            A seven-class dermoscopic lesion classifier developed as an
            undergraduate research project. Not a medical device; not for
            clinical or diagnostic use.
          </p>
          <p class="footer-note footer-author">
            N. Barış Çelik · Department of Computer Engineering,<br>
            Yıldız Teknik Üniversitesi, İstanbul
          </p>
        </div>
        <div class="footer-cols">
          <div>
            <p class="footer-head">Project</p>
            ${PAGES.map((p) => `<a href="${p.href}">${p.label}</a>`).join("")}
          </div>
          <div>
            <p class="footer-head">Specification</p>
            <p><span>Architecture</span> EfficientNet-B0</p>
            <p><span>Initialisation</span> ImageNet-1k</p>
            <p><span>Dataset</span> HAM10000 (n = 10,015)</p>
            <p><span>Protocol</span> LP-FT, BatchNorm frozen</p>
            <p><span>Input</span> 224 × 224 × 3, raw [0, 255]</p>
          </div>
        </div>
      </div>
      <div class="wrap footer-legal">
        <p>
          Epidemiological figures are cited from GLOBOCAN 2022 (IARC/WHO), the
          SEER programme, and T.C. Sağlık Bakanlığı Halk Sağlığı Genel
          Müdürlüğü. Full sources on the
          <a href="/references.html">references page</a>.
        </p>
      </div>
    </footer>`
  );
})();
