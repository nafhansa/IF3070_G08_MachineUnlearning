// chart garis/titik sederhana dengan SVG native. seri digambar sekali per run;
// saat frame berpindah hanya garis cursor yang digeser (agar play tetap ringan).
(function (root) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const VIEW = { w: 640, h: 190, left: 54, right: 12, top: 10, bottom: 26 };

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  const fmt = (v) => (Math.abs(v) >= 100 || Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(2));

  // spec: { title, yLabel, yMin?, yMax?, series: [{ name, values, kind: "line"|"dot", color, dashed? }] }
  // box sudah ada di DOM (dibuat lebih dulu oleh pemanggil) agar lebarnya sudah final saat diukur.
  // return { setCursor(index) }
  function renderChart(box, spec, axisName, onPick) {
    box.className = "chart";
    const title = document.createElement("h3");
    title.textContent = spec.title;
    box.appendChild(title);
    // lebar viewBox = lebar container (piksel) supaya teks sumbu tidak tergencet
    const view = { ...VIEW, w: Math.max(320, Math.round(box.clientWidth - 14)) };
    const svg = el("svg", { viewBox: `0 0 ${view.w} ${view.h}`, role: "img", "aria-label": spec.title }, box);

    const n = Math.max(...spec.series.map((s) => s.values.length));
    const all = spec.series.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined);
    // skala y dari data asli (atau batas yang ditetapkan spec), tanpa memotong supaya tidak menyesatkan
    let lo = spec.yMin !== undefined ? spec.yMin : Math.min(...all);
    let hi = spec.yMax !== undefined ? spec.yMax : Math.max(...all);
    if (!all.length) { lo = 0; hi = 1; }
    if (hi === lo) { hi = lo + 1; }
    const plotW = view.w - view.left - view.right;
    const plotH = view.h - view.top - view.bottom;
    const sx = (i) => view.left + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
    const sy = (v) => view.top + (1 - (v - lo) / (hi - lo)) * plotH;

    el("line", { x1: view.left, y1: view.top, x2: view.left, y2: view.top + plotH, class: "axis" }, svg);
    el("line", { x1: view.left, y1: view.top + plotH, x2: view.left + plotW, y2: view.top + plotH, class: "axis" }, svg);
    for (const v of [lo, (lo + hi) / 2, hi]) {
      el("text", { x: view.left - 6, y: sy(v) + 4, "text-anchor": "end", class: "tick" }, svg).textContent = fmt(v);
    }
    el("text", { x: view.left, y: view.h - 6, class: "tick" }, svg).textContent = `${axisName} 0`;
    el("text", { x: view.left + plotW, y: view.h - 6, "text-anchor": "end", class: "tick" }, svg).textContent =
      `${axisName} ${n - 1}`;

    for (const s of spec.series) {
      if (s.kind === "dot") {
        s.values.forEach((v, i) => {
          if (v === null || v === undefined) return;
          const c = el("circle", { cx: sx(i), cy: sy(v), r: 2.2 }, svg);
          c.style.fill = s.color;
        });
        continue;
      }
      // garis diputus pada nilai null (mis. temperature di iterasi 0)
      let d = "";
      let pen = false;
      s.values.forEach((v, i) => {
        if (v === null || v === undefined) { pen = false; return; }
        d += `${pen ? "L" : "M"}${sx(i).toFixed(2)},${sy(v).toFixed(2)}`;
        pen = true;
      });
      const path = el("path", { d, fill: "none", "stroke-width": 1.6, "vector-effect": "non-scaling-stroke" }, svg);
      path.style.stroke = s.color;
      if (s.dashed) path.setAttribute("stroke-dasharray", "5 3");
    }

    const cursor = el("line", { y1: view.top, y2: view.top + plotH, class: "cursor",
      "vector-effect": "non-scaling-stroke" }, svg);
    svg.addEventListener("click", (e) => {
      const rect = svg.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * view.w;
      const i = Math.round(((x - view.left) / plotW) * (n - 1));
      onPick(Math.min(n - 1, Math.max(0, i)));
    });

    const legend = document.createElement("div");
    legend.className = "legend";
    for (const s of spec.series) {
      const item = document.createElement("span");
      item.textContent = s.name;
      item.style.setProperty("--swatch", s.color);
      if (s.dashed) item.classList.add("dashed");
      if (s.kind === "dot") item.classList.add("dot");
      legend.appendChild(item);
    }
    box.appendChild(legend);

    return {
      setCursor(i) {
        cursor.setAttribute("x1", sx(i));
        cursor.setAttribute("x2", sx(i));
      },
    };
  }

  root.Charts = { renderChart };
})(this);
