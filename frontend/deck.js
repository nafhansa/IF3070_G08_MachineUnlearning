// gambar deck kapal sebagai SVG. satu satuan viewBox = satu sel grid, jadi koordinat state (x, y)
// dan footprint efektif (width, length dari python, sudah memperhitungkan orientasi) dipakai apa adanya.
(function (root) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function clear(svg) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }

  // tampilkan pesan di area deck bila state tidak tersedia; tidak menggambar kendaraan karangan
  function renderDeckMessage(svg, text) {
    clear(svg);
    svg.setAttribute("viewBox", "0 0 20 6");
    const t = el("text", { x: 10, y: 3, "text-anchor": "middle", class: "deck-empty" }, svg);
    t.textContent = text;
  }

  // opts: { selectedId, onSelect(id) }
  function renderDeck(svg, state, problem, opts) {
    clear(svg);
    const W = problem.ship.width;
    const L = problem.ship.length;
    const margin = 1.1; // ruang label sumbu di kiri/atas
    // x ke kanan, y ke bawah: sama dengan konvensi koordinat python (0,0 di kiri atas)
    svg.setAttribute("viewBox", `${-margin} ${-margin} ${W + margin + 0.3} ${L + margin + 0.3}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.setAttribute("aria-label", `Deck ${W} x ${L}, ${state.loaded_count} kendaraan dimuat`);

    el("rect", { x: 0, y: 0, width: W, height: L, class: "deck-border" }, svg);
    for (let x = 1; x < W; x++) el("line", { x1: x, y1: 0, x2: x, y2: L, class: "grid-line" }, svg);
    for (let y = 1; y < L; y++) el("line", { x1: 0, y1: y, x2: W, y2: y, class: "grid-line" }, svg);
    const step = Math.max(W, L) > 30 ? 5 : 1;
    for (let x = 0; x <= W; x += step) {
      el("text", { x, y: -0.35, "text-anchor": "middle", class: "axis-label" }, svg).textContent = String(x);
    }
    for (let y = 0; y <= L; y += step) {
      el("text", { x: -0.3, y: y + 0.12, "text-anchor": "end", class: "axis-label" }, svg).textContent = String(y);
    }

    const vehicles = new Map(problem.vehicles.map((v) => [v.id, v]));
    for (const p of state.placements) {
      if (!p.loaded) continue; // kendaraan di luar kapal tidak boleh tampak menempati grid
      const v = vehicles.get(p.id);
      const g = el("g", {
        class: "vehicle" + (opts.selectedId === p.id ? " selected" : ""),
        tabindex: 0,
        role: "button",
        "aria-label": `${p.id}, posisi (${p.x}, ${p.y}), orientasi ${p.orientation}, fee ${v.shipping_fee}`,
      }, svg);
      el("rect", { x: p.x, y: p.y, width: p.width, height: p.length, fill: RunModel.vehicleColor(p.id) }, g);
      el("title", {}, g).textContent =
        `${p.id} | (${p.x}, ${p.y}) ${p.orientation} | ${p.width}x${p.length} | fee ${v.shipping_fee} | berat ${v.weight}`;
      // label diletakkan di tengah; bila kotak sempit (1 sel) label tetap pendek (id saja)
      const cx = p.x + p.width / 2;
      const cy = p.y + p.length / 2;
      el("text", { x: cx, y: p.length >= 2 ? cy - 0.05 : cy + 0.13, "text-anchor": "middle" }, g).textContent = p.id;
      if (p.length >= 2) el("text", { x: cx, y: cy + 0.4, "text-anchor": "middle" }, g).textContent = p.orientation;
      const pick = () => opts.onSelect(p.id);
      g.addEventListener("click", pick);
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
      });
    }
  }

  root.Deck = { renderDeck, renderDeckMessage };
})(this);
