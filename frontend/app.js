// controller UI: memuat atau meminta RunResult JSON dari python, mengatur timeline, dan menyinkronkan
// deck, chart, serta panel. semua angka berasal dari python; file ini tidak menjalankan algoritma apa pun.
// mode api (python -m src.server): tombol Run aktif. mode statis (http.server): hanya memuat hasil tersimpan.
(function () {
  "use strict";
  const RM = window.RunModel;
  const $ = (id) => document.getElementById(id);

  // dipakai hanya di mode statis (static server dari folder work/); mode api mengambil daftar dari /api/results
  const STATIC_PRESETS = [
    { algorithm: "hc", label: "HC seed 3001 (eksperimen, instance standar)", url: "../output/hill_climbing/runs/hc_seed3001.json" },
    { algorithm: "hc", label: "HC seed 3002 (eksperimen, instance standar)", url: "../output/hill_climbing/runs/hc_seed3002.json" },
    { algorithm: "hc", label: "HC seed 3003 (eksperimen, instance standar)", url: "../output/hill_climbing/runs/hc_seed3003.json" },
    { algorithm: "sa", label: "SA seed 4001 (eksperimen, instance standar)", url: "../output/simulated_annealing/runs/sa_seed4001.json" },
    { algorithm: "sa", label: "SA seed 4002 (eksperimen, instance standar)", url: "../output/simulated_annealing/runs/sa_seed4002.json" },
    { algorithm: "sa", label: "SA seed 4003 (eksperimen, instance standar)", url: "../output/simulated_annealing/runs/sa_seed4003.json" },
    { algorithm: "ga", label: "GA A pop50 gen50 seed 1001", url: "../output/genetic_algorithm/runs/A_pop50_gen50_seed1001.json" },
    { algorithm: "ga", label: "GA A pop50 gen300 seed 1007", url: "../output/genetic_algorithm/runs/A_pop50_gen300_seed1007.json" },
    { algorithm: "ga", label: "GA B pop100 gen150 seed 2007", url: "../output/genetic_algorithm/runs/B_pop100_gen150_seed2007.json" },
  ];

  const HOWTO = {
    hc: {
      summary: "Satu state current. Tiap iterasi mencoba SATU successor acak dan hanya pindah jika objective-nya lebih tinggi.",
      steps: [
        "Mulai dari state awal random yang feasible.",
        "Tiap iterasi: ambil satu successor random yang feasible (swap / relocate / rotate).",
        "Pindah hanya jika objective candidate > current (strict). Equal dan lebih buruk ditolak.",
        "Berhenti setelah nmax iterasi. Hasil bisa tertahan di local optimum atau plateau (slide 29).",
      ],
    },
    sa: {
      summary: "Seperti stochastic HC, tetapi move yang lebih buruk boleh diterima dengan peluang e^(ΔE/T) yang mengecil saat suhu turun.",
      steps: [
        "Mulai dari state awal random yang feasible; T = T0.",
        "Tiap iterasi: ambil satu successor random feasible, hitung ΔE = candidate − current.",
        "ΔE > 0 diterima; ΔE = 0 diterima (p = 1); ΔE < 0 diterima jika random() < e^(ΔE/T).",
        "Suhu turun: T = T0 · alpha^(t−1). Berhenti saat T < minT atau mencapai batas iterasi.",
        "Local optimum dicek dari SEMUA neighbor feasible; 'stuck' dihitung per episode masuk.",
      ],
    },
    ga: {
      summary: "Menjaga populasi state. Tiap generasi: roulette-wheel selection, one-point crossover, mutasi, lalu repair.",
      steps: [
        "Populasi awal: k state random feasible; fitness = total ShippingFee.",
        "Pilih dua parent dengan roulette wheel (peluang sebanding fitness).",
        "One-point crossover menghasilkan satu child, lalu mutasi (swap/relocate/rotate) dengan peluang per child.",
        "Repair membuat child feasible sebelum masuk populasi; elite terbaik disalin tanpa mutasi.",
        "Deck menampilkan individu terbaik per generasi (bukan 'current state').",
      ],
    },
  };

  const ui = {
    run: null,
    frames: [],
    charts: [],
    frame: 0,
    view: "frame",
    selectedId: null,
    timer: null,
    algorithm: "hc",
    api: false, // true bila server python tersedia
    specs: null, // spesifikasi parameter dari /api/algorithms
    instances: [],
    results: [],
    history: [], // semua hasil yang dimuat/dijalankan di sesi ini, untuk perbandingan
    activeHistory: -1,
  };

  // --- status dan pesan ---------------------------------------------------------

  function setStatus(text, kind) {
    const s = $("status");
    s.textContent = text;
    s.className = `badge badge-${kind}`;
  }

  function showMessage(text, kind) {
    const m = $("message");
    m.hidden = !text;
    m.textContent = text || "";
    m.className = `message ${kind || "info"}`;
  }

  // --- memuat hasil ------------------------------------------------------------

  function presets() {
    if (!ui.api) return STATIC_PRESETS;
    return ui.results.map((r) => ({ algorithm: r.algorithm, label: r.label, url: `/api/results/${r.id}` }));
  }

  function fillPresets() {
    const select = $("preset-select");
    while (select.firstChild) select.removeChild(select.firstChild);
    for (const preset of presets().filter((p) => p.algorithm === ui.algorithm)) {
      const opt = document.createElement("option");
      opt.value = preset.url;
      opt.textContent = preset.label;
      select.appendChild(opt);
    }
  }

  async function loadFromUrl(url) {
    setStatus("memuat…", "running");
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status} saat memuat ${url}`);
      setRun(await resp.json(), url);
    } catch (err) {
      failLoad(err);
    }
  }

  function loadFromFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        setRun(JSON.parse(reader.result), file.name);
      } catch (err) {
        failLoad(err);
      }
    };
    reader.onerror = () => failLoad(new Error("file tidak bisa dibaca"));
    reader.readAsText(file);
  }

  function failLoad(err) {
    setStatus("error", "err");
    showMessage(`Gagal memuat hasil: ${err.message}. Tampilan sebelumnya tidak diubah.`, "error");
  }

  function setRun(data, source, fromHistory = -1) {
    RM.validateRun(data); // lempar error bila bentuk JSON tidak sesuai
    stop();
    if (fromHistory < 0) {
      ui.history.push({ run: data, source });
      ui.activeHistory = ui.history.length - 1;
    } else {
      ui.activeHistory = fromHistory;
    }
    ui.run = data;
    ui.frames = RM.buildFrames(data);
    ui.frame = 0;
    ui.view = "frame";
    ui.selectedId = null;
    selectAlgorithm(data.algorithm, false);
    const p = data.problem;
    $("instance-info").textContent =
      `${p.name || "problem"} | ${p.vehicles.length} kendaraan, deck ${p.ship.width}×${p.ship.length}, ` +
      `MaxCapacity ${p.ship.max_capacity} | sumber: ${(p.metadata && p.metadata.source) || "-"} | file: ${source}`;
    const slider = $("frame-slider");
    slider.max = String(ui.frames.length - 1);
    $("trace-badge").hidden = data.config.record_snapshots !== false;
    $("trace-badge").textContent = "snapshot tidak lengkap: frame tengah tanpa state";
    document.querySelector('[data-view="best"]').hidden = data.algorithm !== "sa";
    renderSummary();
    renderCharts();
    renderHistory();
    setStatus("hasil dimuat", "ok");
    showMessage(data.final_state.feasible ? "" : "PERINGATAN: final state ditandai TIDAK feasible oleh backend.",
      data.final_state.feasible ? "info" : "error");
    render();
  }

  // --- algoritma ------------------------------------------------------------------

  function selectAlgorithm(algo, refillPresets = true) {
    ui.algorithm = algo;
    for (const tab of document.querySelectorAll("[data-algo]")) {
      tab.setAttribute("aria-selected", String(tab.dataset.algo === algo));
    }
    $("algo-summary").textContent = HOWTO[algo].summary;
    const list = $("howto");
    while (list.firstChild) list.removeChild(list.firstChild);
    for (const step of HOWTO[algo].steps) {
      const li = document.createElement("li");
      li.textContent = step;
      list.appendChild(li);
    }
    if (refillPresets) fillPresets();
    buildParamFields();
  }

  // --- run lewat api ----------------------------------------------------------------

  // form parameter dibangun dari spesifikasi server, jadi batas dan default hanya didefinisikan di python
  function buildParamFields() {
    const box = $("param-fields");
    while (box.firstChild) box.removeChild(box.firstChild);
    if (!ui.specs) return;
    for (const spec of ui.specs[ui.algorithm]) {
      const id = `param-${spec.key}`;
      const label = document.createElement("label");
      label.htmlFor = id;
      const input = document.createElement("input");
      input.id = id;
      input.dataset.key = spec.key;
      input.dataset.type = spec.type;
      if (spec.type === "bool") {
        input.type = "checkbox";
        input.checked = spec.default;
        label.className = "check";
        label.append(input, document.createTextNode(spec.label));
        box.appendChild(label);
        continue;
      }
      input.type = "number";
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.step = spec.type === "int" ? "1" : "any";
      input.value = String(spec.default);
      input.required = true;
      label.textContent = `${spec.label} [${spec.min} … ${spec.max}]`;
      box.append(label, input);
    }
  }

  function renderInstanceNote() {
    const info = ui.instances.find((i) => i.id === $("instance-select").value);
    if (!info) return;
    const opt = info.exact_optimum === null ? "optimum tidak diketahui (best-known saja)" :
      `optimum exact = ${info.exact_optimum} (dibuktikan brute force)`;
    $("instance-note").textContent = `${info.vehicles} kendaraan, deck ${info.ship.width}×${info.ship.length}, ` +
      `MaxCapacity ${info.ship.max_capacity}. Sumber: ${info.source}. ${opt}.`;
  }

  async function runSolver(event) {
    event.preventDefault();
    const config = {};
    for (const input of $("param-fields").querySelectorAll("input")) {
      const type = input.dataset.type;
      config[input.dataset.key] = type === "bool" ? input.checked : type === "int" ? Number.parseInt(input.value, 10) :
        Number.parseFloat(input.value);
    }
    const body = { algorithm: ui.algorithm, instance: $("instance-select").value,
      seed: Number.parseInt($("seed-input").value, 10), config };
    const button = $("run-btn");
    button.disabled = true;
    const started = performance.now();
    setStatus(`menjalankan ${ui.algorithm.toUpperCase()} di Python…`, "running");
    showMessage("Solver sedang berjalan di Python. Hasil baru tampil setelah response diterima.", "info");
    try {
      const resp = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) });
      const data = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }));
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      const wall = ((performance.now() - started) / 1000).toFixed(1);
      setRun(data, `run langsung (${body.instance}, seed ${body.seed}, ${wall} s termasuk transfer)`);
    } catch (err) {
      setStatus("error", "err");
      showMessage(`Run gagal: ${err.message}`, "error");
    } finally {
      button.disabled = false;
    }
  }

  // --- riwayat / perbandingan -------------------------------------------------------

  function renderHistory() {
    const body = $("history-body");
    while (body.firstChild) body.removeChild(body.firstChild);
    ui.history.forEach((item, k) => {
      const run = item.run;
      const tr = document.createElement("tr");
      if (k === ui.activeHistory) tr.className = "active";
      const cells = [k + 1, RM.ALGORITHMS[run.algorithm].name, run.problem.name, run.seed,
        Object.entries(run.config).filter(([key]) => key !== "record_snapshots").map(([key, v]) => `${key}=${v}`).join(", "),
        run.final_objective, RM.bestOf(run), RM.evaluationsOf(run) ?? "-", run.runtime_seconds.toFixed(3), item.source];
      cells.forEach((value, c) => {
        const td = document.createElement("td");
        td.textContent = String(value);
        if (c === 4) td.className = "param";
        tr.appendChild(td);
      });
      const td = document.createElement("td");
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = "Lihat";
      b.addEventListener("click", () => setRun(run, item.source, k));
      td.appendChild(b);
      tr.appendChild(td);
      body.appendChild(tr);
    });
  }

  // --- render -------------------------------------------------------------------

  function renderSummary() {
    const dl = $("summary");
    while (dl.firstChild) dl.removeChild(dl.firstChild);
    for (const [key, value] of RM.summaryOf(ui.run)) {
      const dt = document.createElement("dt");
      dt.textContent = key;
      const dd = document.createElement("dd");
      dd.textContent = String(value);
      dl.append(dt, dd);
    }
  }

  function renderCharts() {
    const box = $("charts");
    while (box.firstChild) box.removeChild(box.firstChild);
    const axis = RM.ALGORITHMS[ui.run.algorithm].axis;
    const specs = RM.chartsFor(ui.run);
    // buat semua kotak dulu, baru gambar: lebar kolom grid baru final setelah semua kotak ada
    const boxes = specs.map(() => box.appendChild(document.createElement("div")));
    ui.charts = specs.map((spec, k) => Charts.renderChart(boxes[k], spec, axis, setFrame));
  }

  // state yang sedang ditampilkan di deck beserta keterangannya
  function displayed() {
    const run = ui.run;
    const axis = RM.ALGORITHMS[run.algorithm].axis;
    if (ui.view === "initial") {
      return { state: run.initial_state, label: run.algorithm === "ga" ? "Initial: individu terbaik populasi awal" : "Initial: state awal random" };
    }
    if (ui.view === "final") {
      const label = { hc: "Final: current terakhir (= terbaik untuk HC)", sa: "Final: TERMINAL current (state saat SA berhenti)",
        ga: "Final: best-found sepanjang semua generasi" }[run.algorithm];
      return { state: run.final_state, label };
    }
    if (ui.view === "best") {
      return { state: run.metrics.best_found_state, label: `Best-found SA (iterasi ${run.metrics.best_found_iteration})` };
    }
    const f = ui.frames[ui.frame];
    const what = run.algorithm === "ga" ? "individu terbaik generasi ini" : "current state";
    return { state: f.state, label: `Frame ${axis} ${f.index}: ${what}` };
  }

  function render() {
    if (!ui.run) return;
    const run = ui.run;
    const { state, label } = displayed();
    for (const b of document.querySelectorAll("[data-view]")) b.setAttribute("aria-pressed", String(b.dataset.view === ui.view));
    $("view-label").textContent = label;

    if (state) Deck.renderDeck($("deck"), state, run.problem, { selectedId: ui.selectedId, onSelect: selectVehicle });
    else Deck.renderDeckMessage($("deck"), "snapshot tidak tersedia untuk frame ini");
    renderStats(state);
    renderOutside(state);
    renderDetail(state);

    const lines = ui.view === "frame" ? RM.explainFrame(run, ui.frames, ui.frame) : [label + "."];
    const ex = $("explain");
    while (ex.firstChild) ex.removeChild(ex.firstChild);
    for (const line of lines) {
      const para = document.createElement("p");
      para.textContent = line;
      ex.appendChild(para);
    }

    $("frame-slider").value = String(ui.frame);
    const axis = RM.ALGORITHMS[run.algorithm].axis;
    $("frame-label").textContent = `${axis} ${ui.frame} / ${ui.frames.length - 1}`;
    for (const chart of ui.charts) chart.setCursor(ui.frame);
  }

  function stat(label, value, bad) {
    const div = document.createElement("div");
    div.className = "stat" + (bad ? " bad" : "");
    const l = document.createElement("span");
    l.className = "label";
    l.textContent = label;
    const v = document.createElement("span");
    v.className = "value";
    v.textContent = value;
    div.append(l, v);
    return div;
  }

  function renderStats(state) {
    const box = $("stats");
    while (box.firstChild) box.removeChild(box.firstChild);
    if (!state) {
      box.appendChild(stat("State", "tidak tersedia"));
      return;
    }
    const ship = ui.run.problem.ship;
    // utilisasi dan total potensi fee hanya angka tampilan; objective tetap dari python
    const used = state.placements.filter((p) => p.loaded).reduce((sum, p) => sum + p.width * p.length, 0);
    const potential = ui.run.problem.vehicles.reduce((sum, v) => sum + v.shipping_fee, 0);
    box.append(
      stat("Objective (ShippingFee)", String(state.objective)),
      stat("Berat / MaxCapacity", `${state.total_weight} / ${ship.max_capacity}`),
      stat("Kendaraan dimuat", `${state.loaded_count} / ${state.placements.length}`),
      stat("Utilisasi area deck", `${((100 * used) / (ship.width * ship.length)).toFixed(1)}%`),
      stat("Total potensi fee", String(potential)),
      stat("Feasible (backend)", state.feasible ? "ya" : "TIDAK", !state.feasible),
    );
  }

  function renderOutside(state) {
    const list = $("outside-list");
    while (list.firstChild) list.removeChild(list.firstChild);
    const outside = state ? state.placements.filter((p) => !p.loaded) : [];
    $("outside-count").textContent = String(outside.length);
    const vehicles = new Map(ui.run.problem.vehicles.map((v) => [v.id, v]));
    for (const p of outside) {
      const v = vehicles.get(p.id);
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = `${p.id} · fee ${v.shipping_fee} · berat ${v.weight}`;
      b.style.borderLeftColor = RM.vehicleColor(p.id);
      b.addEventListener("click", () => selectVehicle(p.id));
      li.appendChild(b);
      list.appendChild(li);
    }
  }

  function renderDetail(state) {
    const box = $("vehicle-detail");
    while (box.firstChild) box.removeChild(box.firstChild);
    const v = ui.selectedId && ui.run.problem.vehicles.find((x) => x.id === ui.selectedId);
    if (!v || !state) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = "Klik atau fokus (Tab) kendaraan untuk melihat detail.";
      box.appendChild(p);
      return;
    }
    const p = state.placements.find((x) => x.id === v.id);
    const rows = [
      ["ID", v.id],
      ["Ukuran asli (w × l)", `${v.width} × ${v.length}`],
      ["Status", p.loaded ? "di dalam kapal" : "di luar kapal"],
      ["Orientasi", p.loaded ? p.orientation : "-"],
      ["Footprint efektif", p.loaded ? `${p.width} × ${p.length}` : "-"],
      ["Posisi (x, y)", p.loaded ? `(${p.x}, ${p.y})` : "-"],
      ["ShippingFee", v.shipping_fee],
      ["Weight", v.weight],
      ["ETA", v.eta],
    ];
    const dl = document.createElement("dl");
    for (const [k, val] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = String(val);
      dl.append(dt, dd);
    }
    box.appendChild(dl);
  }

  function selectVehicle(id) {
    ui.selectedId = id;
    render();
  }

  // --- timeline -------------------------------------------------------------------

  function setFrame(i) {
    if (!ui.run) return;
    ui.frame = Math.min(ui.frames.length - 1, Math.max(0, i));
    ui.view = "frame";
    render();
  }

  function stop() {
    if (ui.timer) clearInterval(ui.timer);
    ui.timer = null;
    $("play-btn").textContent = "▶ Play";
  }

  function play() {
    if (!ui.run) return;
    if (ui.timer) return stop();
    if (ui.frame >= ui.frames.length - 1) setFrame(0);
    // 1x: satu run penuh kira-kira 20 detik (minimal 4 frame/detik) agar run panjang tetap bisa didemo
    const speed = Number($("speed-select").value);
    const fps = Math.max(4, ui.frames.length / 20) * speed;
    const stepPerTick = Math.max(1, Math.round(fps / 30));
    $("play-btn").textContent = "⏸ Pause";
    ui.timer = setInterval(() => {
      if (ui.frame >= ui.frames.length - 1) return stop(); // otomatis berhenti di frame terakhir
      setFrame(ui.frame + stepPerTick);
    }, Math.max(33, 1000 / fps));
  }

  // --- event ----------------------------------------------------------------------

  function bind() {
    for (const tab of document.querySelectorAll("[data-algo]")) {
      tab.addEventListener("click", () => selectAlgorithm(tab.dataset.algo));
    }
    for (const b of document.querySelectorAll("[data-view]")) {
      b.addEventListener("click", () => { ui.view = b.dataset.view; render(); });
    }
    $("load-preset-btn").addEventListener("click", () => {
      const url = $("preset-select").value;
      if (url) loadFromUrl(url);
    });
    $("file-input").addEventListener("change", (e) => {
      if (e.target.files.length) loadFromFile(e.target.files[0]);
    });
    $("reset-btn").addEventListener("click", reset);
    $("run-form").addEventListener("submit", runSolver);
    $("instance-select").addEventListener("change", renderInstanceNote);
    $("first-btn").addEventListener("click", () => setFrame(0));
    $("prev-btn").addEventListener("click", () => setFrame(ui.frame - 1));
    $("next-btn").addEventListener("click", () => setFrame(ui.frame + 1));
    $("last-btn").addEventListener("click", () => setFrame(ui.frames.length - 1));
    $("play-btn").addEventListener("click", play);
    $("speed-select").addEventListener("change", () => { if (ui.timer) { stop(); play(); } });
    $("frame-slider").addEventListener("input", (e) => setFrame(Number(e.target.value)));
    document.addEventListener("keydown", (e) => {
      const tag = e.target.tagName;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) || !ui.run) return;
      if (e.target.closest && e.target.closest(".vehicle")) return;
      const keys = { ArrowLeft: () => setFrame(ui.frame - 1), ArrowRight: () => setFrame(ui.frame + 1),
        Home: () => setFrame(0), End: () => setFrame(ui.frames.length - 1), " ": play };
      if (keys[e.key] && !(e.key === " " && tag === "BUTTON")) {
        e.preventDefault();
        keys[e.key]();
      }
    });
  }

  function reset() {
    stop();
    selectAlgorithm("hc");
    const first = presets().find((p) => p.algorithm === "hc");
    if (first) loadFromUrl(first.url);
    else showMessage("Belum ada hasil HC tersimpan untuk demo.", "error");
  }

  // deteksi server python; tanpa server, UI tetap bisa memuat hasil tersimpan (mode statis)
  async function init() {
    bind();
    setStatus("menyiapkan…", "running");
    try {
      const health = await fetch("/api/health");
      if (!health.ok) throw new Error("no api");
      const [specs, instances, results] = await Promise.all(
        ["/api/algorithms", "/api/instances", "/api/results"].map((u) => fetch(u).then((r) => r.json())));
      Object.assign(ui, { api: true, specs, instances, results });
      const select = $("instance-select");
      for (const info of instances) {
        const opt = document.createElement("option");
        opt.value = info.id;
        opt.textContent = info.label;
        select.appendChild(opt);
      }
      renderInstanceNote();
      $("run-form").hidden = false;
    } catch (err) {
      ui.api = false;
      $("run-offline").hidden = false;
    }
    selectAlgorithm("hc");
    reset();
  }

  init();
})();
