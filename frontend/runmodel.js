// adapter RunResult (JSON dari python) -> frame timeline, seri chart, dan teks penjelasan.
// file ini sengaja tanpa DOM agar bisa dites dengan node; tidak ada logika algoritma di sini.
// semua angka (objective, berat, footprint) diambil dari JSON python, bukan dihitung ulang.
(function (root) {
  "use strict";

  const ALGORITHMS = {
    hc: { name: "Stochastic Hill Climbing", axis: "iterasi" },
    hc_stochastic_aima: { name: "Stochastic Hill Climbing (AIMA)", axis: "iterasi" },
    hc_sideways: { name: "Hill Climbing Sideways Move", axis: "iterasi" },
    hc_random_restart: { name: "Hill Climbing Random Restart", axis: "iterasi" },
    sa: { name: "Simulated Annealing", axis: "iterasi" },
    ga: { name: "Genetic Algorithm", axis: "generasi" },
  };

  // hc biasa dan semua variannya (hc_*) berbagi bentuk history dasar yang sama
  const isHc = (run) => run.algorithm === "hc" || run.algorithm.startsWith("hc_");

  // cek bentuk minimum supaya UI tidak menggambar data rusak seolah sah
  function validateRun(run) {
    if (!run || typeof run !== "object") throw new Error("file hasil bukan objek JSON");
    if (!ALGORITHMS[run.algorithm]) throw new Error(`algoritma tidak dikenal: ${run.algorithm}`);
    if (!run.problem || !run.problem.ship || !Array.isArray(run.problem.vehicles)) {
      throw new Error("problem.ship / problem.vehicles tidak ada");
    }
    if (!Array.isArray(run.history) || run.history.length === 0) throw new Error("history kosong");
    for (const key of ["initial_state", "final_state"]) {
      if (!run[key] || !Array.isArray(run[key].placements)) throw new Error(`${key} tidak valid`);
    }
    return run;
  }

  // frame ke-i berisi state yang benar-benar dimiliki run pada iterasi/generasi i.
  // hc/sa: current_state null = sama persis dengan frame sebelumnya (disimpan hanya saat berubah).
  // bila snapshot dimatikan, frame tengah ditandai unknown; tidak ada state yang dikarang.
  function buildFrames(run) {
    const last = run.history.length - 1;
    if (run.algorithm === "ga") {
      return run.history.map((h, i) => ({
        index: i,
        entry: h,
        state: h.best_individual || (i === last ? run.final_state : null),
        stateKnown: Boolean(h.best_individual) || i === last,
      }));
    }
    const recorded = run.config.record_snapshots !== false;
    let current = null;
    return run.history.map((h, i) => {
      if (h.current_state) current = h.current_state;
      let state = recorded ? current : null;
      if (i === 0 && !state) state = run.initial_state;
      if (i === last && !state) state = run.final_state;
      return { index: i, entry: h, state, stateKnown: Boolean(state) };
    });
  }

  // deskripsi seri chart per algoritma; nilai diambil langsung dari history
  function chartsFor(run) {
    const h = run.history;
    const pick = (key) => h.map((x) => x[key]);
    if (run.algorithm === "hc_random_restart") {
      return [{
        title: "Objective current per restart dan best-so-far (current turun saat restart baru)",
        yLabel: "total ShippingFee",
        series: [
          { name: "current objective", values: pick("objective"), kind: "line", color: "var(--c-current)" },
          { name: "best-so-far", values: pick("best_so_far_objective"), kind: "line", color: "var(--c-best)",
            dashed: true },
          { name: "awal restart", values: h.map((x) => (x.reason === "restart_initial" ? x.objective : null)),
            kind: "dot", color: "var(--c-stuck)" },
        ],
      }];
    }
    if (run.algorithm === "hc_sideways") {
      return [{
        title: "Objective current (naik, atau tetap saat sideways move)",
        yLabel: "total ShippingFee",
        series: [
          { name: "current objective", values: pick("objective"), kind: "line", color: "var(--c-current)" },
          { name: "improvement diterima", values: h.map((x) => (x.improved ? x.objective : null)), kind: "dot",
            color: "var(--c-accept)" },
          { name: "sideways move", values: h.map((x) => (x.sideways ? x.objective : null)), kind: "dot",
            color: "var(--c-stuck)" },
        ],
      }];
    }
    if (run.algorithm === "hc_stochastic_aima") {
      return [
        {
          title: "Objective current (HC tidak pernah turun)",
          yLabel: "total ShippingFee",
          series: [
            { name: "current objective", values: pick("objective"), kind: "line", color: "var(--c-current)" },
            { name: "improvement diterima", values: h.map((x) => (x.accepted ? x.objective : null)), kind: "dot",
              color: "var(--c-accept)" },
          ],
        },
        {
          title: "Jumlah uphill neighbor yang tersedia per iterasi",
          yLabel: "jumlah neighbor",
          yMin: 0,
          series: [{ name: "uphill neighbor", values: pick("uphill_count"), kind: "line", color: "var(--c-temp)" }],
        },
      ];
    }
    if (run.algorithm === "hc") {
      return [{
        title: "Objective current (HC tidak pernah turun)",
        yLabel: "total ShippingFee",
        series: [
          { name: "current objective", values: pick("objective"), kind: "line", color: "var(--c-current)" },
          { name: "improvement diterima", values: h.map((x) => (x.accepted ? x.objective : null)), kind: "dot",
            color: "var(--c-accept)" },
        ],
      }];
    }
    if (run.algorithm === "sa") {
      return [
        {
          title: "Objective: current boleh turun, best-so-far tidak",
          yLabel: "total ShippingFee",
          series: [
            { name: "current objective", values: pick("current_objective"), kind: "line", color: "var(--c-current)" },
            { name: "best-so-far", values: pick("best_so_far_objective"), kind: "line", color: "var(--c-best)" },
            { name: "masuk local optimum", values: h.map((x) => (x.stuck_event ? x.current_objective : null)),
              kind: "dot", color: "var(--c-stuck)" },
          ],
        },
        {
          title: "Acceptance probability e^(ΔE/T) (1 bila ΔE ≥ 0)",
          yLabel: "peluang",
          yMin: 0,
          yMax: 1,
          series: [{ name: "acceptance probability", values: pick("acceptance_probability"), kind: "dot",
            color: "var(--c-prob)" }],
        },
        {
          title: "Temperature T (cooling geometric)",
          yLabel: "T",
          yMin: 0,
          series: [{ name: "temperature", values: pick("temperature"), kind: "line", color: "var(--c-temp)" }],
        },
      ];
    }
    return [{
      title: "Fitness populasi per generasi",
      yLabel: "total ShippingFee",
      series: [
        { name: "best fitness", values: pick("best_fitness"), kind: "line", color: "var(--c-best)" },
        { name: "average fitness", values: pick("average_fitness"), kind: "line", color: "var(--c-current)",
          dashed: true },
        { name: "best-so-far", values: pick("best_so_far_fitness"), kind: "line", color: "var(--c-temp)",
          dashed: true },
      ],
    }];
  }

  function formatMove(move) {
    if (!move) return "-";
    const [kind, id, a, b, o] = move;
    if (kind === "swap") return `swap ${id} ↔ ${a}`;
    if (kind === "rotate") return `rotate ${id}`;
    if (a === null || a === undefined) return `relocate ${id} → luar kapal`;
    return `relocate ${id} → (${a}, ${b}) ${o}`;
  }

  const fmt = (v, digits = 3) => (v === null || v === undefined ? "-" : Number.isInteger(v) ? String(v) : v.toFixed(digits));

  // varian hc memilih dari SEMUA neighbor (bukan satu kandidat acak), jadi kalimatnya berbeda dari hc biasa
  function explainHcVariant(run, h, move) {
    const where = run.algorithm === "hc_random_restart" ? `Restart ${h.restart + 1}, langkah ${h.iteration}` :
      `Iterasi ${h.iteration}`;
    const lines = [];
    if (h.reason === "improved") {
      if (run.algorithm === "hc_stochastic_aima") {
        const how = run.config.selection === "uniform" ? "sama rata" : "sebanding dengan kecuramannya";
        lines.push(`${where}: ada ${h.uphill_count} uphill neighbor. Satu dipilih acak (${how}): ${move}, ` +
          `bernilai ${fmt(h.candidate_objective)}, jadi pindah.`);
      } else {
        lines.push(`${where}: neighbor terbaik (${move}) bernilai ${fmt(h.candidate_objective)} > current, jadi pindah.`);
      }
    } else if (h.reason === "sideways") {
      lines.push(`${where}: neighbor terbaik (${move}) bernilai sama (${fmt(h.candidate_objective)}). ` +
        "Ini sideways move: pindah walau tidak membaik.");
    } else if (h.reason === "max_sideways") {
      lines.push(`${where}: neighbor terbaik bernilai sama, tapi batas sideways beruntun sudah tercapai. Berhenti.`);
    } else if (h.reason === "local_optimum") {
      const ending = run.algorithm === "hc_random_restart" ? "restart ini selesai" : "berhenti";
      lines.push(`${where}: tidak ada neighbor yang lebih baik dari current. Sudah di peak (local optimum), ${ending}.`);
    } else {
      lines.push(`${where}: ${h.reason}.`);
    }
    lines.push(`Objective current: ${fmt(h.objective)}.`);
    return lines;
  }

  // kalimat penjelasan keputusan pada frame i, langsung dari field history
  function explainFrame(run, frames, i) {
    const h = frames[i].entry;
    if (run.algorithm === "ga") {
      return [
        `Generasi ${h.generation}: best fitness ${fmt(h.best_fitness)}, rata-rata ${fmt(h.average_fitness, 1)}, ` +
          `terburuk ${fmt(h.worst_fitness)}.`,
        `Best-so-far sejauh ini: ${fmt(h.best_so_far_fitness)}. Individu unik: ${h.unique_individuals} dari ` +
          `${run.config.population_size}.`,
        "Deck menampilkan individu terbaik pada generasi ini (bukan seluruh populasi dan bukan 'current state').",
      ];
    }
    if (run.algorithm === "hc_random_restart" && h.reason === "restart_initial") {
      return [`Restart ${h.restart + 1}: state awal random baru yang feasible, objective ${fmt(h.objective)}.`,
        `Best-so-far dari semua restart: ${fmt(h.best_so_far_objective)}.`];
    }
    if (i === 0) return ["Iterasi 0: state awal random yang feasible."];
    const move = formatMove(h.move);
    if (isHc(run) && run.algorithm !== "hc") return explainHcVariant(run, h, move);
    if (run.algorithm === "hc") {
      const lines = {
        improved: `Candidate (${move}) bernilai ${fmt(h.candidate_objective)} > current sebelumnya, jadi diterima.`,
        equal: `Candidate (${move}) bernilai sama (${fmt(h.candidate_objective)}), jadi ditolak: HC hanya pindah bila strictly better.`,
        worse: `Candidate (${move}) lebih buruk (${fmt(h.candidate_objective)}), jadi ditolak. State tidak berpindah.`,
        no_feasible_candidate: "Tidak menemukan successor feasible setelah percobaan acak terbatas. State tetap (bukan bukti local optimum).",
      };
      return [`Iterasi ${h.iteration}: ${lines[h.reason]}`, `Objective current: ${fmt(h.objective)}.`];
    }
    // sa
    if (h.reason === "no_feasible_candidate") {
      return [`Iterasi ${h.iteration} (T=${fmt(h.temperature)}): tidak ada candidate feasible. State tetap.`];
    }
    const base = `Iterasi ${h.iteration}: candidate (${move}) bernilai ${fmt(h.candidate_objective)}, ` +
      `ΔE = ${fmt(h.delta_e)}, T = ${fmt(h.temperature)}.`;
    const decision = {
      better: "ΔE > 0, jadi pasti diterima.",
      equal: "ΔE = 0, jadi diterima (p = e^0 = 1, move netral).",
      worse_accepted: `ΔE < 0: p = e^(ΔE/T) = ${fmt(h.acceptance_probability, 4)}, random draw = ${fmt(h.random_draw, 4)} < p, jadi DITERIMA walau lebih buruk.`,
      worse_rejected: `ΔE < 0: p = e^(ΔE/T) = ${fmt(h.acceptance_probability, 4)}, random draw = ${fmt(h.random_draw, 4)} ≥ p, jadi ditolak.`,
    }[h.reason];
    const lines = [base, decision, `Current ${fmt(h.current_objective)}, best-so-far ${fmt(h.best_so_far_objective)}.`];
    const prevStuck = frames[i - 1].entry.stuck;
    if (h.stuck_event) lines.push("State ini masuk local optimum: tidak ada satu pun neighbor feasible yang lebih baik.");
    else if (prevStuck && h.stuck === false) lines.push("Keluar dari local optimum (escape) karena menerima move.");
    else if (h.stuck) lines.push("Masih di local optimum.");
    return lines;
  }

  // warna stabil per id (hash sederhana -> hue), sama di semua frame dan algoritma
  function vehicleColor(id) {
    let hash = 0;
    for (const ch of String(id)) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
    const hue = (hash * 137.508) % 360;
    return `hsl(${hue.toFixed(1)}, 62%, 72%)`;
  }

  // ringkasan run untuk panel summary; label final berbeda per algoritma agar tidak menyesatkan
  function summaryOf(run) {
    const m = run.metrics || {};
    const rows = [
      ["Algoritma", ALGORITHMS[run.algorithm].name],
      ["Seed", run.seed],
      ["Parameter", Object.entries(run.config).map(([k, v]) => `${k}=${v}`).join(", ")],
      ["Runtime search", `${run.runtime_seconds.toFixed(3)} s`],
      ["Objective awal", run.initial_objective],
      [run.algorithm === "sa" ? "Objective terminal (final)" : run.algorithm === "ga" ? "Objective best-found (final)" :
        run.algorithm === "hc_random_restart" ? "Objective final (terbaik antar restart)" : "Objective final",
        run.final_objective],
      ["Final feasible (backend)", run.final_state.feasible ? "ya" : "TIDAK"],
    ];
    if (run.algorithm === "sa") {
      rows.push(["Best-found", `${m.best_found_objective} (iterasi ${m.best_found_iteration})`]);
      rows.push(["Episode stuck / escape", `${m.stuck_event_count} / ${m.escape_count}`]);
      rows.push(["Berhenti karena", `${m.termination_reason} setelah ${m.iterations_run} iterasi`]);
    }
    if (run.algorithm === "hc") {
      rows.push(["Improvement / tolak equal / tolak worse", `${m.improvements} / ${m.rejected_equal} / ${m.rejected_worse}`]);
    }
    if (run.algorithm === "hc_sideways") {
      rows.push(["Improvement", m.improvements]);
      rows.push(["Sideways total / maks beruntun", `${m.sideways_moves_total} / ${m.max_consecutive_sideways}`]);
      rows.push(["Berhenti karena", `${m.termination_reason} setelah ${m.iterations} iterasi`]);
    }
    if (run.algorithm === "hc_stochastic_aima") {
      rows.push(["Improvement", m.improvements]);
      rows.push(["Cara memilih uphill neighbor", m.selection]);
      rows.push(["Berhenti karena", `${m.termination_reason} setelah ${m.iterations} iterasi`]);
    }
    if (run.algorithm === "hc_random_restart") {
      rows.push(["Restart", m.restarts]);
      rows.push(["Restart terbaik", `ke-${m.best_restart_index + 1} (objective ${m.best_found_objective})`]);
      rows.push(["Total langkah", m.total_iterations]);
    }
    if (run.algorithm === "ga") rows.push(["Evaluasi fitness", m.fitness_evaluations]);
    return rows;
  }

  // jumlah evaluasi objective oleh search (untuk perbandingan yang adil; null bila tidak tercatat)
  function evaluationsOf(run) {
    const m = run.metrics || {};
    const value = run.algorithm === "ga" ? m.fitness_evaluations : m.objective_evaluations;
    return value === undefined ? null : value;
  }

  // nilai terbaik sepanjang run: sa menyimpan best-found terpisah dari terminal
  function bestOf(run) {
    return run.algorithm === "sa" ? run.metrics.best_found_objective : run.final_objective;
  }

  const api = { ALGORITHMS, isHc, validateRun, buildFrames, chartsFor, formatMove, explainFrame, vehicleColor, summaryOf,
    evaluationsOf, bestOf };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RunModel = api;
})(this);
