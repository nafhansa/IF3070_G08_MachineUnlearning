// test adapter RunResult -> frame memakai JSON hasil run python yang asli (bukan mock)
// jalankan dari work/: node --test frontend/tests/
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const RM = require("../runmodel.js");

const root = path.join(__dirname, "..", "..");
const load = (rel) => JSON.parse(fs.readFileSync(path.join(root, rel), "utf8"));
const HC = "output/hill_climbing/runs/hc_seed3001.json";
const SA = "output/simulated_annealing/runs/sa_seed4003.json";
const GA = "output/genetic_algorithm/runs/A_pop50_gen50_seed1001.json";

const placementsKey = (state) => JSON.stringify(state.placements.map((p) => [p.id, p.loaded, p.x, p.y, p.orientation]));

test("validateRun menerima ketiga jenis hasil nyata", () => {
  for (const rel of [HC, SA, GA]) assert.doesNotThrow(() => RM.validateRun(load(rel)));
});

test("validateRun menolak data rusak dengan pesan jelas", () => {
  assert.throws(() => RM.validateRun(null), /bukan objek/);
  assert.throws(() => RM.validateRun({ algorithm: "xx" }), /algoritma/);
  const broken = load(HC);
  broken.history = [];
  assert.throws(() => RM.validateRun(broken), /history/);
  const noShip = load(GA);
  delete noShip.problem.ship;
  assert.throws(() => RM.validateRun(noShip), /ship/);
});

test("HC: frame = history, state null diwarisi dari frame sebelumnya, frame terakhir = final", () => {
  const run = load(HC);
  const frames = RM.buildFrames(run);
  assert.equal(frames.length, run.history.length);
  assert.equal(frames[0].index, 0);
  assert.equal(placementsKey(frames[0].state), placementsKey(run.initial_state));
  assert.equal(placementsKey(frames.at(-1).state), placementsKey(run.final_state));
  frames.forEach((f, i) => {
    assert.ok(f.stateKnown);
    assert.equal(f.state.objective, run.history[i].objective, `objective frame ${i}`);
  });
  // frame yang ditolak memakai state yang sama persis dengan frame sebelumnya
  const rejected = frames.findIndex((f, i) => i > 0 && !run.history[i].accepted);
  assert.equal(frames[rejected].state, frames[rejected - 1].state);
});

test("SA: current, best-so-far, peluang dan suhu masuk ke seri chart", () => {
  const run = load(SA);
  const frames = RM.buildFrames(run);
  assert.equal(frames.length, run.metrics.iterations_run + 1);
  frames.forEach((f, i) => assert.equal(f.state.objective, run.history[i].current_objective));
  assert.equal(placementsKey(frames.at(-1).state), placementsKey(run.final_state));
  const charts = RM.chartsFor(run);
  const names = charts.flatMap((c) => c.series.map((s) => s.name));
  assert.ok(names.includes("current objective") && names.includes("best-so-far"));
  assert.ok(charts.some((c) => c.series.some((s) => s.name.startsWith("acceptance probability"))));
  const current = charts[0].series.find((s) => s.name === "current objective").values;
  assert.ok(current.some((v, i) => i > 0 && v < current[i - 1]), "current SA nyata pernah turun");
});

test("GA: deck memakai best individual per generasi dan chart best/average", () => {
  const run = load(GA);
  const frames = RM.buildFrames(run);
  assert.equal(frames.length, run.config.generations + 1);
  frames.forEach((f, i) => assert.equal(f.state.objective, run.history[i].best_fitness));
  const names = RM.chartsFor(run)[0].series.map((s) => s.name);
  assert.deepEqual(names, ["best fitness", "average fitness", "best-so-far"]);
});

test("tanpa snapshot: frame tengah ditandai unknown, tidak dikarang", () => {
  const run = load(HC);
  run.config.record_snapshots = false;
  run.history.forEach((h) => { h.current_state = null; });
  const frames = RM.buildFrames(run);
  assert.equal(placementsKey(frames[0].state), placementsKey(run.initial_state));
  assert.equal(placementsKey(frames.at(-1).state), placementsKey(run.final_state));
  assert.equal(frames[5].stateKnown, false);
  assert.equal(frames[5].state, null);
});

test("penjelasan frame sesuai alasan di history", () => {
  const hc = load(HC);
  const hcFrames = RM.buildFrames(hc);
  const equalIdx = hc.history.findIndex((h) => h.reason === "equal");
  assert.match(RM.explainFrame(hc, hcFrames, equalIdx).join(" "), /ditolak/);
  const sa = load(SA);
  const saFrames = RM.buildFrames(sa);
  const worseIdx = sa.history.findIndex((h) => h.reason === "worse_accepted");
  const text = RM.explainFrame(sa, saFrames, worseIdx).join(" ");
  assert.match(text, /ΔE/);
  assert.match(text, /draw/);
  const eventIdx = sa.history.findIndex((h, i) => i > 0 && h.stuck_event);
  assert.match(RM.explainFrame(sa, saFrames, eventIdx).join(" "), /masuk local optimum/);
});

test("format move dan warna kendaraan stabil", () => {
  assert.equal(RM.formatMove(["swap", "V01", "V02"]), "swap V01 ↔ V02");
  assert.equal(RM.formatMove(["rotate", "V03"]), "rotate V03");
  assert.equal(RM.formatMove(["relocate", "V05", 2, 7, "V"]), "relocate V05 → (2, 7) V");
  assert.equal(RM.formatMove(["relocate", "V05", null, null, null]), "relocate V05 → luar kapal");
  assert.equal(RM.formatMove(null), "-");
  assert.equal(RM.vehicleColor("V07"), RM.vehicleColor("V07"));
  assert.notEqual(RM.vehicleColor("V07"), RM.vehicleColor("V08"));
});

test("evaluationsOf dan bestOf membaca metrik yang benar per algoritma", () => {
  const hc = load(HC);
  const ga = load(GA);
  const sa = load(SA);
  assert.equal(RM.evaluationsOf(hc), hc.metrics.objective_evaluations);
  assert.equal(RM.evaluationsOf(ga), ga.metrics.fitness_evaluations);
  assert.equal(RM.evaluationsOf(sa), sa.metrics.objective_evaluations ?? null);
  assert.equal(RM.bestOf(sa), sa.metrics.best_found_objective);
  assert.equal(RM.bestOf(hc), hc.final_objective);
});
