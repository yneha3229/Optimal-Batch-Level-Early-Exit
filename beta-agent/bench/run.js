// Run the JavaScript methods on the suite. One process per (graph, method), 3 in parallel, same limit for all.
//   node beta-agent/bench/run.js [--limit 30] [--jobs 3]        -> bench/results_js.jsonl
// Methods:
//   greedy       classic heuristics only: greedy acyclic set + DSatur clique cover (what a quick practitioner runs)
//   plain_exact  the same exact searches as Pincer, but run the textbook way: whole graph at once, no decomposition,
//                no heuristic warm start, no early stop; first the acyclic set, then the cover, each to its proof
//   pincer       the agent pipeline (decompose, greedy, local search, exact where a gap remains, stop at L = U)
"use strict";
const fs = require("fs");
const path = require("path");
const { fork } = require("child_process");
const E = require("../engine.js");

function solve(method, item, limitMs) {
  const g = E.makeGraph(item.n, item.arcs);
  const t0 = Date.now();
  let L, U, lowerProven, upperProven, acyc, groups;
  if (method === "greedy") {
    const s = new E.Session(g, 1);
    s.run("lower_greedy"); s.run("upper_greedy");
    ({ L, U, lowerProven, upperProven } = s); acyc = s.acyclic; groups = s.groups;
  } else if (method === "plain_exact") {
    const deadline = t0 + limitMs;
    const a = E.acyclicExactOne(g, [], deadline);
    const lowerBound = a.set.length;
    const c = E.coverExactOne(g, [...Array(g.n).keys()].map((v) => [v]), a.proven ? lowerBound : 1, deadline);
    acyc = a.set; groups = c.groups; L = acyc.length; U = groups.length; lowerProven = a.proven; upperProven = c.proven;
    if (!E.isAcyclic(g, acyc) || !E.isCliqueCover(g, groups)) throw new Error("invalid certificate");
  } else if (method === "pincer") {
    const s = new E.Session(g, 1);
    for (const d of E.plannerSteps(s, limitMs)) s.run(d.tool, d.args);
    ({ L, U, lowerProven, upperProven } = s); acyc = s.acyclic; groups = s.groups;
    // when did the bounds last change? (time to the final answer, before any remaining proof effort)
    let tb = 0;
    for (let k = 1; k < s.steps.length; k++) {
      const a = s.steps[k - 1], b = s.steps[k];
      if (a.L !== b.L || a.U !== b.U || a.lowerProven !== b.lowerProven || a.upperProven !== b.upperProven) tb = b.ms;
      if (a.L !== b.L || a.U !== b.U) var tv = b.ms;   // eslint-disable-line no-var
    }
    var tValues = tv || 0, tProof = tb;            // eslint-disable-line no-var
  }
  // a meeting of the bounds proves both optimal: L <= MAIS <= beta <= cover <= U
  const met = L === U;
  return { graph: item.name, family: item.family, n: item.n, arcs: item.arcs.length, method, L, U,
    lower_optimal_proven: lowerProven || met, upper_optimal_proven: upperProven || met, beta_exact: met,
    seconds: (Date.now() - t0) / 1000,
    ...(method === "pincer" ? { seconds_to_final_values: tValues / 1000, seconds_to_last_proof: tProof / 1000 } : {}) };
}

if (process.argv[2] === "--child") {
  const [method, name, limit] = process.argv.slice(3);
  const item = JSON.parse(fs.readFileSync(path.join(__dirname, "suite.json"))).find((x) => x.name === name);
  process.send(solve(method, item, Number(limit)));
  process.exit(0);
}

(async () => {
  const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? Number(process.argv[i + 1]) : d; };
  const limit = arg("--limit", 30) * 1000, jobs = arg("--jobs", 3);
  const mi = process.argv.indexOf("--methods");
  const methods = mi > 0 ? process.argv[mi + 1].split(",") : ["greedy", "plain_exact", "pincer"];
  const suite = JSON.parse(fs.readFileSync(path.join(__dirname, "suite.json")));
  const out = path.join(__dirname, "results_js.jsonl");
  const done = new Set(fs.existsSync(out) ? fs.readFileSync(out, "utf8").trim().split("\n").filter(Boolean)
    .map((l) => { const r = JSON.parse(l); return r.graph + "|" + r.method; }) : []);
  const queue = [];
  for (const item of suite) for (const m of methods) if (!done.has(item.name + "|" + m)) queue.push([m, item.name]);
  console.log(`${queue.length} jobs, limit ${limit / 1000} s, ${jobs} in parallel`);
  const runOne = ([m, name]) => new Promise((resolve) => {
    const child = fork(__filename, ["--child", m, name, String(limit)]);
    const kill = setTimeout(() => child.kill("SIGKILL"), limit + 15000);   // hard stop if a solver overruns
    let got = null;
    child.on("message", (r) => { got = r; });
    child.on("exit", () => {
      clearTimeout(kill);
      const r = got || { graph: name, method: m, error: "killed after limit" };
      fs.appendFileSync(out, JSON.stringify(r) + "\n");
      console.log(`${m.padEnd(12)} ${name.padEnd(24)} ${r.error || `[${r.L}, ${r.U}] ${r.seconds.toFixed(2)} s`}`);
      resolve();
    });
  });
  const workers = Array.from({ length: jobs }, async () => { while (queue.length) await runOne(queue.shift()); });
  await Promise.all(workers);
})();
