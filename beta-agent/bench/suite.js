// The benchmark suite: fixed, reproducible graphs. Writes bench/suite.json.  Run: node beta-agent/bench/suite.js
"use strict";
const fs = require("fs");
const path = require("path");
const E = require("../engine.js");

function random(n, p, seed, twoWay) {
  const rng = E.rngFrom(seed * 7919 + n), arcs = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    if (rng() >= p) continue;
    if (rng() < twoWay) arcs.push([i, j], [j, i]);
    else arcs.push(rng() < 0.5 ? [i, j] : [j, i]);
  }
  return arcs;
}

// Caching groups: receivers in k groups; inside a group most pairs share messages; a few one-way arcs between groups.
function groups(k, m, seed) {
  const rng = E.rngFrom(seed * 104729 + k * m), arcs = [], n = k * m;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const same = Math.floor(i / m) === Math.floor(j / m);
    if (same && rng() < 0.75) arcs.push([i, j], [j, i]);
    else if (same && rng() < 0.5) arcs.push(rng() < 0.5 ? [i, j] : [j, i]);
    else if (!same && rng() < 0.06) arcs.push(rng() < 0.5 ? [i, j] : [j, i]);
  }
  return { n, arcs };
}

const ring = (n) => [...Array(n).keys()].flatMap((i) => [[i, (i + 1) % n], [(i + 1) % n, i]]);
const dcycle = (n) => [...Array(n).keys()].map((i) => [i, (i + 1) % n]);

const suite = [];
for (const n of [10, 15, 20, 25, 30, 40, 50, 60]) for (const p of [0.2, 0.4]) for (const seed of [1, 2]) {
  suite.push({ name: `random_n${n}_p${p}_s${seed}`, family: "random", n, arcs: random(n, p, seed, 0.4) });
}
for (const [k, m] of [[3, 5], [4, 6], [5, 6], [6, 8]]) for (const seed of [1, 2]) {
  const g = groups(k, m, seed);
  suite.push({ name: `groups_${k}x${m}_s${seed}`, family: "caching groups", n: g.n, arcs: g.arcs });
}
for (const n of [5, 7, 9, 11]) suite.push({ name: `ring_${n}`, family: "textbook", n, arcs: ring(n) });
for (const n of [6, 12]) suite.push({ name: `directed_cycle_${n}`, family: "textbook", n, arcs: dcycle(n) });

const out = path.join(__dirname, "suite.json");
fs.writeFileSync(out, JSON.stringify(suite));
console.log(`${suite.length} graphs -> ${out}`);
