// Tests for engine.js.  Run:  node --test beta-agent/
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const E = require("./engine.js");

function randomGraph(n, p, seed, twoWay = 0.4) {
  const rng = E.rngFrom(seed), arcs = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    if (rng() >= p) continue;
    if (rng() < twoWay) arcs.push([i, j], [j, i]);
    else arcs.push(rng() < 0.5 ? [i, j] : [j, i]);
  }
  return E.makeGraph(n, arcs);
}

function bruteMais(g) {
  let best = 0;
  for (let m = 0; m < 1 << g.n; m++) {
    const set = [...Array(g.n).keys()].filter((v) => (m >> v) & 1);
    if (set.length > best && E.isAcyclic(g, set)) best = set.length;
  }
  return best;
}

function bruteCover(g) {  // smallest k such that the receivers split into k mutual cliques
  for (let k = 1; k <= g.n; k++) {
    const color = new Array(g.n).fill(-1);
    const ok = (v) => {
      if (v === g.n) return true;
      for (let c = 0; c < k; c++) {
        let fine = true;
        for (let u = 0; u < v; u++) if (color[u] === c && !(g.adj[u][v] && g.adj[v][u])) { fine = false; break; }
        if (!fine) continue;
        color[v] = c;
        if (ok(v + 1)) return true;
      }
      color[v] = -1;
      return false;
    };
    if (ok(0)) return k;
  }
  return g.n;
}

function runPlanner(g, budget = 4000) {
  const s = new E.Session(g, 7);
  for (const d of E.plannerSteps(s, budget)) s.run(d.tool, d.args);
  return s;
}

const cycle = (n, twoWay) => E.makeGraph(n, [...Array(n).keys()].flatMap((i) =>
  twoWay ? [[i, (i + 1) % n], [(i + 1) % n, i]] : [[i, (i + 1) % n]]));

test("exact solvers match brute force on 300 random graphs", () => {
  for (let seed = 1; seed <= 300; seed++) {
    const n = 3 + (seed % 7), g = randomGraph(n, 0.2 + (seed % 5) * 0.15, seed);
    const mais = E.acyclicExactOne(g, [], Date.now() + 5000);
    assert.ok(mais.proven && E.isAcyclic(g, mais.set));
    assert.equal(mais.set.length, bruteMais(g), `MAIS seed ${seed}`);
    const cov = E.coverExactOne(g, E.coverGreedyOne(g), 1, Date.now() + 5000);
    assert.ok(cov.proven && E.isCliqueCover(g, cov.groups));
    assert.equal(cov.groups.length, bruteCover(g), `cover seed ${seed}`);
  }
});

test("heuristics always return valid certificates", () => {
  for (let seed = 1; seed <= 60; seed++) {
    const g = randomGraph(15 + (seed % 20), 0.15 + (seed % 4) * 0.1, seed);
    const rng = E.rngFrom(seed);
    const a = E.acyclicGreedyOne(g);
    assert.ok(E.isAcyclic(g, a));
    assert.ok(E.isAcyclic(g, E.acyclicAnnealOne(g, a, 30, rng)));
    const c = E.coverGreedyOne(g);
    assert.ok(E.isCliqueCover(g, c));
    assert.ok(E.isCliqueCover(g, E.coverLocalOne(g, c, 30, rng)));
  }
});

test("checkers reject bad certificates", () => {
  const g = cycle(4, false);
  assert.equal(E.isAcyclic(g, [0, 1, 2, 3]), false);
  assert.equal(E.isAcyclic(g, [0, 1, 2]), true);
  assert.equal(E.isCliqueCover(g, [[0, 1], [2], [3]]), false);   // 0 and 1 know each other only one way
  assert.equal(E.isCliqueCover(g, [[0], [1], [2]]), false);       // 3 not covered
  const s = new E.Session(g, 1);
  assert.throws(() => s.offerAcyclic([0, 1, 2, 3], "test"));
  assert.throws(() => s.offerGroups([[0, 1, 2, 3]], "test"));
});

test("known cases", () => {
  // Five friends in a ring, each knows both neighbours: 2 <= beta <= 3 (true beta = 2.5, so these bounds cannot meet)
  let s = runPlanner(cycle(5, true));
  assert.deepEqual([s.L, s.U, s.lowerProven, s.upperProven], [2, 3, true, true]);
  assert.equal(s.verdict().kind, "gap");
  // Directed cycle of 6 (each knows the next): MAIS = 5, no two-way pairs so U = 6
  s = runPlanner(cycle(6, false));
  assert.deepEqual([s.L, s.U], [5, 6]);
  // Everyone knows everyone: one XOR serves all, beta = 1
  const K = E.makeGraph(5, [...Array(5).keys()].flatMap((i) => [...Array(5).keys()].filter((j) => j !== i).map((j) => [i, j])));
  s = runPlanner(K);
  assert.deepEqual([s.L, s.U, s.exact], [1, 1, true]);
  // Nobody knows anything: beta = n
  s = runPlanner(E.makeGraph(4, []));
  assert.deepEqual([s.L, s.U, s.exact], [4, 4, true]);
});

test("bounds add up over components", () => {
  // two-way triangle (beta = 1) + directed 3-cycle (beta = 2) + an isolated receiver (1), with one-way arcs between
  const arcs = [[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [3, 6]];
  const s = runPlanner(E.makeGraph(7, arcs));
  assert.deepEqual(s.comps.map((c) => c.length), [3, 3, 1]);
  assert.equal(s.L, 1 + 2 + 1);
  assert.equal(s.U, 1 + 3 + 1);
});

test("session log records every step with its bounds", () => {
  const s = runPlanner(randomGraph(12, 0.4, 3));
  assert.ok(s.steps.length >= 4);
  for (const st of s.steps) assert.ok(st.L <= st.U);
  for (let k = 1; k < s.steps.length; k++) {
    assert.ok(s.steps[k].L >= s.steps[k - 1].L && s.steps[k].U <= s.steps[k - 1].U, "bounds never get worse");
  }
});
