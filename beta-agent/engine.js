/*
 * beta-agent engine: bounds on the broadcast rate beta of an index-coding problem, and the agent that drives them.
 *
 * Model. Receivers 0..n-1; receiver i wants message x_i. An arc i -> j means "receiver i already has x_j"
 * (side information). adj[i][j] = true for that arc.
 *
 * The two bounds (the "pincer"):
 *   L = size of a largest ACYCLIC induced subgraph (MAIS). No scheme can use fewer than L transmissions.
 *   U = number of groups in a smallest CLIQUE COVER: split the receivers into groups in which every pair knows each
 *       other's message (arcs both ways). Broadcasting the XOR of each group's messages serves everyone, so
 *       U transmissions suffice.
 *   L <= beta <= U.  If L == U, beta is known exactly.
 * Both quantities add up over strongly connected components (a directed cycle, or a two-way pair, never leaves its
 * component), so every solver works component by component.
 *
 * Trust. Solvers only PROPOSE certificates (a vertex set, a list of groups). The Session checks each one
 * (isAcyclic, isCliqueCover) before it can change L or U. A tool can waste time, never produce a wrong bound.
 *
 * Works in the browser (global BetaEngine) and in Node (module.exports).
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------------------------------------- graph
  function makeGraph(n, arcs) {
    const adj = Array.from({ length: n }, () => new Array(n).fill(false));
    for (const [i, j] of arcs) if (i !== j && i >= 0 && j >= 0 && i < n && j < n) adj[i][j] = true;
    return { n, adj };
  }

  function arcsOf(g) {
    const out = [];
    for (let i = 0; i < g.n; i++) for (let j = 0; j < g.n; j++) if (g.adj[i][j]) out.push([i, j]);
    return out;
  }

  function mutual(g, i, j) {
    return g.adj[i][j] && g.adj[j][i];
  }

  /** Strongly connected components (Tarjan, iterative). */
  function scc(g) {
    const n = g.n, index = new Array(n).fill(-1), low = new Array(n).fill(0), on = new Array(n).fill(false);
    const stack = [], comps = [];
    let counter = 0;
    for (let r = 0; r < n; r++) {
      if (index[r] !== -1) continue;
      const work = [[r, 0]];
      while (work.length) {
        const top = work[work.length - 1];
        const v = top[0];
        if (top[1] === 0 && index[v] === -1) {
          index[v] = low[v] = counter++;
          stack.push(v);
          on[v] = true;
        }
        let pushed = false;
        while (top[1] < n) {
          const w = top[1]++;
          if (!g.adj[v][w]) continue;
          if (index[w] === -1) { work.push([w, 0]); pushed = true; break; }
          if (on[w]) low[v] = Math.min(low[v], index[w]);
        }
        if (pushed) continue;
        work.pop();
        if (work.length) { const u = work[work.length - 1][0]; low[u] = Math.min(low[u], low[v]); }
        if (low[v] === index[v]) {
          const c = [];
          let w;
          do { w = stack.pop(); on[w] = false; c.push(w); } while (w !== v);
          comps.push(c.sort((a, b) => a - b));
        }
      }
    }
    return comps.sort((a, b) => b.length - a.length || a[0] - b[0]);
  }

  // -------------------------------------------------------------------------------------------- checkers
  /** True iff the subgraph induced on `set` has no directed cycle (Kahn's algorithm). */
  function isAcyclic(g, set) {
    const S = [...new Set(set)];
    if (S.length !== set.length) return false;
    if (S.some((v) => !(v >= 0 && v < g.n))) return false;
    const indeg = new Map(S.map((v) => [v, 0]));
    for (const u of S) for (const v of S) if (g.adj[u][v]) indeg.set(v, indeg.get(v) + 1);
    const q = S.filter((v) => indeg.get(v) === 0);
    let seen = 0;
    while (q.length) {
      const u = q.pop();
      seen++;
      for (const v of S) if (g.adj[u][v]) { indeg.set(v, indeg.get(v) - 1); if (indeg.get(v) === 0) q.push(v); }
    }
    return seen === S.length;
  }

  /** True iff `groups` partition the receivers and every pair inside a group knows each other's message. */
  function isCliqueCover(g, groups) {
    const seen = new Array(g.n).fill(0);
    for (const grp of groups) {
      for (const v of grp) { if (!(v >= 0 && v < g.n)) return false; seen[v]++; }
      for (let a = 0; a < grp.length; a++) for (let b = a + 1; b < grp.length; b++) {
        if (!mutual(g, grp[a], grp[b])) return false;
      }
    }
    return seen.every((c) => c === 1);
  }

  // ------------------------------------------------------------------------------------------------ helpers
  function rngFrom(seed) {
    let s = (seed >>> 0) || 1;
    return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  }

  function sub(g, verts) {
    return { n: verts.length, adj: verts.map((a) => verts.map((b) => g.adj[a][b])) };
  }

  /** Run f on every SCC (as a small graph), map local answers back to global labels. */
  function perComponent(g, f) {
    return scc(g).map((comp) => ({ comp, res: f(sub(g, comp), comp) }));
  }

  // can v join the acyclic set S (bool array) without closing a cycle? (no S-path from an out- to an in-neighbour)
  function canAdd(g, S, v) {
    const n = g.n;
    for (let u = 0; u < n; u++) if (S[u] && g.adj[v][u] && g.adj[u][v]) return false;
    const seen = new Array(n).fill(false), stack = [];
    for (let u = 0; u < n; u++) if (S[u] && g.adj[v][u]) { seen[u] = true; stack.push(u); }
    while (stack.length) {
      const u = stack.pop();
      if (g.adj[u][v]) return false;
      for (let w = 0; w < n; w++) if (S[w] && !seen[w] && g.adj[u][w]) { seen[w] = true; stack.push(w); }
    }
    return true;
  }

  // ---------------------------------------------------------------------------------------- lower bound L
  /** Greedy: repeatedly keep vertices that cannot lie on a cycle (no in- or out-arc among the undecided),
   *  otherwise delete the vertex with the most in x out arcs; then try to put deleted vertices back. */
  function acyclicGreedyOne(g) {
    const n = g.n, alive = new Array(n).fill(true), keep = new Array(n).fill(false), deleted = [];
    let left = n;
    while (left > 0) {
      let changed = true;
      while (changed) {
        changed = false;
        for (let v = 0; v < n; v++) {
          if (!alive[v]) continue;
          let din = 0, dout = 0;
          for (let u = 0; u < n; u++) if (alive[u] && u !== v) { if (g.adj[u][v]) din++; if (g.adj[v][u]) dout++; }
          if (din === 0 || dout === 0) { alive[v] = false; keep[v] = true; left--; changed = true; }
        }
      }
      if (left === 0) break;
      let best = -1, bestScore = -1;
      for (let v = 0; v < n; v++) {
        if (!alive[v]) continue;
        let din = 0, dout = 0;
        for (let u = 0; u < n; u++) if (alive[u]) { if (g.adj[u][v]) din++; if (g.adj[v][u]) dout++; }
        if (din * dout > bestScore) { bestScore = din * dout; best = v; }
      }
      alive[best] = false; left--; deleted.push(best);
    }
    for (const v of deleted.reverse()) if (canAdd(g, keep, v)) keep[v] = true;
    return keep.map((k, v) => (k ? v : -1)).filter((v) => v >= 0);
  }

  function topoOrder(g, set) {
    const S = new Set(set), indeg = new Map(set.map((v) => [v, 0]));
    for (const u of set) for (const v of set) if (g.adj[u][v]) indeg.set(v, indeg.get(v) + 1);
    const q = set.filter((v) => indeg.get(v) === 0), out = [];
    while (q.length) {
      const u = q.shift(); out.push(u);
      for (const v of S) if (g.adj[u][v]) { indeg.set(v, indeg.get(v) - 1); if (indeg.get(v) === 0) q.push(v); }
    }
    return out;
  }

  /** Simulated annealing on a topological order of the kept set (Galinier et al. 2013 move: insert an outside
   *  vertex just after its last kept in-neighbour or just before its first kept out-neighbour and drop the kept
   *  vertices that would point backwards). Tries the best of 5 sampled vertices per move. */
  function acyclicAnnealOne(g, init, ms, rng) {
    const n = g.n;
    let seq = topoOrder(g, init), best = seq.slice(), T = 0.6;
    const deadline = Date.now() + ms;
    while (Date.now() < deadline) {
      for (let it = 0; it < Math.max(50, 2 * n); it++) {
        const pos = new Map(seq.map((v, k) => [v, k]));
        const outside = [];
        for (let v = 0; v < n; v++) if (!pos.has(v)) outside.push(v);
        if (!outside.length) return seq.slice().sort((a, b) => a - b);
        let move = null;
        for (let t = 0; t < 5; t++) {
          const v = outside[Math.floor(rng() * outside.length)];
          const kin = [], kout = [];
          for (const u of seq) { if (g.adj[u][v]) kin.push(u); if (g.adj[v][u]) kout.push(u); }
          const pMinus = kin.length ? Math.max(...kin.map((u) => pos.get(u))) + 1 : 0;
          const pPlus = kout.length ? Math.min(...kout.map((u) => pos.get(u))) : seq.length;
          for (const p of [pMinus, pPlus]) {
            const conf = new Set();
            for (const u of kin) if (pos.get(u) >= p) conf.add(u);
            for (const u of kout) if (pos.get(u) < p) conf.add(u);
            const delta = conf.size - 1;
            if (!move || delta < move.delta) move = { v, p, conf, delta };
          }
        }
        if (move.delta <= 0 || rng() < Math.exp(-move.delta / T)) {
          seq = seq.slice(0, move.p).filter((u) => !move.conf.has(u)).concat([move.v],
            seq.slice(move.p).filter((u) => !move.conf.has(u)));
          if (seq.length > best.length) best = seq.slice();
        }
      }
      T = Math.max(0.05, T * 0.97);
    }
    return best.slice().sort((a, b) => a - b);
  }

  /** Exact maximum acyclic set by branch and bound (include / exclude each vertex, most-connected first).
   *  Returns {set, proven}; proven = false if the deadline cut the search. */
  function acyclicExactOne(g, init, deadline) {
    const n = g.n;
    const order = [...Array(n).keys()].sort((a, b) => deg(g, b) - deg(g, a));
    let best = init.slice(), proven = true, nodes = 0;
    const S = new Array(n).fill(false);
    function rec(k, size) {
      if ((++nodes & 1023) === 0 && Date.now() > deadline) { proven = false; return; }
      if (!proven) return;
      if (size + (n - k) <= best.length) return;
      if (k === n) { best = S.map((x, v) => (x ? v : -1)).filter((v) => v >= 0); return; }
      const v = order[k];
      if (canAdd(g, S, v)) { S[v] = true; rec(k + 1, size + 1); S[v] = false; }
      rec(k + 1, size);
    }
    rec(0, 0);
    return { set: best.slice().sort((a, b) => a - b), proven };
  }

  function deg(g, v) {
    let d = 0;
    for (let u = 0; u < g.n; u++) { if (g.adj[u][v]) d++; if (g.adj[v][u]) d++; }
    return d;
  }

  // ---------------------------------------------------------------------------------------- upper bound U
  // Clique cover of the "know each other" graph = colouring of its complement, the CONFLICT graph:
  // two receivers conflict (must be in different groups) unless both arcs between them exist.
  function conflicts(g) {
    return Array.from({ length: g.n }, (_, i) => Array.from({ length: g.n }, (_, j) => i !== j && !mutual(g, i, j)));
  }

  function colorsToGroups(colors) {
    const m = new Map();
    colors.forEach((c, v) => { if (!m.has(c)) m.set(c, []); m.get(c).push(v); });
    return [...m.values()];
  }

  /** DSatur colouring of the conflict graph (classic greedy). */
  function coverGreedyOne(g) {
    const n = g.n, C = conflicts(g), color = new Array(n).fill(-1);
    for (let step = 0; step < n; step++) {
      let best = -1, bestSat = -1, bestDeg = -1;
      for (let v = 0; v < n; v++) {
        if (color[v] >= 0) continue;
        const sat = new Set(), d = C[v].filter(Boolean).length;
        for (let u = 0; u < n; u++) if (C[v][u] && color[u] >= 0) sat.add(color[u]);
        if (sat.size > bestSat || (sat.size === bestSat && d > bestDeg)) { best = v; bestSat = sat.size; bestDeg = d; }
      }
      const used = new Set();
      for (let u = 0; u < n; u++) if (C[best][u] && color[u] >= 0) used.add(color[u]);
      let c = 0;
      while (used.has(c)) c++;
      color[best] = c;
    }
    return colorsToGroups(color);
  }

  /** Min-conflicts search: try to recolour with one colour fewer than the current cover, repeatedly. */
  function coverLocalOne(g, groups, ms, rng) {
    const n = g.n, C = conflicts(g);
    let bestGroups = groups;
    const deadline = Date.now() + ms;
    while (Date.now() < deadline && bestGroups.length > 1) {
      const k = bestGroups.length - 1;
      const color = new Array(n).fill(0);
      bestGroups.forEach((grp, c) => grp.forEach((v) => { color[v] = Math.min(c, k - 1); }));
      const tabu = new Map();
      let solved = false;
      for (let it = 0; Date.now() < deadline && it < 20000; it++) {
        const bad = [];
        for (let v = 0; v < n; v++) for (let u = 0; u < n; u++) if (C[v][u] && color[u] === color[v]) { bad.push(v); break; }
        if (!bad.length) { solved = true; break; }
        const v = bad[Math.floor(rng() * bad.length)];
        let bestC = color[v], bestCost = Infinity;
        for (let c = 0; c < k; c++) {
          if (c === color[v] || (tabu.get(v * 1000 + c) || 0) > it) continue;
          let cost = 0;
          for (let u = 0; u < n; u++) if (C[v][u] && color[u] === c) cost++;
          if (cost < bestCost || (cost === bestCost && rng() < 0.5)) { bestCost = cost; bestC = c; }
        }
        tabu.set(v * 1000 + color[v], it + 7);
        color[v] = bestC;
      }
      if (!solved) break;
      bestGroups = colorsToGroups(color);
    }
    return bestGroups;
  }

  /** Exact minimum cover: is the conflict graph k-colourable, for k = lower .. current-1 (backtracking, most
   *  constrained receiver first)? Returns {groups, proven}. `lower` may be any valid lower bound on the cover
   *  (an acyclic set is one: its receivers pairwise conflict, so they need different groups). */
  function coverExactOne(g, groups, lower, deadline) {
    const n = g.n, C = conflicts(g);
    let best = groups, proven = true;
    for (let k = best.length - 1; k >= Math.max(1, lower); k--) {
      const color = new Array(n).fill(-1);
      let nodes = 0, timedOut = false;
      const rec = (done) => {
        if ((++nodes & 1023) === 0 && Date.now() > deadline) { timedOut = true; return false; }
        if (done === n) return true;
        let v = -1, vSat = -1;
        for (let u = 0; u < n; u++) {
          if (color[u] >= 0) continue;
          const s = new Set();
          for (let w = 0; w < n; w++) if (C[u][w] && color[w] >= 0) s.add(color[w]);
          if (s.size > vSat) { vSat = s.size; v = u; }
        }
        const maxUsed = Math.max(-1, ...color);
        for (let c = 0; c < Math.min(k, maxUsed + 2); c++) {   // symmetry: a new colour only as the next index
          let ok = true;
          for (let w = 0; w < n; w++) if (C[v][w] && color[w] === c) { ok = false; break; }
          if (!ok) continue;
          color[v] = c;
          if (rec(done + 1)) return true;
          if (timedOut) return false;
          color[v] = -1;
        }
        return false;
      };
      const found = rec(0);
      if (timedOut) { proven = false; break; }
      if (!found) break;          // not k-colourable: the current best (k+1 groups) is optimal
      best = colorsToGroups(color);
    }
    return { groups: best, proven };
  }

  // ---------------------------------------------------------------------------------- the agent's session
  const TOOLS = [
    { name: "analyze_graph", side: "info",
      description: "Describe the graph: receivers, arcs, two-way pairs, strongly connected components (both bounds add up over components, so each is solved separately). Fast. Call first." },
    { name: "lower_greedy", side: "lower",
      description: "Lower bound L: greedy acyclic set (keep receivers that cannot be on a cycle, delete the most-connected one, repeat). Milliseconds." },
    { name: "lower_local_search", side: "lower", ms: true,
      description: "Lower bound L: simulated annealing for a larger acyclic set, starting from the best known one. Takes `ms` milliseconds (default 400)." },
    { name: "lower_exact", side: "lower", ms: true,
      description: "Lower bound L: exact maximum acyclic set by branch and bound, per component. Proves L optimal (no larger acyclic set exists) if it finishes within `ms` (default 3000). Exponential: fine up to about 25-30 receivers per component." },
    { name: "upper_greedy", side: "upper",
      description: "Upper bound U: greedy clique cover (DSatur colouring of the receivers that do NOT know each other). Milliseconds." },
    { name: "upper_local_search", side: "upper", ms: true,
      description: "Upper bound U: min-conflicts search for a cover with one group fewer, repeatedly, for `ms` milliseconds (default 400)." },
    { name: "upper_exact", side: "upper", ms: true,
      description: "Upper bound U: exact minimum clique cover by backtracking, searching down from the current U to the known lower bound L. Proves U optimal for clique covers if it finishes within `ms` (default 3000)." },
    { name: "get_status", side: "info",
      description: "Current L and U, whether each is proven optimal for its method, and the tools already used." },
  ];

  class Session {
    constructor(g, seed) {
      this.g = g;
      this.rng = rngFrom(seed || 12345);
      this.comps = scc(g);
      this.acyclic = [];                       // certificate for L (global labels)
      this.groups = [...Array(g.n).keys()].map((v) => [v]); // certificate for U: everyone alone (always valid)
      this.lowerProven = g.n === 0;
      this.upperProven = g.n === 0;
      this.steps = [];
      this.t0 = Date.now();
      this.log("start", {}, `Receivers: ${g.n}, arcs: ${arcsOf(g).length}. Start with the trivial bounds 0 <= beta <= ${g.n}.`);
    }

    get L() { return this.acyclic.length; }
    get U() { return this.groups.length; }
    get exact() { return this.g.n > 0 && this.L === this.U; }
    get done() { return this.exact || (this.lowerProven && this.upperProven); }

    snapshot() {
      return { L: this.L, U: this.U, acyclic: this.acyclic.slice(), groups: this.groups.map((x) => x.slice()),
        lowerProven: this.lowerProven, upperProven: this.upperProven };
    }

    log(tool, args, note, extra) {
      const step = { k: this.steps.length, tool, args, note, ms: Date.now() - this.t0, ...this.snapshot(), ...(extra || {}) };
      this.steps.push(step);
      if (this.onStep) this.onStep(step);
      return step;
    }

    // Offers: checked before they change anything.
    offerAcyclic(set, tool) {
      if (!isAcyclic(this.g, set)) throw new Error(`${tool} proposed a set with a cycle (rejected)`);
      if (set.length > this.L) { this.acyclic = set.slice().sort((a, b) => a - b); return true; }
      return false;
    }
    offerGroups(groups, tool) {
      if (!isCliqueCover(this.g, groups)) throw new Error(`${tool} proposed an invalid cover (rejected)`);
      if (groups.length < this.U) { this.groups = groups.map((x) => x.slice().sort((a, b) => a - b)); return true; }
      return false;
    }

    // Each per-component solver gets the component's best known part as its starting point.
    localAcyclic(comp) { const pos = new Map(comp.map((v, k) => [v, k])); return this.acyclic.filter((v) => pos.has(v)).map((v) => pos.get(v)); }
    localGroups(comp) {
      const pos = new Map(comp.map((v, k) => [v, k]));
      return this.groups.map((grp) => grp.filter((v) => pos.has(v)).map((v) => pos.get(v))).filter((x) => x.length);
    }

    run(tool, args) {
      args = args || {};
      const ms = Math.max(10, Math.min(60000, Number(args.ms) || (tool.endsWith("exact") ? 3000 : 400)));
      const L0 = this.L, U0 = this.U;
      const g = this.g;
      const t = Date.now();
      let note = "", result = {};
      if (tool === "analyze_graph") {
        const arcs = arcsOf(g), two = arcs.filter(([i, j]) => i < j && g.adj[j][i]).length;
        result = { receivers: g.n, arcs: arcs.length, two_way_pairs: two,
          components: this.comps.map((c) => c.length) };
        note = `${g.n} receivers, ${arcs.length} arcs (${two} two-way pairs). ${this.comps.length} strongly connected component(s) of sizes ${this.comps.map((c) => c.length).join(", ")}. Bounds add up over components.`;
      } else if (tool === "lower_greedy" || tool === "lower_local_search" || tool === "lower_exact") {
        const deadline = Date.now() + ms;
        let proven = true;
        const set = [];
        const share = ms / Math.max(1, this.comps.length);
        for (const comp of this.comps) {
          const h = sub(g, comp);
          let local;
          if (tool === "lower_greedy") local = acyclicGreedyOne(h);
          else if (tool === "lower_local_search") {
            const init = this.localAcyclic(comp);
            local = acyclicAnnealOne(h, init.length ? init : acyclicGreedyOne(h), share, this.rng);
          } else {
            const init = this.localAcyclic(comp);
            const r = acyclicExactOne(h, init.length ? init : acyclicGreedyOne(h), deadline);
            local = r.set; proven = proven && r.proven;
          }
          set.push(...local.map((k) => comp[k]));
        }
        const better = this.offerAcyclic(set, tool);
        if (tool === "lower_exact" && proven) this.lowerProven = true;
        result = { L: this.L, improved: better, proven_optimal: this.lowerProven };
        note = better ? `Found an acyclic set of ${set.length} receivers: L ${L0} -> ${this.L}.` : `No larger acyclic set found (L stays ${this.L}).`;
        if (tool === "lower_exact") note += proven ? " Search finished: L is the maximum acyclic set (proven)." : " Time ran out before the search finished.";
      } else if (tool === "upper_greedy" || tool === "upper_local_search" || tool === "upper_exact") {
        const deadline = Date.now() + ms;
        let proven = true;
        const groups = [];
        const share = ms / Math.max(1, this.comps.length);
        const lowerLocal = (comp) => this.localAcyclic(comp).length;
        for (const comp of this.comps) {
          const h = sub(g, comp);
          let local;
          if (tool === "upper_greedy") local = coverGreedyOne(h);
          else if (tool === "upper_local_search") local = coverLocalOne(h, bestOf(this.localGroups(comp), coverGreedyOne(h)), share, this.rng);
          else {
            const r = coverExactOne(h, bestOf(this.localGroups(comp), coverGreedyOne(h)), lowerLocal(comp), deadline);
            local = r.groups; proven = proven && r.proven;
          }
          groups.push(...local.map((grp) => grp.map((k) => comp[k])));
        }
        const better = this.offerGroups(groups, tool);
        if (tool === "upper_exact" && proven) this.upperProven = true;
        result = { U: this.U, improved: better, proven_optimal: this.upperProven };
        note = better ? `Found a clique cover with ${groups.length} groups: U ${U0} -> ${this.U}.` : `No smaller clique cover found (U stays ${this.U}).`;
        if (tool === "upper_exact") note += proven ? " Search finished: no clique cover with fewer groups exists (proven)." : " Time ran out before the search finished.";
      } else if (tool === "get_status") {
        result = this.status();
        note = `Status: ${this.L} <= beta <= ${this.U}.`;
      } else {
        throw new Error(`unknown tool ${tool}`);
      }
      const step = this.log(tool, tool.endsWith("greedy") || tool === "analyze_graph" || tool === "get_status" ? {} : { ms },
        note, { took: Date.now() - t });
      return { ...result, L: this.L, U: this.U, beta_exact: this.exact, note: step.note };
    }

    status() {
      return { L: this.L, U: this.U, beta_exact: this.exact, lower_proven_optimal: this.lowerProven,
        upper_proven_optimal: this.upperProven, components: this.comps.map((c) => c.length),
        tools_used: this.steps.slice(1).map((s) => s.tool) };
    }

    verdict() {
      if (this.exact) return { kind: "exact", text: `beta = ${this.L}: the lower and upper bounds meet.` };
      if (this.lowerProven && this.upperProven) {
        return { kind: "gap", text: `${this.L} <= beta <= ${this.U}. Both bounds are optimal for their method, so these two bounds cannot pin beta down further on this graph (stronger tools, such as LP bounds, would be needed).` };
      }
      return { kind: "open", text: `${this.L} <= beta <= ${this.U}. The search was stopped before both bounds were proven optimal.` };
    }
  }

  function bestOf(a, b) { return a.length && a.length <= b.length ? a : b; }

  /** The built-in planner: a cost-aware playbook (iterative deepening). Cheap greedy bounds first; then exact
   *  search with a tiny time slice, doubled only after it times out, on whichever side is still unproven; a short
   *  local search improves the incumbent between attempts (a better incumbent prunes the exact search). Stops as
   *  soon as L = U (that alone proves both optimal) or both sides are proven. Yields one decision at a time so the
   *  UI can animate it. */
  function* plannerSteps(session, budgetMs) {
    const why = (text, tool, args) => ({ why: text, tool, args: args || {} });
    const left = () => budgetMs - (Date.now() - session.t0);
    yield why("Look at the structure first: components can be solved separately.", "analyze_graph");
    yield why("Cheapest lower bound: a greedy acyclic set.", "lower_greedy");
    yield why("Cheapest upper bound: a greedy clique cover (the XOR scheme).", "upper_greedy");
    let slice = 50;
    for (let round = 0; !session.done && left() > 20; round++) {
      for (const side of ["lower", "upper"]) {
        if (session.done || left() <= 20) return;
        if (side === "lower" ? session.lowerProven : session.upperProven) continue;
        const ms = Math.max(20, Math.min(slice, left()));
        yield why(round === 0
          ? `Gap ${session.L}..${session.U}. Exact search is often instant on small components: try it with ${ms} ms.`
          : `Still unproven: retry the exact search with a doubled slice (${ms} ms).`, side + "_exact", { ms });
        if (session.done) return;
        const proven = side === "lower" ? session.lowerProven : session.upperProven;
        if (!proven && left() > 40) {
          const lms = Math.max(20, Math.min(Math.floor(slice / 2), left() - 20));
          yield why(`Exact search timed out: improve the incumbent by local search (${lms} ms) so the next attempt prunes more.`,
            side + "_local_search", { ms: lms });
        }
      }
      slice *= 2;
    }
  }

  const api = { makeGraph, arcsOf, scc, isAcyclic, isCliqueCover, Session, TOOLS, plannerSteps,
    acyclicGreedyOne, acyclicAnnealOne, acyclicExactOne, coverGreedyOne, coverLocalOne, coverExactOne, rngFrom };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BetaEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
