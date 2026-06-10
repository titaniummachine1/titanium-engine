// Parity reference: run the original ACE v7 JS engine at fixed depth.
// Usage: node ace_ref_run.js <depth> [ace numeric moves...]
"use strict";
const path = require("path");
const { Quoridor, Search } = require(path.join(__dirname, "acev7_engine.js"));

const depth = parseInt(process.argv[2] || "8", 10);
const moves = process.argv.slice(3).map(Number);

const g = new Quoridor();
for (const m of moves) g.makeMove(m);
console.log("hash", g.hashLo, g.hashHi);

const s = new Search(g);
const r = s.think(1e9, depth, true);
console.log(JSON.stringify(r));
