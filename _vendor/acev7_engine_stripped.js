var DELTA = [-9, 9, -1, 1], DIRBIT = [1, 2, 4, 8];
var MATE = 100000, MAX_PLY = 64;

// ---------- zobrist ----------
var zSeed = 0x9e3779b9 | 0;
function zrand() { zSeed ^= zSeed << 13; zSeed ^= zSeed >>> 17; zSeed ^= zSeed << 5; return zSeed >>> 0; }
var Z_PAWN_LO = [], Z_PAWN_HI = [];
for (var zi = 0; zi < 2; zi++) {
  Z_PAWN_LO.push(new Uint32Array(81)); Z_PAWN_HI.push(new Uint32Array(81));
  for (var zj = 0; zj < 81; zj++) { Z_PAWN_LO[zi][zj] = zrand(); Z_PAWN_HI[zi][zj] = zrand(); }
}
var Z_HW_LO = new Uint32Array(64), Z_HW_HI = new Uint32Array(64);
var Z_VW_LO = new Uint32Array(64), Z_VW_HI = new Uint32Array(64);
for (var zs = 0; zs < 64; zs++) { Z_HW_LO[zs] = zrand(); Z_HW_HI[zs] = zrand(); Z_VW_LO[zs] = zrand(); Z_VW_HI[zs] = zrand(); }
var Z_TURN_LO = zrand(), Z_TURN_HI = zrand();

// ---------- game state ----------
function Quoridor() {
  this.histM = new Int16Array(1024);
  this.histFrom = new Int16Array(1024);
  this.histLw = new Int16Array(1024);
  this.hashesU = new Uint32Array(2048);
  this.reset();
}

Quoridor.prototype.reset = function () {
  this.pawn = [76, 4];           // player 0 bottom (8,4) goal row 0; player 1 top (0,4) goal row 8
  this.wl = [10, 10];
  this.turn = 0;
  this.hw = new Uint8Array(64);
  this.vw = new Uint8Array(64);
  this.blocked = new Uint8Array(81); // bits N=1 S=2 W=4 E=8 (walls only; bounds checked separately)
  this.hashLo = (Z_PAWN_LO[0][76] ^ Z_PAWN_LO[1][4]) >>> 0;
  this.hashHi = (Z_PAWN_HI[0][76] ^ Z_PAWN_HI[1][4]) >>> 0;
  this.histLen = 0;
  this.lastWallPly = 0;  // repetition can only reach back to the last wall placement
  this.wallStamp = 0;    // bumped on every wall make/unmake; dist fields depend only on walls
};

Quoridor.prototype.loadState = function (st) {
  this.reset();
  for (var i = 0; i < st.moves.length; i++) this.makeMove(st.moves[i]);
};

var BORDER = new Uint8Array(81);
for (var bc = 0; bc < 81; bc++) {
  var br = (bc / 9) | 0, bcl = bc % 9;
  BORDER[bc] = (br === 0 ? 1 : 0) | (br === 8 ? 2 : 0) | (bcl === 0 ? 4 : 0) | (bcl === 8 ? 8 : 0);
}
Quoridor.prototype.canStep = function (cell, dir) {
  return ((this.blocked[cell] | BORDER[cell]) & DIRBIT[dir]) === 0;
};

Quoridor.prototype.winner = function () {
  if (this.pawn[0] < 9) return 0;
  if (this.pawn[1] >= 72) return 1;
  return -1;
};

// ---------- wall mechanics ----------
Quoridor.prototype.setWallBits = function (type, slot, on) {
  var r = (slot / 8) | 0, c = slot % 8, a, b, cc, dd;
  if (type === 0) {
    a = r * 9 + c; b = a + 1; cc = a + 9; dd = b + 9;
    if (on) { this.blocked[a] |= 2; this.blocked[b] |= 2; this.blocked[cc] |= 1; this.blocked[dd] |= 1; }
    else { this.blocked[a] &= ~2; this.blocked[b] &= ~2; this.blocked[cc] &= ~1; this.blocked[dd] &= ~1; }
  } else {
    a = r * 9 + c; b = a + 9; cc = a + 1; dd = b + 1;
    if (on) { this.blocked[a] |= 8; this.blocked[b] |= 8; this.blocked[cc] |= 4; this.blocked[dd] |= 4; }
    else { this.blocked[a] &= ~8; this.blocked[b] &= ~8; this.blocked[cc] &= ~4; this.blocked[dd] &= ~4; }
  }
};

Quoridor.prototype.wallFits = function (type, slot) {
  var r = (slot / 8) | 0, c = slot % 8;
  if (this.hw[slot] || this.vw[slot]) return false;
  if (type === 0) {
    if (c > 0 && this.hw[slot - 1]) return false;
    if (c < 7 && this.hw[slot + 1]) return false;
  } else {
    if (r > 0 && this.vw[slot - 8]) return false;
    if (r < 7 && this.vw[slot + 8]) return false;
  }
  return true;
};

// Conservative "cannot possibly seal" precheck (over-counts anchors, so safe to skip BFS)
Quoridor.prototype.wallNeedsPathCheck = function (type, slot) {
  var r = (slot / 8) | 0, c = slot % 8, anchors = 0;
  if (type === 0) { if (c === 0) anchors++; if (c === 7) anchors++; }
  else { if (r === 0) anchors++; if (r === 7) anchors++; }
  for (var dr = -2; dr <= 2 && anchors < 2; dr++) {
    var rr = r + dr; if (rr < 0 || rr > 7) continue;
    for (var dc = -2; dc <= 2; dc++) {
      var ccc = c + dc; if (ccc < 0 || ccc > 7) continue;
      var ss = rr * 8 + ccc;
      if (this.hw[ss] || this.vw[ss]) { anchors++; if (anchors >= 2) break; }
    }
  }
  return anchors >= 2;
};

var BFS_Q = new Int16Array(81);
Quoridor.prototype.hasPath = function (player) {
  var goal = player === 0 ? 0 : 8, start = this.pawn[player];
  if (((start / 9) | 0) === goal) return true;
  var seen = this._seen || (this._seen = new Uint8Array(81));
  seen.fill(0);
  var head = 0, tail = 0;
  BFS_Q[tail++] = start; seen[start] = 1;
  var blk2 = this.blocked;
  while (head < tail) {
    var u = BFS_Q[head++], bm2 = blk2[u] | BORDER[u];
    for (var d = 0; d < 4; d++) {
      if (bm2 & DIRBIT[d]) continue;
      var v = u + DELTA[d];
      if (seen[v]) continue;
      if (((v / 9) | 0) === goal) return true;
      seen[v] = 1; BFS_Q[tail++] = v;
    }
  }
  return false;
};

Quoridor.prototype.wallLegal = function (type, slot) {
  if (this.wl[this.turn] <= 0) return false;
  if (!this.wallFits(type, slot)) return false;
  if (!this.wallNeedsPathCheck(type, slot)) return true;
  this.setWallBits(type, slot, true);
  var ok = this.hasPath(0) && this.hasPath(1);
  this.setWallBits(type, slot, false);
  return ok;
};

// ---------- pawn moves ----------
Quoridor.prototype.genPawnMoves = function (out, n) {
  var me = this.turn, s = this.pawn[me], o = this.pawn[1 - me];
  for (var d = 0; d < 4; d++) {
    if (!this.canStep(s, d)) continue;
    var t = s + DELTA[d];
    if (t !== o) { out[n++] = t; continue; }
    if (this.canStep(o, d)) { out[n++] = o + DELTA[d]; continue; }
    var p1 = d < 2 ? 2 : 0, p2 = d < 2 ? 3 : 1;
    if (this.canStep(o, p1)) { var w1 = o + DELTA[p1]; if (w1 !== s) out[n++] = w1; }
    if (this.canStep(o, p2)) { var w2 = o + DELTA[p2]; if (w2 !== s) out[n++] = w2; }
  }
  return n;
};

// ---------- make / unmake (allocation-free) ----------
Quoridor.prototype.makeMove = function (m) {
  var hl = this.histLen;
  this.histM[hl] = m; this.histLw[hl] = this.lastWallPly;
  if (m < 100) {
    var p = this.turn;
    this.histFrom[hl] = this.pawn[p];
    this.hashLo = (this.hashLo ^ Z_PAWN_LO[p][this.pawn[p]] ^ Z_PAWN_LO[p][m]) >>> 0;
    this.hashHi = (this.hashHi ^ Z_PAWN_HI[p][this.pawn[p]] ^ Z_PAWN_HI[p][m]) >>> 0;
    this.pawn[p] = m;
  } else if (m < 200) {
    var s0 = m - 100;
    this.hw[s0] = 1; this.setWallBits(0, s0, true); this.wl[this.turn]--; this.wallStamp++;
    this.hashLo = (this.hashLo ^ Z_HW_LO[s0]) >>> 0; this.hashHi = (this.hashHi ^ Z_HW_HI[s0]) >>> 0;
    this.lastWallPly = hl + 1;
  } else {
    var s1 = m - 200;
    this.vw[s1] = 1; this.setWallBits(1, s1, true); this.wl[this.turn]--; this.wallStamp++;
    this.hashLo = (this.hashLo ^ Z_VW_LO[s1]) >>> 0; this.hashHi = (this.hashHi ^ Z_VW_HI[s1]) >>> 0;
    this.lastWallPly = hl + 1;
  }
  this.turn ^= 1;
  this.hashLo = (this.hashLo ^ Z_TURN_LO) >>> 0; this.hashHi = (this.hashHi ^ Z_TURN_HI) >>> 0;
  this.hashesU[hl * 2] = this.hashLo; this.hashesU[hl * 2 + 1] = this.hashHi;
  this.histLen = hl + 1;
};

Quoridor.prototype.unmakeMove = function () {
  var hl = --this.histLen;
  var m = this.histM[hl];
  this.lastWallPly = this.histLw[hl];
  this.turn ^= 1;
  this.hashLo = (this.hashLo ^ Z_TURN_LO) >>> 0; this.hashHi = (this.hashHi ^ Z_TURN_HI) >>> 0;
  if (m < 100) {
    var p = this.turn, from = this.histFrom[hl];
    this.hashLo = (this.hashLo ^ Z_PAWN_LO[p][from] ^ Z_PAWN_LO[p][m]) >>> 0;
    this.hashHi = (this.hashHi ^ Z_PAWN_HI[p][from] ^ Z_PAWN_HI[p][m]) >>> 0;
    this.pawn[p] = from;
  } else if (m < 200) {
    var s0 = m - 100;
    this.hw[s0] = 0; this.setWallBits(0, s0, false); this.wl[this.turn]++; this.wallStamp--;
    this.hashLo = (this.hashLo ^ Z_HW_LO[s0]) >>> 0; this.hashHi = (this.hashHi ^ Z_HW_HI[s0]) >>> 0;
  } else {
    var s1 = m - 200;
    this.vw[s1] = 0; this.setWallBits(1, s1, false); this.wl[this.turn]++; this.wallStamp--;
    this.hashLo = (this.hashLo ^ Z_VW_LO[s1]) >>> 0; this.hashHi = (this.hashHi ^ Z_VW_HI[s1]) >>> 0;
  }
};

// ---------- distance fields ----------
Quoridor.prototype.computeDist = function (player, dist) {
  dist.fill(255);
  var goal = player === 0 ? 0 : 8, head = 0, tail = 0;
  for (var c = 0; c < 9; c++) { var cell = goal * 9 + c; dist[cell] = 0; BFS_Q[tail++] = cell; }
  var blk = this.blocked;
  while (head < tail) {
    var u = BFS_Q[head++], du = dist[u] + 1, bm = blk[u] | BORDER[u];
    for (var d = 0; d < 4; d++) {
      if (bm & DIRBIT[d]) continue;
      var v = u + DELTA[d];
      if (dist[v] > du) { dist[v] = du; BFS_Q[tail++] = v; }
    }
  }
};

Quoridor.prototype.markPath = function (player, dist, mark) {
  var cur = this.pawn[player], bit = 1 << player, guard = 0;
  mark[cur] |= bit;
  while (dist[cur] > 0 && guard++ < 100) {
    for (var d = 0; d < 4; d++) {
      if (!this.canStep(cur, d)) continue;
      var v = cur + DELTA[d];
      if (dist[v] === dist[cur] - 1) { cur = v; mark[cur] |= bit; break; }
    }
  }
};

Quoridor.prototype.legalMoves = function () {
  var out = new Int16Array(160), n = 0;
  n = this.genPawnMoves(out, n);
  if (this.wl[this.turn] > 0) {
    for (var slot = 0; slot < 64; slot++) {
      if (this.wallLegal(0, slot)) out[n++] = 100 + slot;
      if (this.wallLegal(1, slot)) out[n++] = 200 + slot;
    }
  }
  return Array.prototype.slice.call(out.subarray(0, n));
};

// ---------- search ----------
var TT_BITS = 20, TT_SIZE = 1 << TT_BITS, TT_MASK = TT_SIZE - 1;

function Search(game) {
  this.g = game;
  this.ttKeyHi = new Uint32Array(TT_SIZE);
  this.ttKeyLo = new Uint32Array(TT_SIZE);
  this.ttMeta = new Int32Array(TT_SIZE);   // move | flag<<10 | depth<<12, 0 = empty
  this.ttScore = new Int32Array(TT_SIZE);
  this.historyTbl = new Int32Array(512);
  this.cm = new Int16Array(512);           // countermove table
  this.killers = []; this.moveBuf = []; this.scoreBuf = [];
  for (var i = 0; i < MAX_PLY; i++) {
    this.killers.push([0, 0]);
    this.moveBuf.push(new Int16Array(160));
    this.scoreBuf.push(new Int32Array(160));
  }
  this.pathLo = new Uint32Array(MAX_PLY); this.pathHi = new Uint32Array(MAX_PLY);
  this.D0 = []; this.D1 = [];
  for (var dp = 0; dp < MAX_PLY; dp++) { this.D0.push(new Uint8Array(81)); this.D1.push(new Uint8Array(81)); }
  this.dist0 = this.D0[0]; this.dist1 = this.D1[0]; // refs into D0/D1, swapped per ply
  this.pathMark = new Uint8Array(81);
  this.nodes = 0; this.deadline = 0; this.cachedStamp = -1;
}

Search.prototype.checkTime = function () {
  if ((this.nodes & 1023) === 0 && Date.now() > this.deadline) throw "time";
};

Search.prototype.refreshDist = function (ply) {
  var g = this.g;
  if (this.cachedStamp === g.wallStamp) return; // refs already valid for these walls
  if (this.cachedStamp === g.wallStamp - 1 && g.histLen > 0) {
    // exactly one wall added since the cached config: refs hold its dists.
    // recompute a player's field only if the wall cuts a shortest-path edge
    // (|dist diff| === 1); equal-dist edges lie on no shortest path.
    var m = g.histM[g.histLen - 1];
    if (m >= 100) {
      var slot = m % 100, a = (slot >> 3) * 9 + (slot & 7), b2, c2, e2;
      if (m < 200) { b2 = a + 9; c2 = a + 1; e2 = a + 10; }  // hw: two vertical edges
      else { b2 = a + 1; c2 = a + 9; e2 = a + 10; }          // vw: two horizontal edges
      var d0 = this.dist0, d1 = this.dist1;
      if (d0[a] !== d0[b2] || d0[c2] !== d0[e2]) {
        this.dist0 = this.D0[ply]; g.computeDist(0, this.dist0); // redirect first: never write an ancestor's array
      }
      if (d1[a] !== d1[b2] || d1[c2] !== d1[e2]) {
        this.dist1 = this.D1[ply]; g.computeDist(1, this.dist1);
      }
      this.cachedStamp = g.wallStamp;
      return;
    }
  }
  this.dist0 = this.D0[ply]; this.dist1 = this.D1[ply]; // own arrays: ancestors stay intact
  g.computeDist(0, this.dist0);
  g.computeDist(1, this.dist1);
  this.cachedStamp = g.wallStamp;
};

// ---------- HalfPW net tables (auto-generated; must match netlib_halfpw.js) ----------
// NET_DATA stripped
var NET_H = NET_DATA.H;
var NET_WS = Float64Array.from(NET_DATA.Wskip);
var NET_W1C = Float64Array.from(NET_DATA.W1C);
var NET_PO = Float64Array.from(NET_DATA.PO);
var NET_PX = Float64Array.from(NET_DATA.PX);
var NET_B1 = Float64Array.from(NET_DATA.B1);
var NET_W2 = Float64Array.from(NET_DATA.W2);
var NET_HID = new Float64Array(NET_H);
var NET_MIRC = new Int16Array(81);
var NET_MIRS = new Int16Array(64);
var NET_BKT = new Int16Array(81);
for (var nmi = 0; nmi < 81; nmi++) NET_MIRC[nmi] = (8 - ((nmi / 9) | 0)) * 9 + (nmi % 9);
for (var nms = 0; nms < 64; nms++) NET_MIRS[nms] = (7 - ((nms / 8) | 0)) * 8 + (nms % 8);
for (var nbc = 0; nbc < 81; nbc++) NET_BKT[nbc] = ((((nbc / 9) | 0) / 3) | 0) * 3 + (((nbc % 9) / 3) | 0);

// ---------- HalfPW net tables (auto-generated; must match netlib_halfpw.js) ----------
// NET_DATA stripped
var NET_H = NET_DATA.H;
var NET_WS = Float64Array.from(NET_DATA.Wskip);
var NET_W1C = Float64Array.from(NET_DATA.W1C);
var NET_PO = Float64Array.from(NET_DATA.PO);
var NET_PX = Float64Array.from(NET_DATA.PX);
var NET_B1 = Float64Array.from(NET_DATA.B1);
var NET_W2 = Float64Array.from(NET_DATA.W2);
var NET_HID = new Float64Array(NET_H);
var NET_MIRC = new Int16Array(81);
var NET_MIRS = new Int16Array(64);
var NET_BKT = new Int16Array(81);
for (var nmi = 0; nmi < 81; nmi++) NET_MIRC[nmi] = (8 - ((nmi / 9) | 0)) * 9 + (nmi % 9);
for (var nms = 0; nms < 64; nms++) NET_MIRS[nms] = (7 - ((nms / 8) | 0)) * 8 + (nms % 8);
for (var nbc = 0; nbc < 81; nbc++) NET_BKT[nbc] = ((((nbc / 9) | 0) / 3) | 0) * 3 + (((nbc % 9) / 3) | 0);

Search.prototype.evaluate = function () {
  var g = this.g, me = g.turn, opp = 1 - me;
  var dMe = me === 0 ? this.dist0[g.pawn[0]] : this.dist1[g.pawn[1]];
  var dOpp = opp === 0 ? this.dist0[g.pawn[0]] : this.dist1[g.pawn[1]];
  var wMe = g.wl[me], wOpp = g.wl[opp];
  if (wMe === 0 && wOpp === 0) {
    if (dMe <= dOpp) return 3000 + (dOpp - dMe) * 50 - dMe;
    return -3000 - (dMe - dOpp) * 50 + dOpp;
  }
  var pd = dOpp - dMe, wd = wMe - wOpp;
  var out = NET_WS[0] + NET_WS[1] * pd + NET_WS[2] * wd + NET_WS[3] * dMe + NET_WS[4] * dOpp
    + NET_WS[9] * pd * (wMe + wOpp) / 20 + NET_WS[10] * wd * (dMe + dOpp) / 16;
  if (wOpp === 0) { out += NET_WS[6]; if (dMe <= dOpp) out += NET_WS[5]; }
  else if (wMe === 0) { out += NET_WS[8]; if (dOpp <= dMe - 1) out += NET_WS[7]; }
  if (dOpp <= 4) out += NET_WS[11] * (wMe < 3 ? wMe : 3);
  if (dMe <= 4) out += NET_WS[12] * (wOpp < 3 ? wOpp : 3);
  var H = NET_H, j, o, s;
  if (!this.npAcc0) {
    this.npAcc0 = new Float64Array(H); this.npAcc1 = new Float64Array(H);
    this.npHw = new Uint8Array(64); this.npVw = new Uint8Array(64);
    this.npB0 = -1; this.npB1v = -1; this.npStamp = -1;
  }
  var hw = g.hw, vw = g.vw, A0 = this.npAcc0, A1 = this.npAcc1;
  var b0 = NET_BKT[g.pawn[0]], b1 = NET_BKT[NET_MIRC[g.pawn[1]]];
  var rb0 = b0 !== this.npB0, rb1 = b1 !== this.npB1v;
  if (rb0 || rb1) { // bucket crossing: rebuild that perspective from scratch
    if (rb0) { for (j = 0; j < H; j++) A0[j] = 0; }
    if (rb1) { for (j = 0; j < H; j++) A1[j] = 0; }
    for (s = 0; s < 64; s++) {
      if (hw[s]) {
        if (rb0) { o = (b0 * 128 + s) * H;           for (j = 0; j < H; j++) A0[j] += NET_W1C[o + j]; }
        if (rb1) { o = (b1 * 128 + NET_MIRS[s]) * H; for (j = 0; j < H; j++) A1[j] += NET_W1C[o + j]; }
      }
      if (vw[s]) {
        if (rb0) { o = (b0 * 128 + 64 + s) * H;           for (j = 0; j < H; j++) A0[j] += NET_W1C[o + j]; }
        if (rb1) { o = (b1 * 128 + 64 + NET_MIRS[s]) * H; for (j = 0; j < H; j++) A1[j] += NET_W1C[o + j]; }
      }
    }
    if (rb0) this.npB0 = b0;
    if (rb1) this.npB1v = b1;
    var chf = this.npHw, cvf = this.npVw;
    for (s = 0; s < 64; s++) { chf[s] = hw[s]; cvf[s] = vw[s]; }
    this.npStamp = g.wallStamp;
  } else if (this.npStamp !== g.wallStamp) { // wall diff: one row add per change
    var ch = this.npHw, cv = this.npVw, sg;
    for (s = 0; s < 64; s++) {
      if (hw[s] !== ch[s]) {
        sg = hw[s] ? 1 : -1;
        o = (b0 * 128 + s) * H;           for (j = 0; j < H; j++) A0[j] += sg * NET_W1C[o + j];
        o = (b1 * 128 + NET_MIRS[s]) * H; for (j = 0; j < H; j++) A1[j] += sg * NET_W1C[o + j];
        ch[s] = hw[s];
      }
      if (vw[s] !== cv[s]) {
        sg = vw[s] ? 1 : -1;
        o = (b0 * 128 + 64 + s) * H;           for (j = 0; j < H; j++) A0[j] += sg * NET_W1C[o + j];
        o = (b1 * 128 + 64 + NET_MIRS[s]) * H; for (j = 0; j < H; j++) A1[j] += sg * NET_W1C[o + j];
        cv[s] = vw[s];
      }
    }
    this.npStamp = g.wallStamp;
  }
  var hid = NET_HID;
  if (me === 0) {
    for (j = 0; j < H; j++) hid[j] = NET_B1[j] + A0[j];
    o = g.pawn[0] * H;        for (j = 0; j < H; j++) hid[j] += NET_PO[o + j];
    o = g.pawn[1] * H;        for (j = 0; j < H; j++) hid[j] += NET_PX[o + j];
  } else {
    for (j = 0; j < H; j++) hid[j] = NET_B1[j] + A1[j];
    o = NET_MIRC[g.pawn[1]] * H; for (j = 0; j < H; j++) hid[j] += NET_PO[o + j];
    o = NET_MIRC[g.pawn[0]] * H; for (j = 0; j < H; j++) hid[j] += NET_PX[o + j];
  }
  for (j = 0; j < H; j++) { var a2 = hid[j] < 0 ? 0 : (hid[j] > 1 ? 1 : hid[j]); out += NET_W2[j] * a2 * 200; }
  return out | 0;
};

Search.prototype.genMoves = function (ply, checkLegal) {
  var g = this.g, out = this.moveBuf[ply], n = 0;
  n = g.genPawnMoves(out, n);
  if (g.wl[g.turn] > 0) {
    for (var slot = 0; slot < 64; slot++) {
      if (checkLegal) {
        if (g.wallLegal(0, slot)) out[n++] = 100 + slot;
        if (g.wallLegal(1, slot)) out[n++] = 200 + slot;
      } else { // lazy: geometry only; path-seal checked when the move is searched
        if (g.wallFits(0, slot)) out[n++] = 100 + slot;
        if (g.wallFits(1, slot)) out[n++] = 200 + slot;
      }
    }
  }
  return n;
};

Search.prototype.orderMoves = function (ply, n, ttMove, cmMove) {
  var g = this.g, out = this.moveBuf[ply], sc = this.scoreBuf[ply];
  var distMe = g.turn === 0 ? this.dist0 : this.dist1;
  var k = this.killers[ply];
  for (var i = 0; i < n; i++) {
    var m = out[i], s;
    if (m === ttMove) s = 2000000000;
    else if (m < 100) s = 1000000 - distMe[m] * 1000;
    else if (m === k[0]) s = 900000;
    else if (m === cmMove) s = 870000;
    else if (m === k[1]) s = 850000;
    else s = this.historyTbl[m];
    sc[i] = s;
  }
  for (var a = 1; a < n; a++) {
    var mv = out[a], ms = sc[a], b = a - 1;
    while (b >= 0 && sc[b] < ms) { out[b + 1] = out[b]; sc[b + 1] = sc[b]; b--; }
    out[b + 1] = mv; sc[b + 1] = ms;
  }
};

Search.prototype.ab = function (depth, alpha, beta, ply, allowNull, prevMove) {
  this.nodes++; this.checkTime();
  var g = this.g, prev = 1 - g.turn;
  if ((prev === 0 && g.pawn[0] < 9) || (prev === 1 && g.pawn[1] >= 72)) return -(MATE - ply);
  if (ply >= MAX_PLY - 1) return 0;
  this.pathLo[ply] = g.hashLo; this.pathHi[ply] = g.hashHi;
  if (ply > 0) { // repetition: search line, then game history back to last wall
    for (var ri = ply - 1; ri >= 0; ri--)
      if (this.pathLo[ri] === g.hashLo && this.pathHi[ri] === g.hashHi) return 0;
    var hz = g.hashesU, lwp = g.lastWallPly;
    for (var gi = g.histLen * 2 - 4; gi >= lwp * 2; gi -= 2)
      if (hz[gi] === g.hashLo && hz[gi + 1] === g.hashHi) return 0;
  }

  this.refreshDist(ply);
  var nd0 = this.dist0, nd1 = this.dist1, nst = this.cachedStamp; // restored on every unmake
  if (depth <= 0) return this.evaluate();

  // TT probe (typed, always-replace)
  var idx = g.hashLo & TT_MASK, ttMove = 0;
  var meta = this.ttMeta[idx];
  if (meta !== 0 && this.ttKeyHi[idx] === g.hashHi && this.ttKeyLo[idx] === g.hashLo) {
    ttMove = meta & 1023;
    var tdepth = meta >> 12, tflag = (meta >> 10) & 3;
    if (tdepth >= depth && ply > 0) {
      var es = this.ttScore[idx]; // mate scores stored node-relative
      if (es > MATE - 2 * MAX_PLY) es -= ply; else if (es < -(MATE - 2 * MAX_PLY)) es += ply;
      if (tflag === 0) return es;
      if (tflag === 1 && es >= beta) return es;
      if (tflag === 2 && es <= alpha) return es;
    }
  }

  // reverse futility: hopeless to fall below beta at shallow depth
  if (depth <= 4 && beta > -2000 && beta < 2000) {
    var sev = this.evaluate();
    if (sev - 90 * depth >= beta) return sev;
  }

  // null move
  if (allowNull && depth >= 3 && ply > 0) {
    var ev = this.evaluate();
    if (ev >= beta) {
      g.turn ^= 1;
      g.hashLo = (g.hashLo ^ Z_TURN_LO) >>> 0; g.hashHi = (g.hashHi ^ Z_TURN_HI) >>> 0;
      var ns = 0;
      try { ns = -this.ab(depth - 3, -beta, -beta + 1, ply + 1, false, 0); }
      finally {
        g.turn ^= 1;
        g.hashLo = (g.hashLo ^ Z_TURN_LO) >>> 0; g.hashHi = (g.hashHi ^ Z_TURN_HI) >>> 0;
        this.dist0 = nd0; this.dist1 = nd1; this.cachedStamp = nst;
      }
      if (ns >= beta && ns < MATE - 200) return beta;
    }
  }

  var n = this.genMoves(ply, ply === 0);
  if (n === 0) return this.evaluate();
  var cmMove = prevMove > 0 ? this.cm[prevMove] : 0;
  this.orderMoves(ply, n, ttMove, cmMove);

  var best = -Infinity, bestMove = 0, flag = 2, moves = this.moveBuf[ply];
  var localMoves = ply === 0 ? Array.prototype.slice.call(moves.subarray(0, n)) : null;

  for (var i = 0; i < n; i++) {
    var m = localMoves ? localMoves[i] : moves[i];
    if (depth <= 2 && ply > 0 && i >= 10 && m >= 100 && m !== ttMove && this.historyTbl[m] <= 0 && best > -MATE + 200) continue; // frontier LMP
    if (m >= 100 && ply > 0 && g.wallNeedsPathCheck(m < 200 ? 0 : 1, m % 100)) {
      g.setWallBits(m < 200 ? 0 : 1, m % 100, true);
      var pathsOk = g.hasPath(0) && g.hasPath(1);
      g.setWallBits(m < 200 ? 0 : 1, m % 100, false);
      if (!pathsOk) continue; // sealing wall: pseudo-legal only
    }
    g.makeMove(m);
    var score, newDepth = depth - 1;
    try {
      if (i >= 4 && depth >= 3 && m >= 100 && m !== ttMove) {
        var red = 1 + (i >= 12 ? 1 : 0) + (depth >= 6 && i >= 24 ? 1 : 0); // graduated LMR
        var rd = newDepth - red; if (rd < 0) rd = 0;
        score = -this.ab(rd, -alpha - 1, -alpha, ply + 1, true, m);
        if (score > alpha) score = -this.ab(newDepth, -beta, -alpha, ply + 1, true, m);
      } else if (i > 0) {
        score = -this.ab(newDepth, -alpha - 1, -alpha, ply + 1, true, m);
        if (score > alpha && score < beta) score = -this.ab(newDepth, -beta, -alpha, ply + 1, true, m);
      } else {
        score = -this.ab(newDepth, -beta, -alpha, ply + 1, true, m);
      }
    } finally { g.unmakeMove(); this.dist0 = nd0; this.dist1 = nd1; this.cachedStamp = nst; }

    if (score > best) {
      best = score; bestMove = m;
      if (score > alpha) {
        alpha = score; flag = 0;
        if (ply === 0) { this.rootBest = m; this.rootScore = score; }
        if (alpha >= beta) {
          flag = 1;
          if (m >= 100) {
            var kk = this.killers[ply];
            if (kk[0] !== m) { kk[1] = kk[0]; kk[0] = m; }
            this.historyTbl[m] += depth * depth;
            if (this.historyTbl[m] > 100000000) for (var h = 0; h < 512; h++) this.historyTbl[h] >>= 1;
          }
          if (prevMove > 0) this.cm[prevMove] = m;
          break;
        }
      }
    }
  }

  if (best === -Infinity) return this.evaluate(); // all pseudo-legal moves were sealing walls
  var ts = best; // store mate scores node-relative
  if (ts > MATE - 2 * MAX_PLY) ts += ply; else if (ts < -(MATE - 2 * MAX_PLY)) ts -= ply;
  this.ttKeyHi[idx] = g.hashHi; this.ttKeyLo[idx] = g.hashLo;
  this.ttMeta[idx] = bestMove | (flag << 10) | (depth << 12);
  this.ttScore[idx] = ts;
  return best;
};

// entry: returns {move, score, depth, nodes, ms}
Search.prototype.think = function (timeMs, maxDepth, full) {
  var t0 = Date.now();
  this.deadline = t0 + timeMs;
  this.nodes = 0; this.rootBest = 0; this.rootScore = 0;
  var g = this.g;
  // snapshot for time-abort restore (exception may unwind through unmade moves)
  var sp0 = g.pawn[0], sp1 = g.pawn[1], sw0 = g.wl[0], sw1 = g.wl[1], sturn = g.turn;
  var slo = g.hashLo, shi = g.hashHi, shist = g.histLen, slwp = g.lastWallPly, sstamp = g.wallStamp;
  var shw = g.hw.slice(), svw = g.vw.slice(), sblocked = g.blocked.slice();
  var lastBest = 0, lastScore = 0, lastDepth = 0, stable = 0;
  maxDepth = maxDepth || 30;
  for (var d = 1; d <= maxDepth; d++) {
    try {
      var sc;
      if (d >= 4 && lastScore > -2000 && lastScore < 2000) { // aspiration
        var lo = lastScore - 75, hi = lastScore + 75;
        for (;;) {
          sc = this.ab(d, lo, hi, 0, true, 0);
          if (sc <= lo) lo = -Infinity;
          else if (sc >= hi) hi = Infinity;
          else break;
        }
      } else {
        sc = this.ab(d, -Infinity, Infinity, 0, true, 0);
      }
      stable = (this.rootBest === lastBest) ? stable + 1 : 0;
      lastBest = this.rootBest; lastScore = sc; lastDepth = d;
      if (sc > MATE - 200 || sc < -(MATE - 200)) break;          // forced result
      if (!full && d >= 6 && stable >= 2 && Date.now() - t0 > timeMs * 0.1) break; // easy move
    } catch (err) {
      if (err === "time") {
        g.pawn[0] = sp0; g.pawn[1] = sp1; g.wl[0] = sw0; g.wl[1] = sw1; g.turn = sturn;
        g.hw.set(shw); g.vw.set(svw); g.blocked.set(sblocked);
        g.hashLo = slo; g.hashHi = shi; g.histLen = shist; g.lastWallPly = slwp; g.wallStamp = sstamp;
        this.cachedStamp = -1;
        break;
      }
      throw err;
    }
    if (Date.now() - t0 > timeMs * 0.6) break;
  }
  if (!lastBest) {
    this.refreshDist(0);
    this.genMoves(0, true);
    lastBest = this.moveBuf[0][0];
  }
  return { move: lastBest, score: lastScore, depth: lastDepth, nodes: this.nodes, ms: Date.now() - t0 };
};

if (typeof module !== "undefined") module.exports = { Quoridor: Quoridor, Search: Search, MATE: MATE };

