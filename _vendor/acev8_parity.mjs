import { execSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const bin = path.join(root, 'engine', 'target', 'release', 'titanium.exe');

for (const depth of [6, 8, 10, 12, 13]) {
  const jsOut = execSync(`node _vendor/acev8_ref_run.js ${depth}`, { cwd: root, encoding: 'utf8' });
  const js = JSON.parse(jsOut.trim().split(/\r?\n/).pop());
  const rsOut = execSync(`"${bin}" ace-bench ${depth}`, { cwd: root, encoding: 'utf8' });
  const rs = JSON.parse(rsOut.trim().split(/\r?\n/).pop());
  const ok = js.move === rs.move && js.score === rs.score && js.nodes === rs.nodes;
  console.log(
    `depth ${depth}: ${ok ? 'OK' : 'FAIL'} move=${js.move} score=${js.score} nodes=${js.nodes}`,
  );
  if (!ok) {
    console.log('  js', js);
    console.log('  rs', rs);
    process.exit(1);
  }
}
