import fs from "fs";
import path from "path";

function walk(dir, handler) {
  for (const file of fs.readdirSync(dir)) {
    const full = path.join(dir, file);
    if (fs.statSync(full).isDirectory()) {
      walk(full, handler);
    } else {
      handler(full);
    }
  }
}

const typesDir = "dist-types-cjs";
const outDir = "dist";

if (!fs.existsSync(typesDir)) {
  console.warn(`Directory not found: ${typesDir}`);
  process.exit(0);
}

walk(typesDir, (fullPath) => {
  if (fullPath.endsWith(".d.ts")) {
    fs.renameSync(fullPath, fullPath.replace(/\.d\.ts$/, ".d.cts"));
  }
});

walk(typesDir, (fullPath) => {
  if (fullPath.endsWith(".d.cts")) {
    const relative = path.relative(typesDir, fullPath);
    const destination = path.join(outDir, relative);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    const contents = fs
      .readFileSync(fullPath, "utf8")
      .replace(
        /(\bfrom\s+["'])(\.\.?\/[^"']+)\.js(["'])/g,
        "$1$2.cjs$3"
      )
      .replace(
        /(\bimport\s*\(\s*["'])(\.\.?\/[^"']+)\.js(["'])/g,
        "$1$2.cjs$3"
      );
    fs.writeFileSync(destination, contents);
  }
});

fs.rmSync(typesDir, { recursive: true, force: true });
