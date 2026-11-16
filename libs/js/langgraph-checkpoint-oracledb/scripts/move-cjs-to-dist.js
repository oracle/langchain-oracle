import { resolve, dirname, parse, format } from "node:path";
import { readdir, readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

function abs(relativePath) {
  return resolve(dirname(fileURLToPath(import.meta.url)), relativePath);
}

async function moveAndRename(source, dest) {
  for (const file of await readdir(abs(source), { withFileTypes: true })) {
    if (file.isDirectory()) {
      await moveAndRename(`${source}/${file.name}`, `${dest}/${file.name}`);
    } else if (file.isFile()) {
      const parsed = parse(file.name);
      const sourcePath = abs(`${source}/${file.name}`);
      const destinationDir = abs(dest);

      await mkdir(destinationDir, { recursive: true });

      if (parsed.ext === ".js") {
        const renamed = format({ name: parsed.name, ext: ".cjs" });
        const content = await readFile(sourcePath, "utf8");
        const rewrittenRequires = content.replace(
          /require\("(\..+?).js"\)/g,
          (_, p1) => {
            return `require("${p1}.cjs")`;
          },
        );
        const rewrittenSourceMapping = rewrittenRequires.replace(
          /sourceMappingURL=(.+?)\.js\.map/g,
          (match, p1) => {
            return `sourceMappingURL=${p1}.cjs.map`;
          },
        );
        const rewritten = rewrittenSourceMapping.replace(
          /"file":"(.+?)\.js"/g,
          (match, p1) => {
            return `"file":"${p1}.cjs"`;
          },
        );

        await writeFile(`${destinationDir}/${renamed}`, rewritten, "utf8");
      } else if (parsed.ext === ".map" && parsed.name.endsWith(".js")) {
        const renamed = format({
          name: parsed.name.slice(0, -3),
          ext: ".cjs.map",
        });
        const content = JSON.parse(await readFile(sourcePath, "utf8"));

        if (typeof content.file === "string" && content.file.endsWith(".js")) {
          content.file = content.file.replace(/\.js$/, ".cjs");
        }

        if (Array.isArray(content.sources)) {
          content.sources = content.sources.map((sourceMapEntry) => {
            if (
              typeof sourceMapEntry === "string" &&
              sourceMapEntry.endsWith(".js")
            ) {
              return sourceMapEntry.replace(/\.js$/, ".cjs");
            }

            return sourceMapEntry;
          });
        }

        await writeFile(
          `${destinationDir}/${renamed}`,
          JSON.stringify(content),
        );
      }
    }
  }
}

moveAndRename("../dist-cjs", "../dist").catch((err) => {
  console.error(err);
  process.exit(1);
});
