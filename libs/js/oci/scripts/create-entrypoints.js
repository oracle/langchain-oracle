import * as fs from "fs";
import * as path from "path";

const entrypoints = {
  index: "index",
};

const updateJsonFile = (relativePath, updateFunction) => {
  const contents = fs.readFileSync(relativePath).toString();
  const res = updateFunction(JSON.parse(contents));
  fs.writeFileSync(relativePath, JSON.stringify(res, null, 2) + "\n");
};

const generateFiles = () => {
  const files = [...Object.entries(entrypoints), ["index", "index"]].flatMap(
    ([key, value]) => {
      const nrOfDots = key.split("/").length - 1;
      const relativePath = "../".repeat(nrOfDots) || "./";
      const compiledPath = `${relativePath}dist/${value}.js`;
      const declarationPath = `${relativePath}dist/${value}.js`;
      return [
        [
          `${key}.cjs`,
          `module.exports = require('${relativePath}dist/${value}.cjs');`,
        ],
        [`${key}.js`, `export * from '${compiledPath}'`],
        [`${key}.d.ts`, `export * from '${compiledPath}'`],
        [`${key}.d.cts`, `export * from '${declarationPath}'`],
      ];
    }
  );

  return Object.fromEntries(files);
};

const updateConfig = () => {
  const generatedFiles = generateFiles();
  const filenames = Object.keys(generatedFiles);

  updateJsonFile("./package.json", (json) => ({
    ...json,
    exports: Object.assign(
      Object.fromEntries(
        [...Object.keys(entrypoints)].map((key) => {
          const entryPoint = {
            types: `./${key}.d.ts`,
            import: `./${key}.js`,
            require: `./${key}.cjs`,
          };

          return [key === "index" ? "." : `./${key}`, entryPoint];
        })
      ),
      { "./package.json": "./package.json" }
    ),
    files: ["dist/", ...filenames],
  }));

  Object.entries(generatedFiles).forEach(([filename, content]) => {
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, content);
  });

  fs.writeFileSync("./.gitignore", filenames.join("\n") + "\n");
};

const cleanGenerated = () => {
  const filenames = Object.keys(generateFiles());
  filenames.forEach((fname) => {
    try {
      fs.unlinkSync(fname);
    } catch {
      // Ignore missing generated files.
    }
  });
};

const command = process.argv[2];

if (command === "pre") {
  cleanGenerated();
} else {
  updateConfig();
}
