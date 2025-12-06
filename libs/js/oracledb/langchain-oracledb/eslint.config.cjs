// eslint.config.cjs — CJS flat config compatible with ESLint v9 + TS-ESLint v8

const js = require("@eslint/js");
const tseslint = require("typescript-eslint");
const prettier = require("eslint-config-prettier");
const noInstanceOf = require("eslint-plugin-no-instanceof");
const importPlugin = require("eslint-plugin-import");

module.exports = [
  js.configs.recommended,
  prettier,
  ...tseslint.configs.recommended,

  {
    files: ["src/**/*.{ts,js,tsx,jsx}"],

    ignores: [
      "src/utils/@cfworker",
      "src/utils/fast-json-patch",
      "src/utils/js-sha1",
      ".eslintrc.cjs",
      "scripts",
      "node_modules",
      "dist",
      "dist-cjs",
      "*.js",
      "*.cjs",
      "*.d.ts",
    ],

    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      parser: tseslint.parser,              // ✅ correct parser object
      parserOptions: {
        project: "./tsconfig.json",
      },
    },

    plugins: {
      "@typescript-eslint": tseslint.plugin, // ✅ correct plugin object
      "no-instanceof": noInstanceOf,
      import: importPlugin,
    },

    rules: {
      "no-process-env": 2,
      "no-instanceof/no-instanceof": 2,

      "@typescript-eslint/explicit-module-boundary-types": 0,
      "@typescript-eslint/no-empty-function": 0,
      "@typescript-eslint/no-shadow": 0,
      "@typescript-eslint/no-empty-interface": 0,
      "@typescript-eslint/no-use-before-define": [
        "error",
        { functions: false, classes: true, variables: true },
      ],
      "@typescript-eslint/no-unused-vars": ["warn", { args: "none" }],
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-misused-promises": "error",

      camelcase: 0,
      "class-methods-use-this": 0,
      "import/extensions": [2, "ignorePackages"],
      "import/no-extraneous-dependencies": [
        "error",
        { devDependencies: ["**/*.test.ts"] },
      ],
      "import/no-unresolved": 0,
      "import/prefer-default-export": 0,

      "keyword-spacing": "error",
      "max-classes-per-file": 0,
      "max-len": 0,
      "no-await-in-loop": 0,
      "no-bitwise": 0,
      "no-console": 0,
      "no-restricted-syntax": 0,
      "no-shadow": 0,
      "no-continue": 0,
      "no-void": 0,
      "no-underscore-dangle": 0,
      "no-use-before-define": 0,
      "no-useless-constructor": 0,
      "no-return-await": 0,
      "consistent-return": 0,
      "no-else-return": 0,
      "func-names": 0,
      "no-lonely-if": 0,
      "prefer-rest-params": 0,
      "new-cap": ["error", { properties: false, capIsNew: false }],
    },
  },
];
