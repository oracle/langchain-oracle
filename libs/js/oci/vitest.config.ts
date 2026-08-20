import { configDefaults, defineConfig, type UserConfigExport } from "vitest/config";

export default defineConfig((env) => {
  const common: UserConfigExport = {
    test: {
      environment: "node",
      testTimeout: 30_000,
      maxWorkers: "50%",
      exclude: configDefaults.exclude,
      setupFiles: ["./scripts/vitest-setup.ts"],
      passWithNoTests: false,
    },
  };

  if (env.mode === "int") {
    return {
      test: {
        ...common.test,
        include: ["**/*.int.test.ts"],
        testTimeout: 100_000,
        name: "int",
      },
    };
  }

  return {
    test: {
      ...common.test,
      include: ["**/*.test.ts"],
      exclude: [...configDefaults.exclude, "**/*.int.test.ts"],
      typecheck: { enabled: true },
    },
  };
});
