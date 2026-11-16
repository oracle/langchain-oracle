import "dotenv/config";
import {
  configDefaults,
  defineConfig,
  type UserConfigExport,
} from "vitest/config";

export default defineConfig((env) => {
  const { ORACLE_USERNAME, ORACLE_PASSWORD, ORACLE_DSN } = process.env;
  const hasOracleCredentials = ORACLE_USERNAME && ORACLE_PASSWORD && ORACLE_DSN;
  const excludeIntegrationTests = hasOracleCredentials
    ? configDefaults.exclude
    : [...configDefaults.exclude, "**/*.int.test.ts"];

  const common: UserConfigExport = {
    test: {
      environment: "node",
      hideSkippedTests: true,
      testTimeout: 30_000,
      maxWorkers: 0.5,
      exclude: excludeIntegrationTests,
      setupFiles: [import.meta.resolve("dotenv/config")],
      passWithNoTests: false,
    },
  };

  if (env.mode === "standard-unit") {
    return {
      test: {
        ...common.test,
        testTimeout: 100_000,
        exclude: excludeIntegrationTests,
        include: ["**/*.standard.test.ts"],
        name: "standard-unit",
        environment: "node",
      },
    };
  }

  if (env.mode === "standard-int") {
    return {
      test: {
        ...common.test,
        testTimeout: 100_000,
        exclude: excludeIntegrationTests,
        passWithNoTests: false,
        include: ["**/*.standard.int.test.ts"],
        name: "standard-int",
        environment: "node",
      },
    };
  }

  if (env.mode === "int") {
    return {
      test: {
        ...common.test,
        globals: false,
        testTimeout: 100_000,
        exclude: excludeIntegrationTests,
        passWithNoTests: false,
        include: ["**/*.int.test.ts"],
        name: "int",
        environment: "node",
      },
    };
  }

  return {
    test: {
      ...common.test,
      environment: "node",
      include: configDefaults.include,
      typecheck: { enabled: true },
      passWithNoTests: true,
    },
  };
});
