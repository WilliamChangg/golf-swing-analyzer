import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "**/dist/**",
      "**/dist-types/**",
      "**/target/**",
      "**/node_modules/**",
      // Generated from the Pydantic contracts; style is the generator's concern.
      "packages/types/src/generated/**",
      "**/.venv/**",
    ],
  },

  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,

  {
    languageOptions: {
      parserOptions: {
        projectService: {
          // Config files live outside every tsconfig `include`, so give them
          // the default project rather than silently skipping them.
          allowDefaultProject: [
            "*.config.ts",
            "*.config.js",
            "apps/desktop/*.config.ts",
          ],
        },
        tsconfigRootDir: import.meta.dirname,
      },
      globals: { ...globals.browser, ...globals.node },
    },
    rules: {
      // Unused args are allowed when prefixed with _, which keeps interface
      // implementations readable without disabling the check.
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // A floating promise in this codebase means an un-awaited engine call,
      // whose failure would vanish silently. Keep it an error.
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", fixStyle: "inline-type-imports" },
      ],
    },
  },

  {
    files: ["apps/desktop/src/**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks, "react-refresh": reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
    },
  },

  {
    // Config files are not covered by the app tsconfigs.
    files: ["*.config.{ts,js}", "**/*.config.{ts,js}"],
    ...tseslint.configs.disableTypeChecked,
  },
);
