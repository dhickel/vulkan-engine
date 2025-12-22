# Agent Instructions

## Living Documentation
This project maintains a living documentation file located at `DEVELOPMENT.md`. This document serves as the single source of truth for the project's current state, architecture, and development direction.

**Instructions:**
1.  **Read `DEVELOPMENT.md` first:** Before starting any task, read this file to understand the current focal point, active issues, and architectural patterns.
2.  **Resume from Focal Point:** If you are not assigned a specific task, check the "Current Focal Point" section in `DEVELOPMENT.md` and continue work from there.
3.  **Update Documentation:** When you make changes to the codebase (refactoring, new features, bug fixes), you **MUST** update `DEVELOPMENT.md` to reflect these changes.
    -   Update the "Current Focal Point" if you complete a major step.
    -   Add new "Known Issues" if you discover them.
    -   Update module documentation if you change the architecture.
    -   Log your changes in a "Recent Changes" section or similar if appropriate for tracking context between agents.
4.  **Handover:** When finishing your turn, ensure `DEVELOPMENT.md` is up-to-date so the next agent can pick up exactly where you left off.

## Code Style & Comments
-   **Inline Comments:** Keep them sparse (1-2 lines). Focus on "why", not "what".
-   **Documentation Comments:** Use `///` for public structs, enums, and modules. Document the logical flow and purpose of major components heavily.
-   **Architecture:** Follow the existing patterns defined in `DEVELOPMENT.md`.

## Project Structure
-   `src/renderer/src/vulkan/`: Core Vulkan rendering logic.
-   `src/renderer/src/data/`: Data structures and asset loading.
-   `src/input/`: Input handling.
