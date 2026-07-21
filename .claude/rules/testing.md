---
paths:
  - "include/**/*.hpp"
  - "tests/**/*.cpp"
  - "examples/**/*.cpp"
---

# Testing policy: verify math in Python before touching C++

This project's core risk isn't syntax — it's whether the linear algebra is
*correct* (transition matrices, process-noise terms, IMM mixing weights,
covariance updates). Compiling C++ to check a matrix equation is slow
per-iteration. Checking it in Python first is fast. Use both, in this order.

## Step 1 — verify the math in Python (fast, do this first)

Python is installed in this environment. For any new or changed equation
(a model's `transition()`/`jacobian()`/`process_noise()`, IMM mixing math,
a coordinate transform, an error-ellipse computation, etc.):

- Run **ad hoc `python3 -c "..."` commands** (or a short back-and-forth of
  them) via the Bash tool to reproduce the equation with NumPy and check it
  against a hand-computable case — the same style of check
  `tests/unit/test_kalman_cv.cpp` already does in C++ (e.g. "constant
  velocity for 0.5s at vx=2 should move px by exactly 1.0").
- **Do not create a `.py` file for this.** These are scratch, throwaway
  verification commands, not project artifacts — nothing under
  `models/`, `filters/`, or `imm/` is Python, and this project doesn't want
  a shadow Python reimplementation accumulating alongside the real one.
  If a check is worth keeping permanently, it belongs in
  `tests/unit/*.cpp`, not a saved `.py` file.
- Cross-check symmetry/positive-semi-definiteness of any new covariance or
  process-noise matrix numerically (e.g. `numpy.linalg.eigvalsh(Q)` should
  have no negative eigenvalues) before writing the C++ version — this catches
  sign errors and index-mixups in a per-axis loop far faster than a compile
  cycle would.
- For IMM mixing math specifically, hand-verify the mode-probability update
  and mixed-initial-condition formulas against a tiny 2-model, 2-step
  numeric example before trusting a change to `imm/mixing.hpp`.

## Step 2 — only then touch C++

Once the Python check confirms the math, port it to the real C++
implementation (Eigen-based, per `math/backend.hpp`), and validate with the
actual test suite:

```sh
cmake -B build -DAUGUR_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

The C++ tests (`tests/unit/`) remain the source of truth for what ships —
the Python step is a fast pre-check to catch errors before paying the
compile cost, not a substitute for the real test suite, and not something
that gets committed to the repository.

## When to skip the Python step

Pure refactors, folder/naming changes, doc updates, and anything that
doesn't touch an equation don't need a Python check — go straight to
building/testing the C++ (or skip testing entirely for a docs-only change).
