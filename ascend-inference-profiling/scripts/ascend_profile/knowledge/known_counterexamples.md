# Known segmentation / classification counterexamples

This file is the central registry for cases that **previously broke** the
segmentation or classification stages and were patched. It serves two
purposes:

1. **Human-readable regression log**: each entry documents a real-world
   profiling case that triggered a hard error or mis-classification in
   an earlier version of this skill, the commit that fixed it, and the
   counterexample's root cause.

2. **Test fixture source**: entries in this file are the specification
   for golden regression tests (see `deferred-work.md` §7). When you
   add a new counterexample, you MUST also add a corresponding test
   case in ``tests/test_segment_validator.py`` or a new test file.

## How to add a counterexample

1. Collect the minimal `kernel_details.csv` rows that reproduce the
   failure. Keep the file small — one rank, the narrowest row range
   that still triggers the error.
2. Add a test in `tests/` that runs the segmenter against this fixture
   and asserts the correct output.
3. Document the case here with:
   - **Profile**: model / workload / profiling configuration
   - **Symptom**: hard error or mis-classification
   - **Root cause**: why the segmenter failed
   - **Fix**: commit hash or PR reference
   - **Fixture path**: location of the test fixture file

## Counterexamples

<!-- No counterexamples recorded yet. Add entries as they are discovered. -->
