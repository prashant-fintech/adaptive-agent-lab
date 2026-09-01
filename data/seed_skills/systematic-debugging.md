---
name: systematic-debugging
topic: debugging
version: 1
status: active
---
1. Reproduce the failure with the smallest possible input.
2. Read the full stack trace bottom-up; identify the first frame in project code.
3. Form one hypothesis, add one probe (print/assert), re-run.
4. Fix the root cause, not the symptom, then re-run the reproduction.
5. Add a regression test that fails without the fix.
