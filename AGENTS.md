# AGENTS.md
Project: forecast daily blood donations per facility and blood group (Malaysia MoH open data).
Read docs/DONORCAST_PLAN.md before every task.

## Hard rules
- Never edit data/raw/ or src/donorcast/evaluate.py unless the task says so.
- Never use a same-day breakdown column as a feature; lagged only.
- Features are computed as of the forecast origin. No random splits.
- Never run the test-period evaluation unless the task says "final".
- All thresholds and dates live in src/donorcast/config.py.
- Never weaken or delete a test for an invariant in plan section 6.

## Protocol
1. Plan first: list files and approach. Wait for approval.
2. Change only what the task asks. Put other issues under "Follow-ups".
3. Before finishing: ruff, pytest. Notebooks must run top to bottom.
4. Final message: files changed, how to verify, follow-ups.
