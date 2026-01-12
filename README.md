# SchedulingOptimization
Hybrid reservation optimizer (LLM vs CP-SAT) with deterministic validator, Firestore backend, Google Forms/Sheets intake, and Cloud Run execution.

Reservation Optimization: LLM vs CP-SAT + Deterministic Validator

This repo implements a hybrid reservation-optimization pipeline for a multi-room boarding facility. The system is designed for research and evaluation: we compare an LLM planner’s decision process against a rule-based optimizer (CP-SAT) under the same constraints and test cases, and then audit both with a deterministic validator that produces rule compliance, score, and confidence.

The pipeline integrates:

- Google Forms → Google Sheets intake
- Firestore as the system of record
- Cloud Run service that generates plans (LLM or CP-SAT)
- Deterministic validation + scoring
- Sheet updates for operator visibility

Motto : validate LLM decision-making against CP-SAT using the same constraints, scenarios, and audit rules.

**High-level agentic AI architecture**
<img width="1196" height="692" alt="image (2)" src="https://github.com/user-attachments/assets/4952d033-443c-4073-b8dc-2dc7f5f638ce" />

***Phase 1***: Intake & Direct Fit (Apps Script)

1) A user submits a reservation request via Google Form.

***Apps Script:***

2) Writes a request document to Firestore: requests/<request_id>
3) Appends a row to Google Sheet (stores sheet_row back in Firestore)
4) Apps Script attempts a DIRECT FIT:
    - Checks whether the request is feasible as a single-room booking
    - Scans rooms for an available window without overlap
   If direct fit succeeds:
    - Creates booking doc(s) in Firestore: bookings/<booking_id>
    - Updates requests/<request_id>.status = confirmed
    - Updates Sheet status to confirmed
   If direct fit fails:
    - Marks requests/<request_id>.status = needs_optimization (temporary)
    - Triggers Cloud Run: POST /optimize/<request_id>

***Phase 2: Optimization (Cloud Run)***

1) Cloud Run receives request_id
2) Cloud Run builds context (“RAG”) (room + booking + request context)
3) Cloud Run runs a solver:
    - LLM mode: planner outputs a step-by-step plan
    - CP-SAT mode: OR-Tools CP-SAT finds a feasible plan under constraints
   Cloud Run validates and scores the plan:
    - Deterministic rule validator
    - Returns valid, approved, score, confidence
    - Produces a structured violations list
    - Computes economics fields (e.g., revenue / flip-cost / net-profit where enabled)
   If approved:
    - Commits the plan to Firestore
    - Creates booking docs for ConfirmNew steps (bookings/REQ-…-P1, P2, …)
    - Applies moves by updating existing booking docs (MoveExisting)
    - Patches requests/<request_id> with final status + metadata
    - Cloud Run updates the Google Sheet via Apps Script Web App callback

***Phase 3: Results Logging***

1) After a plan is produced (LLM or CP-SAT) and validated:
    - Cloud Run writes evaluation outputs back to Firestore
    - Cloud Run notifies Apps Script (callback)
    - Apps Script updates the Sheet row.

The deterministic validator is the source of truth for rule compliance, scoring, and confidence—making both solvers comparable under the same audit logic.
