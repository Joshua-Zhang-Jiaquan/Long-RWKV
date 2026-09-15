# Independent Code-Quality Review Receipt

- Task ID: `bg_47491641`
- Session identity: `ses_f704427c8ffePpiu656Z9hG0gw`
- Completion: normal, `2026-09-11T09:25:17.688Z`
- Verdict: **FAIL**
- Confidence: **high**
- Repository modifications by reviewer: none

## Independently confirmed

- The 53-test suite passed twice with `PYTHONHASHSEED=101` and `202`.
- Schema typing/envelopes, receipt-interruption recovery mechanics, 13-panel/3-metadata discovery, historical claim anchors, descriptor-relative publication, dynamic views, and Pyright diagnostics were independently confirmed.
- Generated ledger digest `901e21579e5ff4f77011611485ab637235b4c0662d2440f4851dab2e4c3755eb` and audit digest `e58e004a658a39aaf3e21cbfb75403c731e15b0512e71252faed7313e43ea90b` matched the repair receipt.

## Blocking findings

1. **CRITICAL:** `schema.py:166-191` and `service.py:193-206` allow recovery to publish a terminal receipt for a ledger stripped of all 13 panel and 3 metadata rows and carrying one fabricated `observed` claim. The reviewer reproduced `AUDIT_COMPLETE`, `blocked_claims=0` without forging a receipt.
2. **MAJOR:** `service.py:212` invokes ledger construction without a structured exception boundary. Deleting a panel file between enumeration and `inventory.py:169` stat produced an uncaught `FileNotFoundError` rather than structured JSON.
3. **MAJOR:** The regenerated ledger’s current receipt is beneath the explicit repair evidence root, not the default evidence root. The documented default `verify --task 1 --case happy` returned `PUBLICATION_PENDING`, exit 2, and incorrectly reported `blocked_claims=0`.

## Non-blocking findings

Double-read TOCTOU in analyze/render, read-side symlink races, mode-0600 shared artifacts, missing intermediate-directory fsync, a phantom `INVALID_COVERAGE` test status, an unused `sha256_file`, raw string statuses, and missing branch tests were reported as minor.

The full authoritative review is stored in OpenCode session `ses_f704427c8ffePpiu656Z9hG0gw`.
