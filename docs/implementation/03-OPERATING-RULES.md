# Execution, checkpoints, and resources

## Stop at each stage or 75% context utilization

This is an execution rule for every future Codex session, not a guarantee that the assistant can read its own remaining context precisely.

1. At session start read the handoff and current stage only after reviewing the shared rules. Check Git status/diff; never overwrite unrelated work.
2. Monitor the VS Code Codex context indicator when available. Begin checkpoint preparation around 65–70% used; at **75% used / 25% remaining**, stop new implementation and save a handoff. Subscription usage percentage is a different limit and must not substitute for context usage.
3. If context telemetry is unavailable, state that explicitly. Ask the user to signal the UI reading; use conservative small substeps and checkpoint at least after each substantial module or approximately 20 tool interactions. This fallback cannot guarantee an exact 75% boundary.
4. Before any large output, training launch, or long task, checkpoint proactively. Do not load the complete historical 1,800-line plan into each session.
5. On a stop, finish only the smallest safe file write/checkpoint operation, then record current state. Do not start another build stage. Save unsaved edits, but do not label unfinished code verified.
6. If training is running, record process/session ID, host, log/checkpoint paths, last step, and whether it continues. Never launch duplicate training on resume. Save full Trainer state when possible; do not abruptly kill a run merely to end a chat. The user must know if paid compute remains active.
7. Update `HANDOFF.md`, iteration log if relevant, and the Linear issue. Record tests actually run, exact commands/results, artifacts, errors, next action, current model, device, and context reading or “unavailable.”
8. Review/stage named files, commit a clear milestone or WIP checkpoint, push to the confirmed GitHub remote, verify remote SHA. If blocked, preserve local files/commit and explicitly list pending push/Linear sync. A Git commit saves locally; a push saves to GitHub.
9. End with `STOP: S<n> complete` or `STOP: context checkpoint inside S<n>`, plus the next prompt. User starts a fresh chat and says “Resume the handoff; finish only the current stage” or “Continue with the next stage.” Automatic context compaction does not authorize continuing past a stop.

## Model routing and escalation

Use OpenAI assistants throughout development. The specified MiniLM model is a separate training workload; switching Codex models neither changes that model nor supplies a GPU.

| Work | Preferred assistant tier |
|---|---|
| Experiment protocol, metric correctness, leakage checks, difficult debugging | Sol with appropriate reasoning; stronger available OpenAI model if unresolved |
| Normal implementation and reviews | Sol 6.1 (user-confirmed available in VS Code) |
| Well-specified scripts, plotting, execution of verified commands | Terra if available; otherwise Sol |
| Small documentation, status updates, formatting | Luna if available; otherwise current assistant |

“Sol,” “Terra,” and “Luna” are routing preferences, not guaranteed public model IDs or a universal intelligence ranking. The user confirmed Sol 6.1 is available in VS Code; this is the default wherever the stage plan says Sol. Record the selected model; check Terra/Luna availability only when considering those alternatives. Do not invent CLI flags or assert that this app's model menu matches VS Code. Human changes the picker if the agent cannot switch. No parallel agents are needed; work sequentially.

Failure procedure:

1. Save failing command, last/root error, environment, test case, and working diff. Distinguish code defect, missing access, memory shortage, bad data, and unmet scientific target.
2. Make one evidence-based repair and rerun the smallest relevant check. Do not repeatedly rerun a failed long training job unchanged.
3. If still unresolved or the mathematical/API issue exceeds the current tier, request a stronger **available** OpenAI model: “S<n>, step <x> failed with <error>. Tried <repair>, result <result>. Please switch to <available model> for this bounded diagnosis. Handoff saved at <path>.” The failed unit may be repeated; completed work stays preserved.
4. Higher-model retry gets a fresh bounded diagnosis and verification; after two unsuccessful repair attempts on the same blocker, stop for a human decision with evidence. Missing credentials/data require a human, not a higher model. More assistant capability cannot cure GPU OOM; change batch/device only with recorded experimental consequences.
5. Record `failed`, `interrupted`, and `completed` runs distinctly. Failed attempts do not count as the three completed training experiments. Never upgrade a subscription automatically.

## Linear tracking

Use the Linear skill for actual issue operations. Linear authenticated successfully during resource setup; the project and stage issues now exist (see RESOURCES.md). Availability in VS Code must still be checked separately. If absent, connect the supported Linear MCP integration and let the user complete OAuth; confirm current setup instructions rather than copying legacy feature flags.

Each S0–S9 issue contains stage objective, dependencies, owned files, acceptance checks, selected assistant tier, and artifact links. S6 has three experiment subtasks/checklists. Map to the workspace's actual statuses: Todo → In Progress → Done, with Blocked/label when needed. Do not mark Done solely because code exists.

At every stop update completed checks, failed checks, GitHub commit, reports/run IDs, next action, and human blockers. Maintain a local `docs/LINEAR_SYNC.md` queue if the connector fails; label it pending and synchronize on the next successful connection. Local tracking is a temporary fallback, not completion of the Linear requirement. Avoid assigning other people or notifying them without user instructions.

## GitHub saving

Stage 0 resolves owner/name and private/public visibility. Inspect `git status`, `git diff`, and `git remote -v` whenever applicable. This folder was not a Git repository during planning; do not assume the July plan's claim about a parent repo is current.

Before the first add, ignore `.venv/`, `.env*` (allow a value-free example), keys, caches, Python caches, `.DS_Store`, model/checkpoint binaries, and large prediction/embedding arrays. Keep small JSON summaries, configs, lock files, docs, and required PNGs. Verify dataset license before distributing raw data; prefer source/hash/acquisition instructions unless redistribution is confirmed.

Use explicit paths for staging, inspect the staged diff and secret/large-file exposure, commit, then push. Verify `git rev-parse HEAD` equals `git ls-remote origin refs/heads/<actual-branch>` after a successful push. Save that SHA in handoff/Linear; a push command without verification is not delivery evidence. Never force-push, reset, or remove unrelated changes to clean up a checkpoint. Model storage needs its own durable artifact manifest even though weights are excluded from Git.

## Local Mac, Colab, and RunPod pathway

Default: code and analysis in VS Code on the Mac. S1 explicitly detects CPU/MPS; S5 smoke-tests the actual training path. Start float32, moderate sequence length, batch 16. No fixed runtime promise: measure throughput. If local compute is usable, finish locally.

If local runtime/device support is inadequate, user chooses:

- **RunPod for VS Code continuity:** use a single modest CUDA GPU after checking current availability/price, cap planned hours/spend, and record shutdown responsibility. Verify public-key registration and current connection details. Connect through VS Code Remote-SSH, clone the confirmed GitHub commit into `/workspace`, install the locked environment, transfer/verify data hashes, run doctor and the same smoke test, then execute the same CLI. Never paste private keys/API credentials into chat. `/workspace` persistence depends on volume type; back up checkpoints before termination. A stopped Pod may still incur storage costs.
- **Colab alternative:** keep source of truth in GitHub, create a thin notebook to install the same package/config and invoke its CLI, select a GPU in Colab, and have the user perform interactive sign-in/storage mounting. Persist checkpoints and copy reports back before runtime expiry. The normal Colab browser workflow is sufficient; do not depend on unsupported remote SSH. Resources and session duration are not guaranteed.

Before remote launch: record commit, dataset/model hashes, environment, budget, expected duration, and artifact destination. On completion: copy results/model to durable storage, verify checksums, then stop the rental. Do not terminate storage-bearing resources until backups are confirmed. Recheck RunPod host/port after restart.

## Plus and possible upgrade

Start on Plus. Codex subscription allowance, API billing, context-window capacity, and GPU rental are separate resources. This plan requires no paid OpenAI API inference. Upgrade only if actual Codex usage limits repeatedly block the work and the account's offered Pro tier fits the user's budget. A larger subscription does not provide MiniLM training hardware or eliminate stage/context stops.

OpenAI's current help pages describe Codex access by subscription and Pro tiers including $100/month; availability and limits can change. Check the account's checkout and usage view before a purchase. Do not promise a fixed number of stages per allowance or automatic access to a named model.

## Sources checked during planning

- [Sentence Transformers training](https://sbert.net/docs/sentence_transformer/training_overview.html) and [loss definitions](https://sbert.net/docs/package_reference/sentence_transformer/losses.html): Trainer and labeled-pair loss interfaces; pin installed versions at S1.
- [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html): Mac device support; capability still requires local verification.
- [Codex with ChatGPT plans](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan), [Pro tiers](https://help.openai.com/en/articles/9793128), and [separate API billing](https://help.openai.com/en/articles/9039756-managing-billing-settings-on-chatgpt-web-and-platform): check live account offerings before spending.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): resource availability limits.
- [RunPod storage explanation](https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage): volume lifecycle and backup precautions.

Prior local RunPod notes informed the Remote-SSH pathway; account, key, port, and Pod state have not been verified for this project.
