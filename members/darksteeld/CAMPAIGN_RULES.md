# Process rules carried over from the ROGII campaign

Source: `~/Documents/GitHub/rogii-wellbore/review_jul27/POSTMORTEM.md` §6, §8.5, §9 — rules that
were paid for with a three-week campaign. They are transferred verbatim in substance, restated for
this competition. Nothing here is new theory; each line has a measured failure behind it.

## The ship gate (four parts, all must pass)

A change ships only if:
1. pooled metric improves by at least the noise floor,
2. **no** validation block gets worse beyond a pre-set tolerance,
3. worst-case instance does not blow up,
4. ≥4 of 6 folds improve.

A pooled number alone converted **at least four redistributions into apparent wins** in the ROGII
campaign. The per-fold and per-instance criteria caught every one.

## The rules

1. **Compute the oracle ceiling before building the lane.** Zero training, minutes of compute: it
   either kills the lane or prices its maximum. *Single highest-leverage habit the last campaign
   produced.* Every lane that got an oracle-first treatment died cheap; every lane that got built
   first died expensive.
2. **Measure the evaluation channel's own noise floor before reading any delta from it.** On ROGII
   this took 14 unchanged resubmissions. Here: submissions are 5/day — spend 1–2 early on an
   identical resubmission and learn the floor. Every LB reading afterwards is priced in that unit.
   Two "findings" were retroactively voided by this on ROGII.
3. **Pre-register hypothesis AND control before computing.** A control invented after seeing the
   result is not believed, and rightly.
4. **Recompute headlines from artifacts; treat prose as a claim.** Ten prose-vs-artifact drifts in
   one week on ROGII, all resolving to the artifact. Machine-written JSON next to a hand-written
   summary is the specific hazard.
5. **Build the second decorrelated member first, not last.** Variance reduction scales with (1−ρ);
   a new member is worth what it *disagrees* with. The largest measured gain of the ROGII campaign
   was a blend, and it arrived in week three. Every solution above our level was a blend.
6. **Design blend weights against the *hidden* correlation, not the local one.** Local ρ = 0.3963,
   hidden ρ = 0.7974 — that one number capped a whole lane, and it was recoverable by algebra from
   three public scores we already had.
7. **Share of error ≠ share of recoverable gain.** 56.6% of ROGII's error was "level"; 0% of the
   recoverable gain was. Distinguishing test: a leave-block-out recompute of the correction, minutes.
8. **Stop a failing family after failure three, not failure nine.** Failures 5–9 cost a week to
   re-establish what failures 1, 2, 4 already said. Run the killing test (FWER permutation) first,
   not last.
9. **When a fit looks too good, run the adversarial pass.** Ask: can I reproduce this number without
   fitting anything? On ROGII an r = 0.53 "feature" was an algebraic identity — the regression was
   ensembling, not predicting.
10. **Small subsets lie.** Do not read a verdict off a subset that cannot carry it.
11. **Calibrate uncertainty before improving the estimate.** The second-largest ROGII gain came from
    replacing a variance model while keeping the estimate bit-identical. Ask early: does this model
    know when it is wrong?
12. **Pick by true out-of-sample CV, never by the public number** — and **check what is currently
    selected**. The platform's default selection would have chosen our worst private submission.
    Caught by looking, five days out. Here: ≤2 solutions per task, auto-selected by test metric if
    you do nothing, deadline **12:00 31.08.2026**.
    **12b. The currency rule covers every elimination, not just the final pick.** If validation is
    CV, then every kill/keep threshold — lane-closure bands, early-stop gates, "does it transfer"
    tests — is written in CV units; the public number enters only as a pipeline sanity check. This
    is the rule ROGII did not have: a band denominated in **public** feet (">8.3") closed the
    fusion lane on 27–28.07, and by pick time on 02.08 that row was not even a candidate — it
    finished private 8.993 against a bronze cut of 9.164. The pick discipline was executed
    correctly on a candidate set that had already lost the medal. **Operational test:** name every
    step at which a candidate can vanish from the list, and check which currency that step counts
    in. It is usually the public one.
13. **Measure, do not estimate.** Especially for cost, duration and throughput of anything rented.
14. **Read the host's own posts and the rules before spending anything on a question about them.**
    An admin post three months old settled a question we spent a submission on.
15. **Put the compute guard outside the process that can die.** A watchdog inside the session dies
    with the session. That cost $13.50 of idle GPU burn.
16. **Report negatives as results.** A closed lane is an output.
17. **Label inference as inference.** `[V]` vs `[INF]` in every document.

## Competition-specific additions for E-CUP 2026

18. **Licensing is a hard gate, checked before a model is downloaded.** Rules §2.8 requires a license
    permitting *free commercial use* — stricter than "open source". Every model/library entering the
    container gets its license recorded in `ecup_state.json` before any training run.
19. **The metric can change unilaterally (§4.5.1).** Re-read the task page daily; do not overfit to
    an assumed metric.
20. **This repo stays private.** §2.12 bans public discussion of the tasks and publication of any
    part of a solution without written consent. No public GitHub remote.
21. **Team composition is immutable once created, and one ineligible member disqualifies the team.**
    Verify eligibility documents *before* forming the team.
