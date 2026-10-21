# JetRL: the full story

How the project works, why each decision was made, and how to defend it. Read it once top to bottom; the headings are for revision.

---

## 1. The project in 30 seconds

> JetRL trains a PPO policy to fly a jet through a circuit of ring gates. The first policy, trained on one fixed circuit, flies it well but fails on circuits it hasn't seen: it memorised the track. A second policy, trained on procedurally generated circuits with one memorisation shortcut removed from its input, flies unseen circuits. I picked its checkpoint on validation tracks and reported it on a separate test set. Both networks run live in the browser on a TypeScript port of the simulator, checked step by step against the Python version.

What it shows:
- I can design an RL environment and reward and get PPO to work.
- More importantly, I can test whether a policy *generalises* rather than just reporting a training curve.
- It also includes an ablation across seeds that is honest about what it can and cannot conclude.

---

## 2. Pieces

| File | Job |
| --- | --- |
| `jet_env.py` | Gymnasium environment: flight model, gates, reward, termination |
| `track.py` | The fixed circuit and the random circuit generator |
| `train.py` | PPO training, checkpoints, flags for the generalist and ablations |
| `eval_unseen.py` | Evaluate checkpoints on generated circuits (validation or test seeds) |
| `eval_ablation.py` | Best checkpoint per ablation run within a fixed step budget |
| `test_env.py` | Environment checks and a hand-written pilot used as a baseline |
| `export_web.py` | Evaluate, export weights and data for the site, write the parity fixture |
| `web/` | The site: TypeScript simulator, policy forward pass, three.js scene, HUD |

---

## 3. The flight model

A **point-mass flight-path model** at 30 Hz. The state is position, speed, and the direction of travel (heading and climb angle γ), plus a bank angle φ.

- **Pitch stick = load factor** (how many g the wings pull). A *centred* stick gives the trim load factor `cos γ / cos φ`. That is exactly what keeps the current flight path, including in a banked turn. Pulling adds up to 3.5 g and pushing removes up to 1.5 g. This is how fly-by-wire fighters behave: centred stick means "keep doing what you're doing".
- **The load factor turns the velocity:** `γ̇ = g/v (n cos φ − cos γ)` and `ψ̇ = g/v · n sin φ / cos γ`. Banking and pulling is what turns the jet, as in a real aircraft.
- **Stall:** the wing can only produce so much lift. The maximum load factor is (v / 17 m/s)², capped at 6 g. A slow jet can't pull, so it sinks.
- **Speed:** thrust − drag − g·sin γ. Drag has a parasitic part (∝ v²) and an induced part (∝ n²/v²), so hard turns bleed speed.
- **Attitude for display and observation:** the nose points along the velocity, tilted up by the angle of attack the current load factor needs (0.06 rad per g at cruise, rising as speed drops).

**Why this model (and why it was rewritten):**
- The first version integrated Euler angles and let the nose point up to 40° away from where the jet was actually going. In flight it looked nose-high and tail-low, and it was nearly unflyable by hand.
- In the flight-path model, attitude follows velocity by construction. Stalls and energy loss in turns still emerge from the physics.
- It is a deliberate simplification. There are no rotational dynamics and no CFD; the goal is something learnable that looks and flies like an aircraft.

**Controls (actions in [−1, 1]):** throttle rate, pitch (load factor), roll rate (bank is limited to ±77°), and rudder (a small direct heading rate).

---

## 4. The task

- **Fixed circuit:** a 9-gate oval, 240 m straights joined by 220 m-radius ends, with altitude rolling between 34 and 73 m. The gates are rings of 26 m radius.
- **Generated circuits** (`track.random_track`):
  - **Shape:** a stretched circle with low-frequency radial noise and a random direction of travel.
  - **Gates and height:** 7–12 gates spaced 195–225 m apart, at altitudes of 35–100 m.
  - **Flyability:** every bend must be at least 190 m in radius. At cruise speed and 60° bank the jet turns at about 104 m, so this leaves margin for lining up.
  - **Rejection:** circuits that fail a flyability check are redrawn.
- **Passing a gate:** the segment between two consecutive positions must cross the gate's plane within the ring, and gates must be taken in order.
- **Episode ends** on:
  - a lap;
  - a crash (ground, or more than 420 m from the track);
  - 20 s without a gate (the stall limit);
  - 100 s.

### Observation (22 numbers)
- The next two gates' positions in the jet's own body frame (right, up, forward).
- Velocity in the body frame.
- The gravity direction in the body frame (attitude, without angles that wrap).
- Rates, throttle, altitude, speed, angle of attack, distance to the gate, and g.
- The **gate index** (which gate is next), divided by the number of gates.

Body-frame vectors make the problem the same wherever the jet is on the map. Positions are scaled to roughly unit size.

### Reward
- **Progress:** 0.06 × (decrease in distance to the next gate). This is a dense signal pointing the right way.
- **Gate:** +12 × (0.35 + 0.65 × centring), for a pass and more for the middle.
- **Crash:** −40.
- **Stall:** −20 for 20 s without a gate.
- **Lap bonus:** +50 × (52 s / lap time), clipped to 0.45–2.4, which rewards pace.
- **Step penalty:** −0.002 per step.

**Why these terms:** the progress term is a distance-based shaping reward, which gives a learning signal long before the first gate is ever passed. Without the stall limit, an early run learned to circle safely until the time ran out. Without random-gate starts, the policy rarely practises late gates. Section 7 tests whether those two fixes were really needed.

---

## 5. PPO and the training setup

**PPO** (Proximal Policy Optimization, Stable-Baselines3) is an on-policy actor-critic method:
1. Run the current policy in 16 parallel environments for 512 steps each (8,192 transitions).
2. Estimate advantages with GAE (γ = 0.995, λ = 0.95). The high γ is there because a gate is 5–6 s ahead, about 170 steps.
3. Take 10 passes over the batch in minibatches of 1,024, with a **clipped** objective: no update may change an action's probability by more than ±20%. This keeps learning stable without TRPO's second-order maths.

**Networks:** a 128×128 tanh MLP for both actor and critic, with an initial action log-std of −1 (std ≈ 0.37).

**Why PPO:** continuous actions, cheap simulation (so on-policy sample cost doesn't matter), and it is robust to hyperparameters, which matters when the environment itself is changing during development. SAC would also work. PPO runs well on CPU with many parallel environments, and its clipped updates make the run-to-run behaviour easier to reason about.

**Settings that mattered** (found from failed runs):
- **Larger rollouts.** The first runs used 4 × 256; rollouts too short to span gate-to-gate credit didn't learn.
- **Initial noise of log σ = −1.** With more noise, the early policy crashes constantly.
- **A real resume.** The first `train.py` silently built a fresh model on every call.

**Random-gate starts:** half the training episodes start 70 m before a random gate, with earlier gates credited. The lap clock is backdated, so the lap bonus stays on the same scale.

---

## 6. Specialist vs generalist: the core experiment

- **Specialist:** trained only on the fixed circuit.
- **Generalist:** the same algorithm and network, trained with 80% generated circuits, and with the **gate-index input set to zero**.

Why zero the gate index? On one fixed track, "next gate is #6" tells you exactly where you are on the lap, so a policy can memorise a sequence of manoeuvres instead of looking at where the gate actually is. Zeroing it forces the policy to rely on the body-frame gate positions, which carry over to any circuit.

**Evaluation protocol, and why:**
- **Fixed circuit:** each checkpoint, deterministic actions, 30 seeds of start jitter, full laps from the start line.
- **Unseen circuits:**
  - **Validation** set: generated from seed 777000 onwards. The generalist's deployed checkpoint was chosen here.
  - **Test** set: from seed 888000 onwards, never used for any decision. The reported numbers come from it.
  - Choosing *and* reporting on the same tracks would overstate the result. The generalist's validation score swings between checkpoints, which is exactly when this matters.
- **Baseline:** a hand-written pilot in `test_env.py`. It points toward an aim point 30 m before the gate, banks proportionally to the heading error, holds a climb angle, and manages speed. It is evaluated on the same seeds.

**Results**

| | Fixed circuit (30 starts) | 50 unseen test circuits |
| --- | --- | --- |
| Specialist (episode 10,000, 6.8M steps) | 100% laps, 27.2 s | 46% laps, 55% of gates |
| Generalist (episode 5,000, 3.8M steps) | 100% laps, 33.2 s | **98% laps** (49/50), 99% of gates |
| Hand-written pilot | 100% laps, 39.3 s | – |

How to read it:
- **The specialist beats the hand-written pilot by 12 s on its own circuit.** It learned a real racing line, not just how to survive.
- **On new circuits it laps less than half the time.** That gap between 100% and 46% is the evidence of memorisation.
- **The generalist laps 98% of circuits it never saw**, choosing on validation tracks and reporting on separate test tracks. The one failure was a stall (20 s without a gate), not a crash.
- **Generality costs about 6 s per lap on the fixed circuit.** The generalist flies every track the same safe way.
- **The browser agrees:** on 120 tracks generated by the TypeScript port, the generalist lapped 118 and the specialist 60.
- **The generalist's validation score was unstable** between checkpoints (47% → 100% → 63% → 93%). That is common in PPO and is exactly why the checkpoint choice has to be made on data the reported number doesn't come from. Two checkpoints tied at 100% on validation; the tie was broken by lap time, picking episode 5,000.

---

## 7. Ablation

Question: did the stall limit and random-gate starts actually matter?

Each configuration was trained from scratch with identical settings (16 × 512 rollouts, 4M steps). The best checkpoint within that budget was evaluated on 30 starts.

**Results**

| Configuration | Steps to first checkpoint lapping all 30 starts | Mean lap at that checkpoint |
| --- | --- | --- |
| Stall limit + random-gate starts, seeds 7 / 21 / 33 | 1.45M / 1.31M / 1.37M | 34.8 / 38.5 / 29.7 s |
| No stall limit (seed 7) | 1.12M | 38.1 s |
| No random-gate starts (seed 7) | 1.76M | 37.1 s |

What it says:
- **Every configuration learns to lap every start within 4M steps.** With the final flight model, neither fix is *necessary*.
- **The three seeds of the full setup span 1.31–1.45M steps.** "No stall limit" (1.12M) is a little faster, and "no random-gate starts" (1.76M) is slower. Both differences come from a single seed each, so they hint at a trend (random starts may speed learning) but don't prove one.
- **History worth telling:** with the first flight model, an early run without the stall limit learned to circle safely until the time ran out, and some seeds plateaued at five of nine gates. The fixes were added for real failures; the ablation then showed they matter less once the dynamics and training setup were right.

How to talk about it: with one seed per configuration, a difference is only meaningful if it is larger than the spread between seeds of the *same* configuration. That's why the full configuration was run with three seeds. This is the kind of sentence reviewers want to see: you know what your experiment can and can't show.

---

## 8. The website

- **The policy really runs in the browser.**
  - `web/src/sim/jet.ts` is a line-for-line port of `jet_env.py` in float64.
  - `policy.ts` is the actor's forward pass: two tanh layers and a linear output, the mean action only, on weights exported as base64 float32.
  - `tracks.ts` ports the generator, so New track flies circuits no policy has trained on.
- **Parity test:** `parity.test.ts` replays an episode recorded in Python.
  - Starting from each recorded state, the TypeScript step must reproduce the next state to within 10⁻⁹ m, and the network's actions must match PyTorch to within 10⁻⁴.
  - A free-running replay of 300 steps must stay within 1 mm.
  - So the site is a faithful window onto the trained policies, not an animation.
- **Every number on the site is computed in Python** by `export_web.py`, not in the browser.
- **Modes:**
  - **Story:** a walk from early training to the generalist.
  - **Watch:** any checkpoint.
  - **New track:** specialist vs generalist on generated circuits, with a running tally.
  - **Evolution:** all checkpoints at once.
  - **Race:** fly against the policy yourself.
- **Race assist:** the arrow keys set a bank angle instead of a roll rate, letting go levels the wings and the flight path, and an autothrottle holds cruise speed. It is a layer on the *input*; the physics is identical to what the policy flies.
- **Side panel:**
  - The observation in words (gate distance, bearing and height).
  - The four actions as bars.
  - A ledger of the rewards earned.
  - The live hidden-layer activations.

---

## 9. Questions you should be ready for

**Is 98% on test circuits suspicious?**
Only if the test set leaked. The checkpoint was chosen on different tracks (validation seeds) and the test set was never used for any choice. The circuits also share a distribution (smooth loops, minimum turn radius), so "unseen" means new instances, not a different kind of track. Say that.

**Why did the specialist fail on new tracks?**
It could memorise. The gate index plus a single fixed layout means "at gate 6, do this", rather than "fly toward the gate I see". Removing the index and varying the track removed the shortcut.

**Why not train the generalist from the specialist (fine-tuning)?**
I tried it. The specialist's network relies on the gate-index input, and zeroing it at fine-tuning time made the policy collapse. Training from scratch with the input masked worked.

**Is the flight model realistic?**
It is a standard flight-path (point-mass) model with a load-factor limit, induced drag and stall. That is enough for the jet to behave like an aircraft: bank-to-turn, energy loss in turns, stall at low speed. It is not a 6-DOF rigid-body model and has no aerodynamic coefficients from data.

**Why is the generalist slower on the fixed circuit?**
It flies every circuit the same general way, while the specialist has tuned one racing line. Generality costs pace on a known track.

**How would you take this further?**
- More seeds for every configuration, with confidence intervals.
- A full 6-DOF model (JSBSim) to test whether the approach survives realistic dynamics.
- Wind and sensor noise.
- Curriculum learning on harder circuits.
- Measuring generalisation outside the training distribution (tighter turns, bigger altitude changes).

---

## 10. Words to know

- **Body frame:** coordinates relative to the aircraft (right, up, forward).
- **Load factor (g):** lift divided by weight; 1 g in level flight, more in turns and pulls.
- **Angle of attack:** the angle between the nose and the direction of travel.
- **Stall:** when the wing can't produce the lift asked for, so the jet sinks.
- **Generalisation:** performance on inputs not seen during training.
- **Validation vs test set:** one is used to make choices, the other only to report.
- **Ablation:** removing one component to measure what it contributes.
- **Parity test:** checking two implementations produce the same results.
