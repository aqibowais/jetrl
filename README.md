# JetRL

A simplified jet flies a closed lap of ring gates in 3D. Proximal Policy Optimization learns the lap, and the browser shows that saved run: a chase camera, a cockpit HUD, a reward curve, and a checkpoint timeline. You can also take the stick and fly the track yourself.

## What to look at first

Open the cockpit. The server loads the strongest checkpoint already on disk and flies it in real time. Nothing trains when the page opens.

```powershell
cd "c:\Users\Aqib\Desktop\Ai projects\JetRL"
.\.venv\Scripts\Activate.ps1
python server.py
```

Open `http://127.0.0.1:8000`.

The server does not start training. It loops the saved policy with the highest mean reward, which on this disk is episode 400, at 1x.

`PROGRESS` walks the checkpoints in order, from the earliest to the latest, and then repeats. `1x`, `2x`, and `4x` only change how fast that playback runs. They do not start a new training job and they do not skip physics steps. Click a mark on the timeline to loop that one checkpoint. `COMPARE` plays two checkpoints back to back. `CHASE` sits behind the jet. `OVERVIEW` shows the whole circuit.

Episode 20 dives after the first gate. Episode 400 stays airborne much longer and clears more gates. It still does not finish a reliable full lap. Later checkpoints, through episode 1000, score worse than episode 400, so the page does not default to the last file.

`FLY` hands you the same jet. `W` and `S` change throttle, the arrow keys pitch and roll, and `A` and `D` yaw. The lap timer starts when you pass gate 1. Your best manual time sits next to the best AI lap time on the same display.

## Recorded training run

A 400-episode PPO run is already on disk under `checkpoints/` and `logs/rewards.jsonl`.

| Window | Mean episode reward |
| --- | --- |
| First 20 episodes | -34.04 |
| Last 20 episodes | 11.27 |

Early checkpoints crash after one gate. Later checkpoints often survive the full episode limit and pass two or three gates. That is a real improvement, not a finished racing line. A hand-written pilot in `test_env.py` can complete the lap in about 42 seconds, so the track is solvable. The learned policy has not reached that yet. Longer training is the next step, not a change of algorithm.

`logs/reward_curve.png` is the plot of that run.

## Run locally

Python 3.12 is what this project was built with. Python 3.10 or newer should work.

```powershell
cd "c:\Users\Aqib\Desktop\Ai projects\JetRL"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The pinned PyTorch build is the CPU wheel. On this machine, `torch==2.14` failed to load its DLL. `torch==2.5.1` from the CPU index is the version in `requirements.txt`. If `pip install -r requirements.txt` pulls a broken torch build, install that CPU wheel explicitly:

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Check the environment, with no window and no training:

```powershell
python test_env.py
```

That runs a random policy, then checks that a crash ends the episode, that flying through the wrong ring does not count, and that passing every gate in order ends the episode on a completed lap.

## Train

Training stays stopped until you confirm it. `python train.py` prints the plan and waits for `y`. Closing the prompt or typing anything else cancels the run.

```powershell
python train.py --episodes 400 --checkpoint-every 20
```

The cockpit has the same gate. `TRAIN` only opens the plan. Nothing starts until `CONFIRM START`.

Checkpoints land in `checkpoints/` every N episodes, with the episode number, mean reward, and timestamp beside the weights. Episode rewards append to `logs/rewards.jsonl`. Plot them with:

```powershell
python plot_rewards.py
```

One training process at a time. A second `train.py` or a second confirm from the page is refused while the first is alive.

## Method

The aircraft is a small rigid body, not an aerodynamic simulation. Throttle produces thrust along the nose. Lift depends on speed and angle of attack, and it tilts when the jet banks. Gravity, drag, and a little ground effect near the surface are the other forces. Banking also yaws the nose, so a roll becomes a turn. The step rate is 30 Hz.

The track is a stadium of nine rings. Altitude and heading change around the lap. A gate counts only if the jet crosses that ring in order, from behind, and inside the ring. Missing it or clipping a later ring does nothing.

The reward is dense on purpose. Closing on the next gate pays a little every step. A centered pass pays more than a pass near the rim. Hitting the ground or leaving the airspace around the track pays a large penalty and ends the episode. Finishing all nine gates pays a bonus that grows when the lap is faster.

The policy is Stable-Baselines3 PPO, a two-layer network of width 128, on CPU. Observations are in the aircraft frame: where the next two gates are, body velocity, which way is down, and the current rates. Actions are throttle, pitch, roll, and yaw, each in `[-1, 1]`.

The page plays a saved checkpoint. It does not run the optimizer. `1x` is real time. `2x` and `4x` send the same physics steps faster. They do not drop steps, so the path stays the path that checkpoint actually flew. `PROGRESS` only changes which saved file is loaded next.

## Layout

```text
jet_env.py             Gymnasium environment and the flight model
track.py               Gate positions and the centerline
test_env.py            Random policy, crash, and lap checks
train.py               PPO, confirmation prompt, checkpoints, reward log
plot_rewards.py        Reward curve from the log
compare_checkpoints.py Early versus late checkpoint rollouts
server.py              FastAPI, WebSocket stream, replay, manual flight
test_ws.py             Small client that checks the stream and the confirm gate
static/                Three.js scene, HUD, reward chart, timeline
checkpoints/           Saved policies (created by training)
logs/                  Reward log and the reward plot
```

## Scope

This is a toy aircraft in a browser, not a flight simulator and not a hardware controller. The learned policy in the saved 400-episode run survives longer and scores higher than the early policy. It does not yet fly a clean lap. Treat that gap as the result of this run, not as a claim that PPO cannot learn the track.
