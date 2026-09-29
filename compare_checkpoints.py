"""Compare an early checkpoint with a late one."""

from stable_baselines3 import PPO

from jet_env import JetEnv


def run(path: str, seed: int) -> None:
    model = PPO.load(path, device="cpu")
    env = JetEnv()
    obs, _info = env.reset(seed=seed)
    terminated = False
    truncated = False
    info = {}
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
    gates = sum(info["gates_cleared"])
    print(
        f"{path} seed {seed}  {info['reason']:14s}  "
        f"reward {env.episode_reward:7.1f}  gates {gates}  steps {info['steps']}"
    )
    env.close()


if __name__ == "__main__":
    for checkpoint in ("checkpoints/ep_00020.zip", "checkpoints/ep_00400.zip"):
        for seed in (0, 1, 2):
            run(checkpoint, seed)
        print("---")
