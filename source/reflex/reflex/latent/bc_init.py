"""Stage 2: behavior-cloning (action-matching) initialisation of the latent pathway, with DAgger.

Student: LatentActorCritic with ENC_INIT=random (orthogonal encoder, random z columns, decoder body
copied from the Stage 1 teacher and initially blind to the box).
Teacher: the Stage 1 teacher actor reading the environment's own legacy box channels obs[160:170]
(run with ideal, noise-free box measurements so those channels are clean).
Loss: |student_action(obs[0:160], E(priv)) - teacher_action(obs[0:170])|^2, weighted 5x on frames
where the box is visible.  Only the encoder and the decoder's z columns are trained, so what z encodes
is decided by what the (frozen) decoder body needs to reproduce the teacher's actions -- learned
through the action loss, with no designed meaning per dimension.  Pure PPO from the same blind start
(no behavior cloning) did not learn to use z.
DAgger: in round 0 the teacher drives; in later rounds the student drives and the teacher labels.

Output: a latent checkpoint (model_state_dict only), passed as --warm_start_ckpt to
reflex/latent/train.py, where joint PPO then trains encoder and decoder together.

Usage (Isaac Lab Python, with the Stage 2 environment variables set):
  python bc_init.py --headless --num_envs 256 --teacher_ckpt teacher_stage1.pt --out bc_init.pt [--rounds 3 --steps 600]
"""
import argparse, json, sys
from reflex import paths as _paths
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=256); p.add_argument("--steps", type=int, default=600)
p.add_argument("--rounds", type=int, default=3); p.add_argument("--epochs", type=int, default=6); p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--final_epochs", type=int, default=0, help="after the DAgger rounds: this many extra epochs on ALL data with a cosine lr decay to 1e-5"); p.add_argument("--init_from", default=None, help="continue from a previous bc_init ckpt instead of the blind start"); p.add_argument("--seed", type=int, default=11, help="env + torch seed (independent initialisations per training seed)"); p.add_argument("--teacher_ckpt", required=True); p.add_argument("--out", required=True)
p.add_argument("--motion_file", default=_paths.MOTION_FILE)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import gymnasium as gym, torch
import whole_body_tracking.tasks  # noqa
import reflex.latent.register  # noqa
from reflex.latent.latent_env_cfg import G1BoxCatchLatentEnvCfg
from reflex.latent import latent_actor_critic as lac
assert lac.ENC_INIT == "random", "set BOX_LATENT_ENC_INIT=random"
cfg = G1BoxCatchLatentEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.commands.motion.motion_file = a.motion_file
env = gym.make("Reflex-Latent-G1-v0", cfg=cfg); raw = env.unwrapped
torch.manual_seed(a.seed); obs, _ = env.reset(seed=a.seed); n_obs = obs["policy"].shape[1]; dev = "cuda:0"; print(f"[bc] seed {a.seed}", flush=True)
old = torch.load(a.teacher_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
ac = lac.LatentActorCritic({"policy": torch.zeros(1, n_obs), "critic": torch.zeros(1, old["critic.0.weight"].shape[1])},
                           {"policy": ["policy"], "critic": ["critic"]}, 29, actor_hidden_dims=[512, 256, 128],
                           critic_hidden_dims=[512, 256, 128], activation="elu", init_noise_std=1.0, noise_std_type="scalar")
print("[bc] warm start:", ac.warm_start_latent(old), flush=True)
if a.init_from:
    ac.load_state_dict(torch.load(a.init_from, map_location="cpu", weights_only=False)["model_state_dict"]); print(f"[bc] continuing from {a.init_from}", flush=True)
ac = ac.to(dev); teacher = lac.legacy_actor_from_state(old).to(dev).eval()
for q in ac.parameters(): q.requires_grad_(False)
train_p = list(ac.e_priv_lin.parameters()) + list(ac.e_priv_res.parameters()); W0 = ac.actor[0].weight; W0.requires_grad_(True)
for q in train_p: q.requires_grad_(True)
colmask = torch.zeros_like(W0); colmask[:, ac.keep:] = 1.0
opt = torch.optim.Adam(train_p + [W0], lr=a.lr)
OBS = []

def fit(tag, epochs=None, cosine=False):
    epochs = a.epochs if epochs is None else epochs
    X = torch.cat(OBS); N = X.shape[0]; g = torch.Generator(device="cpu").manual_seed(0); perm = torch.randperm(N, generator=g)
    te, tr = perm[: N // 10], perm[N // 10:]
    with torch.no_grad(): A = torch.cat([teacher(X[i:i + 131072, :170]) for i in range(0, N, 131072)])   # chunked: a single forward over ~2M rows needs >2 GB of GPU memory
    box_on = (X[:, ac.plo + 7] > 0.5)
    def loss_on(idx, flight_only=False):
        with torch.no_grad():
            o = X[idx]; m = box_on[idx] if flight_only else torch.ones(len(idx), dtype=torch.bool, device=dev)
            d = torch.cat([(ac.actor(ac.features(o[i:i + 131072])) - A[idx[i:i + 131072]]) ** 2 for i in range(0, len(idx), 131072)])
            return float(d[m].mean()) if bool(m.any()) else float("nan")
    l0, l0f = loss_on(te), loss_on(te, True)
    for ep in range(epochs):
        if cosine:
            import math
            for g_ in opt.param_groups: g_["lr"] = 1e-5 + 0.5 * (a.lr - 1e-5) * (1 + math.cos(math.pi * ep / max(1, epochs - 1)))
        sh = tr[torch.randperm(len(tr))]
        for i in range(0, len(sh), 4096):
            j = sh[i:i + 4096]; o = X[j]
            w = 1.0 + 4.0 * box_on[j].float()[:, None]                      # box-visible frames matter most
            loss = (w * (ac.actor(ac.features(o)) - A[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); W0.grad.mul_(colmask); opt.step()
    print(f"[bc] {tag}: samples {N} (box visible {float(box_on.float().mean()):.2f})  held-out action MSE all {l0:.5f} -> {loss_on(te):.5f} | "
          f"box-visible {l0f:.5f} -> {loss_on(te, True):.5f}", flush=True)

for r in range(a.rounds):
    ac.eval()
    with torch.no_grad():
        for t in range(a.steps):
            o = obs["policy"]; OBS.append(o.clone())
            act = teacher(o[:, :170]) if (r == 0 and not a.init_from) else ac.actor(ac.features(o))
            obs, _, _, _, _ = env.step(act)
    fit(f"round {r} ({'teacher' if r == 0 else 'student'} drove)")
if a.final_epochs > 0: fit(f"final {a.final_epochs} epochs, cosine lr", epochs=a.final_epochs, cosine=True)
with torch.no_grad():
    z = ac.encode(torch.cat(OBS)[:, ac.plo:ac.phi]); print("[bc] z std per dim:", [round(float(v), 3) for v in z.std(0)],
          " |W_z| rms:", round(float(W0[:, ac.keep:].pow(2).mean().sqrt()), 4), flush=True)
for q in ac.parameters(): q.requires_grad_(True)
torch.save({"model_state_dict": ac.state_dict(), "iter": 0, "infos": {"bc_init_from": a.teacher_ckpt}}, a.out)
print(f"[bc] DONE -> {a.out}", flush=True)
env.close(); app.close()
