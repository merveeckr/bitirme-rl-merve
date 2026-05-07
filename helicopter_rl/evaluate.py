"""
evaluate.py — Kapsamlı değerlendirme & 3D görselleştirme
=========================================================
Tüm fazların eğitim sonuçlarını karşılaştırır, uçuş yollarını 3D olarak
çizer ve istatistiksel analiz (ödül eğrileri, başarı oranı, çarpışma oranı)
yapar.

Kullanım:
  python evaluate.py --phase 1               # sadece Faz 1
  python evaluate.py --phase 2 --episodes 8  # Faz 2, 8 bölüm
  python evaluate.py --phase 3 --n_obs 4     # Faz 3, 4 engel
  python evaluate.py --all                   # tüm fazları karşılaştır
  python evaluate.py --demo                  # hızlı demo (her fazdan 3)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation

from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helicopter_rl.envs.base_env import FlightControlEnv3D
from helicopter_rl.envs.obstacle_env import ObstacleHelicopterEnv

FIGURES_DIR = "figures"

# ─────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────

def load_model(phase: int, n_obstacles: int = 1):
    candidates = {
        1: ["models/phase1/best_model", "models/phase1/ppo_phase1_final"],
        2: ["models/phase2/best_model", "models/phase2/ppo_phase2_final"],
        3: ["models/phase3/best_model", "models/phase3/ppo_phase3_final",
            "models/phase3/stage_3C/ppo_phase3C_final"],
    }
    # Phase 2 with 2+ obstacles: prefer the 2obs fine-tuned model
    if phase == 2 and n_obstacles >= 2:
        for path in ["models/phase2_2obs/best_model", "models/phase2_2obs/ppo_p2_2obs_final"]:
            if os.path.exists(path + ".zip"):
                print(f"  ↳ 2-engel modeli yükleniyor: {path}.zip")
                return PPO.load(path)
        print("  ↳ 2-engel modeli bulunamadı, Phase 2 (1 engel) modeli kullanılıyor")
    for path in candidates.get(phase, []):
        if os.path.exists(path + ".zip"):
            return PPO.load(path)
    raise FileNotFoundError(f"No trained model for Phase {phase}. Run phase{phase}_train.py --train first.")


def make_env(phase: int, n_obstacles: int = 1):
    if phase == 1:
        return FlightControlEnv3D(max_steps=1000, wind_strength_max=3.0)
    elif phase == 2:
        return ObstacleHelicopterEnv(
            n_obstacles=n_obstacles, obstacle_radius=15.0,
            obstacle_height=120.0, safety_margin=25.0,
            max_steps=1200, wind_strength_max=4.0,
        )
    else:
        return ObstacleHelicopterEnv(
            n_obstacles=n_obstacles, obstacle_radius=15.0,
            obstacle_height=120.0, safety_margin=25.0,
            randomize_radius=True, max_steps=1500, wind_strength_max=7.0,
        )


# ─────────────────────────────────────────────────────────────────────────
# Run one evaluation
# ─────────────────────────────────────────────────────────────────────────

def run_evaluation(phase: int, n_episodes: int = 5, n_obstacles: int = 4):
    model = load_model(phase, n_obstacles=n_obstacles)
    env   = make_env(phase, n_obstacles=n_obstacles)

    trajectories, targets, starts, obs_list = [], [], [], []
    results = []

    for ep in range(n_episodes):
        obs_vec, _ = env.reset()
        starts.append(env.pos.copy())
        targets.append(env.target.copy())
        ep_obstacles = env.get_obstacles() if hasattr(env, "get_obstacles") else []

        done = False
        total_r, min_dist = 0.0, float("inf")
        step_rewards = []

        while not done:
            action, _ = model.predict(obs_vec, deterministic=True)
            obs_vec, r, terminated, truncated, info = env.step(action)
            total_r += r
            step_rewards.append(r)
            done = terminated or truncated
            if hasattr(env, "_obstacle_distances"):
                md, _ = env._obstacle_distances()
                min_dist = min(min_dist, md)

        results.append({
            "ep":           ep + 1,
            "reward":       total_r,
            "final_dist":   float(np.linalg.norm(env.target - env.pos)),
            "success":      info.get("success",   False),
            "collision":    info.get("collision",  False),
            "min_obs_dist": min_dist,
            "steps":        env.step_count,
            "step_rewards": step_rewards,
        })
        trajectories.append(env.get_trajectory())
        obs_list.append(ep_obstacles)

    return results, trajectories, targets, starts, obs_list


# ─────────────────────────────────────────────────────────────────────────
# 3D trajectory plot
# ─────────────────────────────────────────────────────────────────────────

def plot_3d(trajectories, targets, starts, obstacles=None,
            title="Helicopter Trajectories", save_path=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")
    cmap = get_cmap("tab10")

    for i, (traj, tgt, st) in enumerate(zip(trajectories, targets, starts)):
        traj = np.asarray(traj)
        c    = cmap(i % 10)

        # Colour path by altitude
        n = len(traj)
        for j in range(n - 1):
            ax.plot(traj[j:j+2, 0], traj[j:j+2, 1], traj[j:j+2, 2],
                    color=c, linewidth=1.8, alpha=0.75)

        ax.scatter(*st,         color="lime",   s=130, marker="o", zorder=6,
                   edgecolors="k", linewidth=0.6)
        ax.scatter(*traj[-1],   color=c,        s=90,  marker="x", zorder=6)
        ax.scatter(*tgt,        color="gold",   s=260, marker="*", zorder=7,
                   edgecolors="k", linewidth=0.6)
        ax.text(tgt[0], tgt[1], tgt[2] + 8, f"T{i+1}", fontsize=7,
                color="darkorange", ha="center")

    # Obstacles
    if obstacles:
        for obs in obstacles:
            _cyl(ax, obs["pos"], obs["radius"], obs["height"])

    ax.set_xlabel("X [m]", labelpad=6)
    ax.set_ylabel("Y [m]", labelpad=6)
    ax.set_zlabel("Alt [m]", labelpad=6)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    proxies = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
               markersize=10, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markersize=14, label="Target"),
    ]
    if obstacles:
        proxies.append(mpatches.Patch(color="red", alpha=0.35, label="Obstacle"))
    ax.legend(handles=proxies, loc="upper right", fontsize=9)

    plt.tight_layout()
    sp = save_path or f"{FIGURES_DIR}/{title.replace(' ', '_')[:40]}.png"
    plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 3D plot saved: {sp}")
    return sp


def _cyl(ax, centre, radius, height, color="red", alpha=0.28):
    theta  = np.linspace(0.0, 2.0 * np.pi, 32)
    z_vals = np.linspace(centre[2], centre[2] + height, 10)
    T, Z   = np.meshgrid(theta, z_vals)
    ax.plot_surface(
        centre[0] + radius * np.cos(T),
        centre[1] + radius * np.sin(T),
        Z, color=color, alpha=alpha, linewidth=0,
    )


# ─────────────────────────────────────────────────────────────────────────
# Stats dashboard
# ─────────────────────────────────────────────────────────────────────────

def plot_stats(results_per_phase: dict, save_path=None):
    """
    Çoklu faz karşılaştırma paneli:
      - Başarı oranı
      - Çarpışma oranı
      - Ortalama ödül
      - Ortalama minimum engel mesafesi
    """
    phases = sorted(results_per_phase.keys())
    labels = [f"Phase {p}" for p in phases]

    def _agg(results, key, default=0.0):
        return [np.mean([r.get(key, default) for r in results_per_phase[p]])
                for p in phases]

    success_rates   = [np.mean([r["success"]   for r in results_per_phase[p]]) * 100 for p in phases]
    collision_rates = [np.mean([r["collision"]  for r in results_per_phase[p]]) * 100 for p in phases]
    avg_rewards     = _agg(None, "reward")
    avg_min_dists   = [np.mean([r.get("min_obs_dist", 999) for r in results_per_phase[p]])
                       for p in phases]

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Phase Comparison Dashboard", fontsize=14, fontweight="bold")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    bar_kw = dict(color=["#4c72b0", "#dd8452", "#55a868"][:len(phases)],
                  edgecolor="k", linewidth=0.6)

    # 1. Success rate
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(labels, success_rates, **bar_kw)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Success Rate [%]")
    ax1.set_title("Target Reach Rate")
    for b, v in zip(bars, success_rates):
        ax1.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Collision rate
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(labels, collision_rates,
                    color=["#c44e52"] * len(phases), edgecolor="k", linewidth=0.6)
    ax2.set_ylim(0, max(collision_rates or [0]) * 1.3 + 5)
    ax2.set_ylabel("Collision Rate [%]")
    ax2.set_title("Collision Rate")
    for b, v in zip(bars2, collision_rates):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=10)

    # 3. Average reward
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(labels, avg_rewards, **bar_kw)
    ax3.set_ylabel("Average Total Reward")
    ax3.set_title("Mean Episode Reward")
    ax3.axhline(0, color="k", linewidth=0.8, linestyle="--")

    # 4. Min obstacle distance
    ax4 = fig.add_subplot(gs[1, 1])
    safe_margin = 25.0
    colors_safe = ["green" if d >= safe_margin else "red" for d in avg_min_dists]
    bars4 = ax4.bar(labels, avg_min_dists, color=colors_safe, edgecolor="k", linewidth=0.6)
    ax4.axhline(safe_margin, color="red", linewidth=1.5, linestyle="--",
                label=f"Safety margin ({safe_margin:.0f}m)")
    ax4.set_ylabel("Avg Min Distance to Obstacle [m]")
    ax4.set_title("Proximity to Obstacles")
    ax4.legend(fontsize=8)

    sp = save_path or f"{FIGURES_DIR}/phase_comparison.png"
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Stats dashboard saved: {sp}")


# ─────────────────────────────────────────────────────────────────────────
# Single-phase evaluation entry
# ─────────────────────────────────────────────────────────────────────────

def evaluate_phase(phase: int, n_episodes: int = 5, n_obstacles: int = 4):
    print(f"\n{'='*60}")
    print(f"Evaluating Phase {phase}")
    print(f"{'='*60}")

    results, trajs, targets, starts, obs_lists = run_evaluation(
        phase, n_episodes=n_episodes, n_obstacles=n_obstacles
    )

    # Print per-episode results
    for r in results:
        status = "✓ REACHED" if r["success"] else ("✗ COLLISION" if r["collision"] else "✗ timeout")
        print(f"  Ep {r['ep']:2d}: reward={r['reward']:8.1f}  "
              f"dist={r['final_dist']:6.1f}m  {status}")

    s_rate = sum(r["success"]   for r in results) / n_episodes * 100
    c_rate = sum(r["collision"] for r in results) / n_episodes * 100
    avg_r  = np.mean([r["reward"] for r in results])
    print(f"\n  Success rate   : {s_rate:.0f}%")
    print(f"  Collision rate : {c_rate:.0f}%")
    print(f"  Avg reward     : {avg_r:.1f}")

    # Plot
    phase_names = {1: "Basic Navigation", 2: "Single Obstacle", 3: "Multi-Obstacle"}
    title = f"Phase {phase} – {phase_names.get(phase, '')}"
    obs_for_plot = obs_lists[-1] if obs_lists else None

    plot_3d(trajs, targets, starts, obstacles=obs_for_plot,
            title=title,
            save_path=f"{FIGURES_DIR}/phase{phase}_trajectories.png")

    return results


# ─────────────────────────────────────────────────────────────────────────
# Flight animation (Colab / Jupyter / CLI)
# ─────────────────────────────────────────────────────────────────────────

def animate_episode(phase: int = 2, n_obstacles: int = 2, interval: int = 40,
                    save_path: str = None):
    """
    Tek bir bölümü 3D animasyon olarak döndürür.

    Colab:  from IPython.display import HTML; HTML(animate_episode(2, 2).to_jshtml())
    CLI:    python evaluate.py --animate --phase 2 --n_obs 2
            → figures/phase2_n2_obs.html  (tarayıcıda aç)
    """
    model = load_model(phase)
    env   = make_env(phase, n_obstacles=n_obstacles)

    obs_vec, _ = env.reset()
    start     = env.pos.copy()
    target    = env.target.copy()
    obstacles = env.get_obstacles() if hasattr(env, "get_obstacles") else []

    positions = [env.pos.copy()]
    done = False
    while not done:
        action, _ = model.predict(obs_vec, deterministic=True)
        obs_vec, _, terminated, truncated, info = env.step(action)
        positions.append(env.pos.copy())
        done = terminated or truncated

    traj   = np.array(positions)
    status = ("✓ ULAŞTI"   if info.get("success")
              else "✗ ÇARPIŞMA" if info.get("collision")
              else "✗ süre doldu")
    print(f"Bölüm sonucu: {status}  |  "
          f"adım: {len(traj)}  |  "
          f"son mesafe: {np.linalg.norm(env.target - env.pos):.1f}m")

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(*start,  color="lime", s=180, marker="o",
               edgecolors="k", linewidth=0.8, zorder=6)
    ax.scatter(*target, color="gold", s=350, marker="*",
               edgecolors="k", linewidth=0.8, zorder=7)
    for obs in obstacles:
        _cyl(ax, obs["pos"], obs["radius"], obs["height"])

    pts = np.vstack([traj, [start], [target]])
    pad = 30
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_zlim(max(0, pts[:, 2].min() - pad), pts[:, 2].max() + pad)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Yükseklik [m]")
    ax.set_title(f"Faz {phase} — {n_obstacles} Engel — {status}",
                 fontsize=13, fontweight="bold")

    proxies = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
               markersize=10, label="Başlangıç"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markersize=14, label="Hedef"),
    ]
    if obstacles:
        proxies.append(mpatches.Patch(color="red", alpha=0.35, label="Engel"))
    ax.legend(handles=proxies, loc="upper right", fontsize=9)

    step_size = max(1, len(traj) // 300)
    frames    = list(range(0, len(traj), step_size))

    path_line, = ax.plot([], [], [], color="royalblue", lw=2.0, alpha=0.8)
    heli_dot,  = ax.plot([], [], [], "o", color="deepskyblue", markersize=13,
                          zorder=10, markeredgecolor="navy", markeredgewidth=1.5)
    step_txt   = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                            fontsize=10, color="navy")

    def update(fi):
        i = fi + 1
        path_line.set_data(traj[:i, 0], traj[:i, 1])
        path_line.set_3d_properties(traj[:i, 2])
        heli_dot.set_data([traj[fi, 0]], [traj[fi, 1]])
        heli_dot.set_3d_properties([traj[fi, 2]])
        step_txt.set_text(f"Adım: {fi * step_size}  |  Yükseklik: {traj[fi, 2]:.0f}m")
        return path_line, heli_dot, step_txt

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    plt.close(fig)

    if save_path:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=max(1, 1000 // interval))
            print(f"  → GIF kaydedildi: {save_path}")
        else:
            html_path = save_path if save_path.endswith(".html") else save_path + ".html"
            matplotlib.rcParams["animation.embed_limit"] = 64
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(anim.to_jshtml())
            print(f"  → HTML kaydedildi: {html_path}  (tarayıcıda aç)")

    return anim


# ─────────────────────────────────────────────────────────────────────────
# Interactive Plotly animation (rotatable 3D)
# ─────────────────────────────────────────────────────────────────────────

def animate_episode_plotly(phase: int = 2, n_obstacles: int = 2, save_path: str = None):
    """
    Döndürülebilir interaktif 3D Plotly animasyonu.

    Colab:  fig = animate_episode_plotly(2, 2); fig.show()
    CLI:    python evaluate.py --plotly --phase 2 --n_obs 2
            → figures/phase2_n2_interactive.html
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly kurulu değil → !pip install plotly")
        return None

    model     = load_model(phase)
    env       = make_env(phase, n_obstacles=n_obstacles)
    obs_vec, _ = env.reset()
    start     = env.pos.copy()
    target    = env.target.copy()
    obstacles = env.get_obstacles() if hasattr(env, "get_obstacles") else []

    positions = [env.pos.copy()]
    done = False
    while not done:
        action, _ = model.predict(obs_vec, deterministic=True)
        obs_vec, _, terminated, truncated, info = env.step(action)
        positions.append(env.pos.copy())
        done = terminated or truncated

    traj   = np.array(positions)
    status = ("✓ ULAŞTI" if info.get("success")
              else "✗ ÇARPIŞMA" if info.get("collision")
              else "✗ süre doldu")
    print(f"Sonuç: {status}  |  adım: {len(traj)}  |  "
          f"son mesafe: {np.linalg.norm(env.target - env.pos):.1f}m")

    step_size   = max(1, len(traj) // 250)
    traj_frames = traj[::step_size]

    # ── Sabit izler (engeller + başlangıç + hedef) ──────────────────────
    base_traces = []

    for obs in obstacles:
        cx, cy = obs["pos"][0], obs["pos"][1]
        r, h   = obs["radius"], obs["height"]
        theta  = np.linspace(0, 2 * np.pi, 36)
        zz     = np.linspace(0, h, 12)
        T, Z   = np.meshgrid(theta, zz)
        base_traces.append(go.Surface(
            x=cx + r * np.cos(T), y=cy + r * np.sin(T), z=Z,
            colorscale=[[0, "crimson"], [1, "crimson"]],
            opacity=0.35, showscale=False,
            name="Engel", hoverinfo="skip",
        ))

    base_traces.append(go.Scatter3d(
        x=[start[0]],  y=[start[1]],  z=[start[2]],
        mode="markers",
        marker=dict(size=10, color="lime", symbol="circle",
                    line=dict(color="darkgreen", width=1)),
        name="Başlangıç",
    ))
    base_traces.append(go.Scatter3d(
        x=[target[0]], y=[target[1]], z=[target[2]],
        mode="markers",
        marker=dict(size=14, color="gold", symbol="diamond",
                    line=dict(color="darkorange", width=1)),
        name="Hedef",
    ))

    n_base = len(base_traces)

    # İlk animasyonlu izler (yol çizgisi + helikopter noktası)
    base_traces.append(go.Scatter3d(
        x=[], y=[], z=[],
        mode="lines",
        line=dict(color="royalblue", width=4),
        name="Uçuş yolu",
    ))
    base_traces.append(go.Scatter3d(
        x=[traj_frames[0, 0]], y=[traj_frames[0, 1]], z=[traj_frames[0, 2]],
        mode="markers",
        marker=dict(size=12, color="deepskyblue", symbol="circle",
                    line=dict(color="navy", width=2)),
        name="Helikopter",
    ))

    # ── Animasyon kareleri ───────────────────────────────────────────────
    frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=traj_frames[:i + 1, 0],
                    y=traj_frames[:i + 1, 1],
                    z=traj_frames[:i + 1, 2],
                    mode="lines",
                    line=dict(color="royalblue", width=4),
                ),
                go.Scatter3d(
                    x=[traj_frames[i, 0]],
                    y=[traj_frames[i, 1]],
                    z=[traj_frames[i, 2]],
                    mode="markers",
                    marker=dict(size=12, color="deepskyblue", symbol="circle",
                                line=dict(color="navy", width=2)),
                ),
            ],
            traces=[n_base, n_base + 1],
            name=str(i),
        )
        for i in range(len(traj_frames))
    ]

    fig = go.Figure(
        data=base_traces,
        frames=frames,
        layout=go.Layout(
            title=dict(text=f"Faz {phase} — {n_obstacles} Engel — {status}",
                       font=dict(size=16)),
            scene=dict(
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                zaxis_title="Yükseklik [m]",
                aspectmode="data",
            ),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=0.02, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Oynat", method="animate",
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Durdur", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(
                steps=[dict(
                    args=[[str(i)], dict(mode="immediate",
                                         frame=dict(duration=50, redraw=True))],
                    method="animate",
                    label=str(i * step_size),
                ) for i in range(len(traj_frames))],
                active=0, y=0, x=0.05, len=0.9,
                currentvalue=dict(prefix="Adım: ", visible=True, xanchor="center"),
            )],
        ),
    )

    if save_path:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        html_path = save_path if save_path.endswith(".html") else save_path + ".html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"  → İnteraktif animasyon: {html_path}")
        print(f"  → Tarayıcıda aç → ▶ Oynat → fare ile döndür")

    return fig


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helicopter RL Evaluation")
    parser.add_argument("--phase",    type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--all",      action="store_true", help="Evaluate all phases")
    parser.add_argument("--demo",     action="store_true", help="Quick demo (3 eps each)")
    parser.add_argument("--animate",  action="store_true", help="Matplotlib HTML animasyonu")
    parser.add_argument("--plotly",   action="store_true", help="İnteraktif Plotly animasyonu (döndürülebilir)")
    parser.add_argument("--gif",      action="store_true", help="HTML yerine GIF kaydet")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--n_obs",    type=int, default=2,  help="Test edilecek engel sayısı")
    parser.add_argument("--interval", type=int, default=40, help="Animasyon kare hızı (ms)")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.demo:
        results_all = {}
        for p in [1, 2, 3]:
            try:
                r = evaluate_phase(p, n_episodes=3, n_obstacles=args.n_obs)
                results_all[p] = r
            except FileNotFoundError as e:
                print(f"  Skipping Phase {p}: {e}")
        if len(results_all) > 1:
            plot_stats(results_all)

    elif args.all:
        results_all = {}
        for p in [1, 2, 3]:
            try:
                r = evaluate_phase(p, n_episodes=args.episodes, n_obstacles=args.n_obs)
                results_all[p] = r
            except FileNotFoundError as e:
                print(f"  Skipping Phase {p}: {e}")
        if len(results_all) > 1:
            plot_stats(results_all)

    elif args.plotly:
        save = os.path.join(FIGURES_DIR,
                            f"phase{args.phase}_n{args.n_obs}_interactive.html")
        animate_episode_plotly(phase=args.phase, n_obstacles=args.n_obs,
                               save_path=save)

    elif args.animate:
        ext  = ".gif" if args.gif else ".html"
        name = f"phase{args.phase}_n{args.n_obs}_obs{ext}"
        save = os.path.join(FIGURES_DIR, name)
        animate_episode(phase=args.phase, n_obstacles=args.n_obs,
                        interval=args.interval, save_path=save)

    else:
        evaluate_phase(args.phase, n_episodes=args.episodes, n_obstacles=args.n_obs)


