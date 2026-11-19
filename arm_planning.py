"""
Moving robotic arm: path planning in JOINT space with A*, RRT and RRT*
======================================================================
A planar arm (2 links by default) must move from a START pose to a GOAL pose
without any LINK touching the red obstacle.

  * Planning happens in joint-angle space (configuration space).
  * Collision check = every link (a line segment) vs the obstacle (a circle).
  * The animation shows the arm following each planned path.
  * Timing uses a quintic (5th-degree) curve, so the arm starts and stops
    smoothly: path = the route, trajectory = route + timing.

Run:    python3 arm_planning.py
Needs:  numpy, matplotlib
Output: arm_motion.gif, arm_paths.png, cspace.png (and a live window)

Presentation idea: with 2 joints A* works (the grid is small). Set
LINKS = [1.0, 0.8, 0.6] to use 3 joints and watch A* give up while RRT/RRT*
still work: that is the curse of dimensionality, live.
"""
import heapq
import itertools
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

# ----------------------------------------------------------------------
# Settings you can change
# ----------------------------------------------------------------------
LINKS = np.array([1.0, 1.0])            # link lengths (number of links = number of joints)
N = len(LINKS)
LIM = np.array([[-90.0, 180.0]] + [[-150.0, 150.0]] * (N - 1))   # joint limits (degrees)
START_Q = np.array([-39.0, 48.0] + [0.0] * (N - 2))              # start pose (degrees)
GOAL_Q = np.array([99.0, -51.0] + [0.0] * (N - 2))               # goal pose (degrees)
OBSTACLES = np.array([[0.9, 1.1, 0.35]])                         # x, y, radius
MARGIN = 0.05           # safety margin: links must stay this far from the obstacle
SEED = 1                # RRT and RRT* are random; change the seed for a new run
SHORTCUT = False        # True = try to delete unneeded waypoints (RRT, RRT*)
SAVE_GIF = True


# ----------------------------------------------------------------------
# Arm geometry and collision checking (vectorised over many poses at once)
# ----------------------------------------------------------------------
def fk_points(Q):
    """Forward kinematics. Q: (M, N) joint angles in degrees.
    Returns (M, N+1, 2): x, y of the base and of every joint / the hand."""
    Q = np.atleast_2d(Q)
    ang = np.radians(np.cumsum(Q, axis=1))
    pts = np.zeros((Q.shape[0], N + 1, 2))
    pts[:, 1:, 0] = np.cumsum(LINKS * np.cos(ang), axis=1)
    pts[:, 1:, 1] = np.cumsum(LINKS * np.sin(ang), axis=1)
    return pts


def clearance(Q):
    """Smallest distance between any link and any obstacle surface, per pose.
    Negative means a link is inside an obstacle (collision)."""
    P = fk_points(Q)
    A, B = P[:, :-1, :], P[:, 1:, :]                 # each link = segment A->B
    AB = B - A
    C, R = OBSTACLES[:, :2], OBSTACLES[:, 2]
    AC = C[None, None, :, :] - A[:, :, None, :]
    t = np.clip((AC * AB[:, :, None, :]).sum(-1) / (AB ** 2).sum(-1)[:, :, None], 0, 1)
    closest = A[:, :, None, :] + t[..., None] * AB[:, :, None, :]
    d = np.linalg.norm(C[None, None, :, :] - closest, axis=-1) - R[None, None, :]
    return d.min(axis=(1, 2))


def valid(Q):
    """True for poses inside the joint limits with no link too close to an obstacle."""
    Q = np.atleast_2d(Q)
    inside = np.all((Q >= LIM[:, 0]) & (Q <= LIM[:, 1]), axis=1)
    return inside & (clearance(Q) > MARGIN)


def edge_free(a, b, step=2.0):
    """Is the straight joint-space move a -> b free of collisions? (checked every `step` degrees)"""
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    Q = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
    return bool(valid(Q).all())


# ----------------------------------------------------------------------
# Planner 1: A*  (grid search over joint angles, f = g + h)
# ----------------------------------------------------------------------
def astar(start, goal, res, max_expansions):
    lo = LIM[:, 0]
    dims = np.floor((LIM[:, 1] - lo) / res).astype(int) + 1

    def to_idx(q):
        return tuple(int(v) for v in np.round((q - lo) / res))

    def to_q(i):
        return lo + np.array(i) * res

    cache = {}

    def free(i):
        if i not in cache:
            inb = all(0 <= i[k] < dims[k] for k in range(N))
            cache[i] = bool(inb and valid(to_q(i))[0])
        return cache[i]

    s, g = to_idx(start), to_idx(goal)
    moves = [m for m in itertools.product((-1, 0, 1), repeat=N) if any(m)]

    def h(i):  # straight-line distance to the goal (never overestimates)
        return float(np.linalg.norm((np.array(i) - np.array(g)) * res))

    open_heap = [(h(s), h(s), 0.0, s)]
    gcost, parent, closed = {s: 0.0}, {s: None}, set()
    expanded = 0
    while open_heap:
        _, _, cost, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == g:
            path = []
            while cur is not None:
                path.append(to_q(cur))
                cur = parent[cur]
            path = path[::-1]
            path[0], path[-1] = start.copy(), goal.copy()
            return path, expanded
        if expanded > max_expansions:
            return None, expanded
        for m in moves:
            nb = tuple(cur[k] + m[k] for k in range(N))
            if not free(nb):
                continue
            ng = cost + res * float(np.linalg.norm(m))
            if ng < gcost.get(nb, 1e18):
                gcost[nb] = ng
                parent[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), h(nb), ng, nb))
    return None, expanded


# ----------------------------------------------------------------------
# Planner 2: RRT  (random tree, first path found)
# ----------------------------------------------------------------------
def rrt(start, goal, rng, step=6.0, max_iter=8000, goal_bias=0.08):
    P = np.zeros((max_iter + 2, N))
    parent = np.full(max_iter + 2, -1)
    P[0] = start
    n = 1
    for _ in range(max_iter):
        q = goal if rng.random() < goal_bias else rng.uniform(LIM[:, 0], LIM[:, 1])
        i = int(np.argmin(((P[:n] - q) ** 2).sum(1)))
        d = q - P[i]
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        new = P[i] + d / L * min(step, L)
        if not edge_free(P[i], new):
            continue
        P[n], parent[n] = new, i
        n += 1
        if np.linalg.norm(new - goal) <= step and edge_free(new, goal):
            P[n], parent[n] = goal, n - 1
            n += 1
            path, k = [], n - 1
            while k != -1:
                path.append(P[k].copy())
                k = parent[k]
            return path[::-1], n
    return None, n


# ----------------------------------------------------------------------
# Planner 3: RRT*  (RRT + best parent + rewire -> shorter paths)
# ----------------------------------------------------------------------
def rrt_star(start, goal, rng, step=6.0, max_iter=1500, goal_bias=0.08,
             radius=25.0, goal_tol=9.0):
    P = np.zeros((max_iter + 2, N))
    parent = np.full(max_iter + 2, -1)
    cost = np.zeros(max_iter + 2)
    children = [set() for _ in range(max_iter + 2)]
    P[0] = start
    n = 1
    for _ in range(max_iter):
        q = goal if rng.random() < goal_bias else rng.uniform(LIM[:, 0], LIM[:, 1])
        i = int(np.argmin(((P[:n] - q) ** 2).sum(1)))
        d = q - P[i]
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        new = P[i] + d / L * min(step, L)
        if not edge_free(P[i], new):
            continue
        dist = np.linalg.norm(P[:n] - new, axis=1)
        near = np.where(dist <= radius)[0]

        # (a) choose the best parent: cheapest cost-from-start that has a free edge
        best = i
        for j in near[np.argsort(cost[near] + dist[near])]:
            if j == i or edge_free(P[j], new):
                best = int(j)
                break
        k = n
        P[k], parent[k] = new, best
        cost[k] = cost[best] + dist[best]
        children[best].add(k)
        n += 1

        # (b) rewire: if going through the new node is cheaper, switch parent
        for j in near:
            j = int(j)
            if j == best or j == 0:
                continue
            c = cost[k] + dist[j]
            if c < cost[j] - 1e-9 and edge_free(new, P[j]):
                children[parent[j]].discard(j)
                parent[j] = k
                children[k].add(j)
                delta = c - cost[j]
                cost[j] = c
                stack = list(children[j])
                while stack:                     # push the saving to all descendants
                    m = stack.pop()
                    cost[m] += delta
                    stack.extend(children[m])

    dg = np.linalg.norm(P[:n] - goal, axis=1)
    best_j, best_total = None, 1e18
    for j in np.where(dg <= goal_tol)[0]:
        total = cost[j] + dg[j]
        if total < best_total and edge_free(P[j], goal):
            best_j, best_total = int(j), total
    if best_j is None:
        return None, n
    path, k = [goal.copy()], best_j
    while k != -1:
        path.append(P[k].copy())
        k = parent[k]
    return path[::-1], n


# ----------------------------------------------------------------------
# Helpers: shortcut, densify, trajectory timing, metrics
# ----------------------------------------------------------------------
def shortcut(path, rng, tries=300):
    """Delete waypoints whenever two non-neighbouring ones can be joined directly."""
    path = list(path)
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i >= 2 and edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def densify(path, step=1.0):
    pts = []
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for t in np.linspace(0.0, 1.0, n + 1)[:-1]:
            pts.append(a + t * (b - a))
    pts.append(path[-1])
    return np.array(pts)


def make_trajectory(path, speed=2.5):
    """Path -> poses per animation frame. Quintic time scaling: speed and
    acceleration are zero at the start and the end (smooth start and stop)."""
    dense = densify(path, 1.0)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    total = arc[-1]
    frames = max(30, int(total / speed))
    t = np.linspace(0.0, 1.0, frames)
    s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
    u = s * total
    return np.column_stack([np.interp(u, arc, dense[:, k]) for k in range(N)])


def metrics(path):
    dense = densify(path, 1.0)
    hand = fk_points(dense)[:, -1, :]
    return {
        "waypoints": len(path),
        "joint_len": float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()),
        "hand_len": float(np.linalg.norm(np.diff(hand, axis=0), axis=1).sum()),
        "min_clear": float(clearance(dense).min()),
    }


# ----------------------------------------------------------------------
# Main: plan with all three, print a table, draw + animate
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    assert valid(START_Q)[0] and valid(GOAL_Q)[0], "start or goal pose is in collision"

    res, max_exp = (3.0, 60000) if N <= 2 else (6.0, 3000)
    results = {}

    t0 = time.time()
    p, info = astar(START_Q, GOAL_Q, res, max_exp)
    results["A*"] = (p, time.time() - t0,
                     f"{info} grid cells searched" if p is not None
                     else f"gave up after {info} cells")
    t0 = time.time()
    p, info = rrt(START_Q, GOAL_Q, rng)
    results["RRT"] = (p, time.time() - t0, f"{info} tree nodes")
    t0 = time.time()
    star_kw = {} if N <= 2 else dict(max_iter=5000, radius=45.0, goal_tol=18.0)  # more joints = more samples
    p, info = rrt_star(START_Q, GOAL_Q, rng, **star_kw)
    results["RRT*"] = (p, time.time() - t0, f"{info} tree nodes")

    if SHORTCUT:
        for name in ("RRT", "RRT*"):
            if results[name][0] is not None:
                results[name] = (shortcut(results[name][0], rng),) + results[name][1:]

    # ---- table
    print(f"\n{N}-joint arm, joint limits {LIM.tolist()}")
    print(f"{'Method':<6} {'time(s)':>8} {'waypts':>7} {'joint path(deg)':>16} "
          f"{'hand path':>10} {'min clearance':>14}   note")
    for name, (p, secs, note) in results.items():
        if p is None:
            print(f"{name:<6} {secs:>8.2f} {'-':>7} {'-':>16} {'-':>10} {'-':>14}   {note}")
        else:
            m = metrics(p)
            print(f"{name:<6} {secs:>8.2f} {m['waypoints']:>7d} {m['joint_len']:>16.1f} "
                  f"{m['hand_len']:>10.2f} {m['min_clear']:>14.3f}   {note}")

    # ---- trajectories for the animation
    trajs, handpath, titles = {}, {}, {}
    for name, (p, secs, note) in results.items():
        if p is None:
            titles[name] = f"{name}: gave up\n({note})"
            continue
        m = metrics(p)
        trajs[name] = make_trajectory(p)
        handpath[name] = fk_points(densify(p, 1.0))[:, -1, :]
        titles[name] = (f"{name}: {m['joint_len']:.0f}° of joint motion\n"
                        f"min clearance {m['min_clear']:.2f}")
    pts = {k: fk_points(v) for k, v in trajs.items()}
    hands = {k: v[:, -1, :] for k, v in pts.items()}      # hand position at every frame
    n_frames = (max(len(v) for v in trajs.values()) if trajs else 1) + 25

    # ---- figure with one panel per planner
    reach = float(LINKS.sum())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    arts = {}
    sp = fk_points(START_Q)[0]
    gp = fk_points(GOAL_Q)[0]
    for ax, name in zip(axes, results):
        for x, y, r in OBSTACLES:
            ax.add_patch(Circle((x, y), r, color="tomato", alpha=0.85, zorder=1))
            ax.add_patch(Circle((x, y), r + MARGIN, fill=False, ls="--", ec="#9b2c2c",
                                lw=1, zorder=1))
        ax.plot(sp[:, 0], sp[:, 1], "-", color="#38a169", lw=4, alpha=0.3, zorder=2)
        ax.plot(gp[:, 0], gp[:, 1], "-", color="#2b6cb0", lw=4, alpha=0.3, zorder=2)
        ax.plot(0, 0, "s", color="#4a5568", ms=14, zorder=2)
        ax.set_xlim(-0.75 * reach, 1.0 * reach)
        ax.set_ylim(-0.9 * reach, 1.0 * reach)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[name], fontsize=11, fontweight="bold")
        if name in trajs:
            ax.plot(handpath[name][:, 0], handpath[name][:, 1], ":", color="#a0aec0",
                    lw=1.5, zorder=2)
            arm, = ax.plot([], [], "-o", color="#2b6cb0", lw=6, ms=9, mfc="#e53e3e",
                           mec="white", zorder=4)
            hand, = ax.plot([], [], "*", color="#d69e2e", ms=20, mec="black", zorder=5)
            trace, = ax.plot([], [], "-", color="#d69e2e", lw=2.5, zorder=3)
            arts[name] = (arm, hand, trace)
        else:
            ax.text(0.5, 0.5, "too many grid cells\n(curse of dimensionality)",
                    transform=ax.transAxes, ha="center", fontsize=13, color="#9b2c2c",
                    fontweight="bold")
    fig.suptitle("Robotic arm path planning in joint space: green = start pose, "
                 "blue = goal pose, red = obstacle", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    def update(f):
        out = []
        for name, (arm, hand, trace) in arts.items():
            k = min(f, len(pts[name]) - 1)
            xy = pts[name][k]
            arm.set_data(xy[:, 0], xy[:, 1])
            hand.set_data([xy[-1, 0]], [xy[-1, 1]])
            hp = hands[name][:k + 1]
            trace.set_data(hp[:, 0], hp[:, 1])
            out += [arm, hand, trace]
        return out

    update(n_frames - 1)
    fig.savefig("arm_paths.png", dpi=150)
    print("\nSaved arm_paths.png")
    update(0)
    ani = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False, repeat=True)
    if SAVE_GIF:
        ani.save("arm_motion.gif", writer=PillowWriter(fps=25), dpi=60)
        print("Saved arm_motion.gif")

    # ---- joint-space map (only for 2 joints, where we can draw it)
    if N == 2:
        q1 = np.arange(LIM[0, 0], LIM[0, 1] + 1, 2.0)
        q2 = np.arange(LIM[1, 0], LIM[1, 1] + 1, 2.0)
        G1, G2 = np.meshgrid(q1, q2)
        ok = valid(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        ax2.imshow(~ok, origin="lower", extent=[q1[0], q1[-1], q2[0], q2[-1]],
                   aspect="auto", cmap="Reds", alpha=0.35)
        for name, color in zip(results, ["#2b6cb0", "#dd6b20", "#2f855a"]):
            p = results[name][0]
            if p is not None:
                arr = np.array(p)
                ax2.plot(arr[:, 0], arr[:, 1], "-o", ms=3, lw=2, color=color, label=name)
        ax2.plot(*START_Q, "go", ms=12, mec="black")
        ax2.plot(*GOAL_Q, "*", color="#d69e2e", ms=18, mec="black")
        ax2.set_xlabel("joint 1 angle q1 (deg)")
        ax2.set_ylabel("joint 2 angle q2 (deg)")
        ax2.set_title("Joint space: the whole arm is ONE dot\npink = forbidden poses")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig("cspace.png", dpi=150)
        print("Saved cspace.png")

    plt.show()


if __name__ == "__main__":
    main()