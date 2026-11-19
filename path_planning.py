"""
2D path planning comparison: A*, RRT, RRT*
-------------------------------------------
Same map, same start/goal, three planners.
Run:  python path_planning.py
Needs: numpy, matplotlib

Idea for the presentation: in 2D the "configuration" is (x, y).
For a robotic arm the configuration is the joint angles (q1..q7),
and the collision check is done on the arm links instead of a point.
The planners themselves stay the same.
"""
import heapq
import math
import time

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# 1. Environment
# ----------------------------------------------------------------------
SIZE = 100.0
START = np.array([5.0, 5.0])
GOAL = np.array([95.0, 95.0])
# circular obstacles: (x, y, radius)
OBSTACLES = np.array([
    [30, 30, 12],
    [50, 60, 14],
    [75, 40, 10],
    [65, 82, 8],
    [20, 70, 9],
], dtype=float)
MARGIN = 1.5  # safety margin ("safest" part of the objective)

rng = np.random.default_rng(1)


def point_free(p):
    """True if point p is inside the map and outside every obstacle (+margin)."""
    if p[0] < 0 or p[0] > SIZE or p[1] < 0 or p[1] > SIZE:
        return False
    d = np.hypot(OBSTACLES[:, 0] - p[0], OBSTACLES[:, 1] - p[1])
    return bool(np.all(d > OBSTACLES[:, 2] + MARGIN))


def segment_free(p, q):
    """True if the straight segment p->q does not touch any obstacle.
    (For a real arm: check every link at every sampled configuration.)"""
    dist = float(np.linalg.norm(q - p))
    n = max(2, int(dist / 0.5) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = p + t * (q - p)
    if np.any(pts < 0) or np.any(pts > SIZE):
        return False
    dx = pts[:, 0][:, None] - OBSTACLES[:, 0][None, :]
    dy = pts[:, 1][:, None] - OBSTACLES[:, 1][None, :]
    d = np.hypot(dx, dy)
    return bool(np.all(d > OBSTACLES[:, 2][None, :] + MARGIN))


def path_length(path):
    if path is None:
        return float("nan")
    return float(sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
                     for i in range(len(path) - 1)))


# ----------------------------------------------------------------------
# 2. A*  (grid search, f = g + h)
# ----------------------------------------------------------------------
def astar(resolution=1.0):
    n = int(SIZE / resolution) + 1
    free = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            free[i, j] = point_free(np.array([i * resolution, j * resolution]))

    s = (int(round(START[0] / resolution)), int(round(START[1] / resolution)))
    g_cell = (int(round(GOAL[0] / resolution)), int(round(GOAL[1] / resolution)))

    def h(c):  # heuristic: straight-line distance to goal (never overestimates)
        return math.hypot(c[0] - g_cell[0], c[1] - g_cell[1]) * resolution

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_heap = [(h(s), 0.0, s)]
    g = {s: 0.0}
    parent = {s: None}
    closed = set()
    expanded = []

    while open_heap:
        _, cost, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded.append(cur)
        if cur == g_cell:
            path = []
            while cur is not None:
                path.append(np.array(cur) * resolution)
                cur = parent[cur]
            return path[::-1], np.array(expanded) * resolution
        for dx, dy in moves:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < n and 0 <= nb[1] < n) or not free[nb]:
                continue
            new_g = cost + math.hypot(dx, dy) * resolution
            if new_g < g.get(nb, float("inf")):
                g[nb] = new_g
                parent[nb] = cur
                heapq.heappush(open_heap, (new_g + h(nb), new_g, nb))
    return None, np.array(expanded) * resolution


# ----------------------------------------------------------------------
# 3. RRT  (random tree, first path found)
# ----------------------------------------------------------------------
def steer(p_from, p_to, step):
    d = p_to - p_from
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return p_from.copy()
    return p_from + d / L * min(step, L)


def rrt(max_iter=5000, step=4.0, goal_sample=0.05, goal_tol=4.0):
    nodes = [START.copy()]
    parent = [-1]
    for _ in range(max_iter):
        q = GOAL if rng.random() < goal_sample else rng.uniform(0, SIZE, 2)
        arr = np.array(nodes)
        i = int(np.argmin(((arr - q) ** 2).sum(axis=1)))
        new = steer(nodes[i], q, step)
        if not segment_free(nodes[i], new):
            continue
        nodes.append(new)
        parent.append(i)
        if np.linalg.norm(new - GOAL) < goal_tol and segment_free(new, GOAL):
            nodes.append(GOAL.copy())
            parent.append(len(nodes) - 2)
            path, k = [], len(nodes) - 1
            while k != -1:
                path.append(nodes[k])
                k = parent[k]
            return path[::-1], np.array(nodes), parent
    return None, np.array(nodes), parent


# ----------------------------------------------------------------------
# 4. RRT*  (RRT + choose best parent + rewire  -> shorter paths)
# ----------------------------------------------------------------------
def rrt_star(max_iter=2500, step=4.0, goal_sample=0.05, radius=12.0, goal_tol=6.0):
    N = max_iter + 1
    P = np.zeros((N, 2))
    P[0] = START
    parent = [-1] * N
    cost = np.zeros(N)
    children = [set() for _ in range(N)]
    n = 1

    for _ in range(max_iter):
        q = GOAL if rng.random() < goal_sample else rng.uniform(0, SIZE, 2)
        i = int(np.argmin(((P[:n] - q) ** 2).sum(axis=1)))
        new = steer(P[i], q, step)
        if not segment_free(P[i], new):
            continue

        dists = np.linalg.norm(P[:n] - new, axis=1)
        near = np.where(dists <= radius)[0]

        # (a) choose the best parent among nearby nodes
        best = i
        best_cost = cost[i] + float(np.linalg.norm(new - P[i]))
        for j in near:
            c = cost[j] + dists[j]
            if c < best_cost and segment_free(P[j], new):
                best, best_cost = int(j), c

        k = n
        P[k] = new
        parent[k] = best
        cost[k] = best_cost
        children[best].add(k)
        n += 1

        # (b) rewire: if going through the new node is cheaper, switch parent
        for j in near:
            j = int(j)
            if j == best:
                continue
            c = best_cost + dists[j]
            if c < cost[j] and segment_free(new, P[j]):
                children[parent[j]].discard(j)
                parent[j] = k
                children[k].add(j)
                delta = c - cost[j]
                cost[j] = c
                stack = list(children[j])  # push the saving to descendants
                while stack:
                    m = stack.pop()
                    cost[m] += delta
                    stack.extend(children[m])

    # connect the best node near the goal
    dg = np.linalg.norm(P[:n] - GOAL, axis=1)
    best_j, best_total = None, float("inf")
    for j in np.where(dg <= goal_tol)[0]:
        total = cost[j] + dg[j]
        if total < best_total and segment_free(P[j], GOAL):
            best_j, best_total = int(j), total
    if best_j is None:
        return None, P[:n], parent[:n]
    path = [GOAL.copy()]
    k = best_j
    while k != -1:
        path.append(P[k].copy())
        k = parent[k]
    return path[::-1], P[:n], parent[:n]


# ----------------------------------------------------------------------
# 5. Run all three and plot
# ----------------------------------------------------------------------
def draw_base(ax, title):
    for x, y, r in OBSTACLES:
        ax.add_patch(plt.Circle((x, y), r, color="tomato", alpha=0.7))
    ax.plot(*START, "go", markersize=10, label="Start")
    ax.plot(*GOAL, "b*", markersize=14, label="Goal")
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, SIZE)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def draw_tree(ax, nodes, parent):
    for k in range(1, len(nodes)):
        p = parent[k]
        if p >= 0:
            ax.plot([nodes[p][0], nodes[k][0]], [nodes[p][1], nodes[k][1]],
                    color="lightgray", linewidth=0.5, zorder=1)


def draw_path(ax, path):
    if path is not None:
        pts = np.array(path)
        ax.plot(pts[:, 0], pts[:, 1], "k-", linewidth=2.5, zorder=3, label="Path")


def main():
    results = {}

    t = time.time()
    p_a, explored = astar(resolution=1.0)
    results["A*"] = (p_a, time.time() - t)

    t = time.time()
    p_r, nodes_r, par_r = rrt()
    results["RRT"] = (p_r, time.time() - t)

    t = time.time()
    p_s, nodes_s, par_s = rrt_star()
    results["RRT*"] = (p_s, time.time() - t)

    print(f"{'Method':<6} {'Path length':>12} {'Time (s)':>10}")
    for name, (path, secs) in results.items():
        print(f"{name:<6} {path_length(path):>12.2f} {secs:>10.2f}")
    print(f"Straight-line distance start->goal (lower bound): "
          f"{np.linalg.norm(GOAL - START):.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    draw_base(axes[0], f"A*  (grid)  length={path_length(p_a):.1f}")
    if explored is not None and len(explored):
        axes[0].plot(explored[:, 0], explored[:, 1], ".", color="lightgray",
                     markersize=2, zorder=1)
    draw_path(axes[0], p_a)

    draw_base(axes[1], f"RRT  length={path_length(p_r):.1f}")
    draw_tree(axes[1], nodes_r, par_r)
    draw_path(axes[1], p_r)

    draw_base(axes[2], f"RRT*  length={path_length(p_s):.1f}")
    draw_tree(axes[2], nodes_s, par_s)
    draw_path(axes[2], p_s)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Path planning comparison (same map, same start and goal)")
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    print("Saved comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
