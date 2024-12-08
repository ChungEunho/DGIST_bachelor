def RUNRUNHURRY(g, s, nodes, bike_pts, min_t):
    d = [float('inf')] * (nodes + 1)
    d[s] = 0
    queue = [(0, s)]
    vis = [False] * (nodes + 1)

    while queue:
        c_dist, c_node = queue[0]
        for i in range(1, len(queue)):
            if queue[i][0] < c_dist:
                c_dist, c_node = queue[i]
        queue.remove((c_dist, c_node))

        if vis[c_node]:
            continue
        vis[c_node] = True

        for nbr, wt in g[c_node]:
            n_dist = c_dist + wt

            if c_node in bike_pts and c_dist >= min_t:
                n_dist = c_dist + wt / 2

            if n_dist < d[nbr]:
                d[nbr] = n_dist
                queue.append((n_dist, nbr))

    return d

nodes, edges, bikes, t = map(int, input().split())
g = [[] for _ in range(nodes + 1)]

for _ in range(edges):
    x, y, w = map(int, input().split())
    g[x].append((y, w))
    g[y].append((x, w))

bike_pts = set()
for _ in range(bikes):
    bike_pts.add(int(input()))

d = RUNRUNHURRY(g, 1, nodes, bike_pts, t)

if d[nodes] == float('inf'):
    print(-1)
else:
    result = d[nodes]
    print(round(result))
