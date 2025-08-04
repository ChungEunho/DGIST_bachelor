def bfs(grid, n, start, end):
    queue = [start]
    visited = set()
    visited.add(start)

    while queue:
        x, y = queue.pop(0)

        if (x, y) == end:
            return True

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited and grid[nx][ny] == 0:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return False

def find_mincut(grid, n, start, end):
    def find_nodetuple(node):
        return node * 2, node * 2 + 1

    def edge_construction(graph, u, v, capacity):
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append([v, capacity])
        graph[v].append([u, 0])

    def bfs_mincut(graph, parent, source, sink, node_count):
        visited = [False] * node_count
        queue = [source]
        visited[source] = True

        while queue:
            u = queue.pop(0)
            for v, capacity in graph.get(u, []):
                if not visited[v] and capacity > 0:
                    parent[v] = u
                    visited[v] = True
                    if v == sink:
                        return True
                    queue.append(v)
        return False

    def edmonds_karp(graph, source, sink, node_count):
        max_flow = 0
        parent = [-1] * node_count

        while bfs_mincut(graph, parent, source, sink, node_count):
            path_flow = float('Inf')
            v = sink

            while v != source:
                u = parent[v]
                for edge in graph[u]:
                    if edge[0] == v:
                        path_flow = min(path_flow, edge[1])
                        break
                v = u

            v = sink
            while v != source:
                u = parent[v]
                for edge in graph[u]:
                    if edge[0] == v:
                        edge[1] -= path_flow
                        break
                for edge in graph[v]:
                    if edge[0] == u:
                        edge[1] += path_flow
                        break
                v = u

            max_flow += path_flow

        return max_flow

    graph = {}
    node_count = 2 * n * n
    source = find_nodetuple(start[0] * n + start[1])[1]
    sink = find_nodetuple(end[0] * n + end[1])[0]

    for x in range(n):
        for y in range(n):
            if grid[x][y] == 0:
                node = x * n + y
                node_in, node_out = find_nodetuple(node)

                edge_construction(graph, node_in, node_out, 1)

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0:
                        neighbor = nx * n + ny
                        neighbor_in = find_nodetuple(neighbor)[0]
                        edge_construction(graph, node_out, neighbor_in, 1)

    return edmonds_karp(graph, source, sink, node_count)

def youcantpassme(n, x1, y1, x2, y2, obstacles):
    if not (0 <= x1 <= n and 0 <= y1 <= n and 0 <= x2 <= n and 0 <= y2 <= n):
        return

    grid = [[0] * (n + 1) for _ in range(n + 1)]
    for obstacle_x, obstacle_y in obstacles:
        if 0 <= obstacle_x <= n and 0 <= obstacle_y <= n:
            grid[obstacle_x][obstacle_y] = 1

    if (x1, y1) == (x2, y2):
        print(0)
        return

    if grid[x1][y1] == 1 or grid[x2][y2] == 1:
        print(0)
        return

    if not bfs(grid, n + 1, (x1, y1), (x2, y2)):
        print(0)
        return

    result = find_mincut(grid, n + 1, (x1, y1), (x2, y2))
    print(result)

n, m = map(int, input().split())
x1, y1, x2, y2 = map(int, input().split())
obstacles = [tuple(map(int, input().split())) for _ in range(m)]
youcantpassme(n, x1, y1, x2, y2, obstacles)
