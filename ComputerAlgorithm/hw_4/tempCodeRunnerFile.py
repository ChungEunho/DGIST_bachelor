def bfs(capacity, flow, source, sink, parent, N):
    visited = [False] * N
    queue = [source]
    visited[source] = True

    while queue:
        u = queue.pop(0)
        for v in range(N):
            if not visited[v] and capacity[u][v] - flow[u][v] > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True
    return False

def find_reachable(capacity, flow, source, N):
    reachable = [False] * N
    queue = [source]
    reachable[source] = True
    while queue:
        u = queue.pop(0)
        for v in range(N):
            if not reachable[v] and capacity[u][v] - flow[u][v] > 0:
                reachable[v] = True
                queue.append(v)
    return reachable

def ford_fulkerson(N, capacity, source, sink, invalid_edges):
    flow = [[0] * N for _ in range(N)]
    parent = [-1] * N
    max_flow = 0

    while bfs(capacity, flow, source, sink, parent, N):
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]

            path_flow = min(path_flow, capacity[u][v] - flow[u][v])
            v = u

        if path_flow == 0:
            break

        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            v = u

        max_flow += path_flow

    # Check if the cut is valid
    reachable = find_reachable(capacity, flow, source, N)

    for u in range(N):
        for v in range(N):
            if reachable[u] and not reachable[v] and (u, v) in invalid_edges:
                # Invalid cut, return -1
                return -1

    return max_flow

def junjaeng_momcho(K, graph, N, opponent_loc, my_loc):
    capacity = [[0] * N for _ in range(N)]
    invalid_edges = set()

    for u in range(N):
        for v, c in graph[u]:
            capacity[u][v] = c 
            if c > K:
                invalid_edges.add((u, v))

    max_flow = ford_fulkerson(N, capacity, opponent_loc, my_loc, invalid_edges)
    return max_flow

# Input handling
N, M, K = map(int, input().split())
graph = [[] for _ in range(N)]

for _ in range(M):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))
    graph[b].append((a, c))

opponent_loc = 0
my_loc = N - 1

answer = junjaeng_momcho(K, graph, N, opponent_loc, my_loc)
print(answer)
