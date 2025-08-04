# 그래프 정의
graph = {}        # 말 그대로 전체 그래프
edges = []

# 입력 받는 부분!!
n = int(input())
for _ in range(n):
    x1, y1, x2, y2 = map(int, input().split())
    if (x1, y1) not in graph:
        graph[(x1, y1)] = []
    if (x2, y2) not in graph:
        graph[(x2, y2)] = []
    graph[(x1, y1)].append((x2, y2))
    graph[(x2, y2)].append((x1, y1))
    edges.append(((x1, y1), (x2, y2)))

# DFS를 이용해서 사이클 찾기!!
def find_cycle(node, parent, visited, path):
    visited.add(node)
    path.append(node)
    
    for neighbor in graph[node]:
        if neighbor == parent:
            continue
        if neighbor in path:
            # 사이클 발견
            cycle_start_index = path.index(neighbor)
            return path[cycle_start_index:]
        if neighbor not in visited:
            result = find_cycle(neighbor, node, visited, path)
            if result:
                return result
                
    path.pop()
    return None

# 사이클 내에 중복되는 선분이 없는지 확인하는 함수!!
def check_valid_cycle(cycle):
    for i in range(len(cycle) - 1):
        s, e = cycle[i], cycle[i + 1]
        if (edges.count((s, e)) + edges.count((e, s))) >= 2:
            return False
    return True


# 사이클을 그래프에서 제거하는 함수!!
def remove_cycle(graph, cycle):
    for i in range(len(cycle) - 1):
        node1, node2 = cycle[i], cycle[i + 1]
        if node2 in graph[node1]:
            graph[node1].remove(node2)
        if node1 in graph[node2]:
            graph[node2].remove(node1)


# 사이클에 사이클을 이루지 않는 선분이 있는지 확인하는 함수!!
def has_external_edge(graph, cycle):
    cycle_set = set(cycle)
    for node in cycle:
        for neighbor in graph[node]:
            if neighbor not in cycle_set:
                return True
    return False

# 유효한 사이클 개수를 찾는 함수!!
def count_cycles(graph):
    cycle_count = 0
    visited = set()
    nodes = list(graph.keys())

    for node in nodes:
        if node not in visited:
            cycle = find_cycle(node, None, visited, [])
            if cycle and len(cycle) > 2:
                # 사이클 내에 중복된 선분이 없고, 외부 연결이 없는 경우에만 유효한 사이클로 판단!!
                if check_valid_cycle(cycle) and not has_external_edge(graph, cycle):
                    # 사이클이 유효하다면 해당 사이클을 그래프에서 제거!!
                    remove_cycle(graph, cycle)
                    visited.update(cycle)
                    cycle_count += 1

    return cycle_count

print(count_cycles(graph))

