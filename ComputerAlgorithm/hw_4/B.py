def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])  
    return parent[x]

def union(parent, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)

    if root_x != root_y:
        parent[root_y] = root_x 

def query_single_forest(parent, n):
    root_set = set(find(parent, i) for i in range(n))
    return len(root_set) == 1

def mogomogo_mogomogo_restaurant(n, memory_limit, edges):
    
    edges.sort(key=lambda x: x[2])  
    parent = [i for i in range(n)]  
    mem_usage_cnt = 0

    for edge in edges:
        a, b, cost = edge
        if find(parent, a) != find(parent, b):  
            union(parent, a, b)  
            mem_usage_cnt += cost

            if query_single_forest(parent, n):
                break

    return mem_usage_cnt if mem_usage_cnt <= memory_limit else -1

n, memory_limit = map(int, input().split())
edges = []
for _ in range(n * (n - 1) // 2):
    a, b, c = map(int, input().split())
    edges.append((a, b, c))

result = mogomogo_mogomogo_restaurant(n, memory_limit, edges)
print(result)
