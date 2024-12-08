def PUTERPUTERCOMPUTER(n, k, t, tasks):
    
    if n==1:
        if (tasks[0] > t):
            return -1
        else:
            return t-tasks[0]
        
    else:
        table = [0] * k
        
        for i in range(k):
            table[i] = tasks[i]
        for i in range(k,len(tasks)):
            num_to_add = tasks[i]
            min_index = table.index(min(table))
            table[min_index] += num_to_add
        
        if (max(table) > t):
            return -1
        else:
            return t - max(table)
        
n,k,t = map(int, input().split())

tasks = []

for _ in range(n):
    tasks.append(int(input()))
    
tasks.sort(reverse = True)

print(PUTERPUTERCOMPUTER(n,k,t,tasks))