def find_pair_within_range(A, T): 
    if T < 1:
        return "No"
    
    bucket_size = T + 1
    buckets = {}
    
    for num in A:
        idx = num // bucket_size
        if idx in buckets:
            return "Yes"
        if idx - 1 in buckets and num - buckets[idx - 1] <= T:
            return "Yes"
        if idx + 1 in buckets and buckets[idx + 1] - num <= T:
            return "Yes"
        
        buckets[idx] = num
    
    return "No"
    
A0 = [100,1000,35,200,755,400,1]
T1 = 33

print(find_pair_within_range(A0, T1))

T2 = 34
print(find_pair_within_range(A0, T2))