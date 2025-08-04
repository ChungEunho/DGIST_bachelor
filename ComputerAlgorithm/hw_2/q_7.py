#optimal_hash(S)를 사용하여 가장 긴 길이의 공통된 subsequence를 찾는 함수를 작성하고자 한다. python3 함수를 완성하시오
# 가장 긴 길이의 공통 subsequence가 둘 이상 존재할 경우, 둘 중 하나만 반환해도 무방하다
# optimal_hash(S)는 어떠한 DNA 염기서열 집합에 대해서든 집합 내에 속한 서로 다른 sequence x,y에 대해서 
# Optimal_hash(X) != optimal_hash(y)를 보장하는 해시 함수이다. 

# 최장 공통 서브시퀀스를 찾는 함수
def DNAcompare(s1, s2):
    max_length = min(len(s1), len(s2))  # 두 문자열의 길이 중 최소값을 선택
    result = ""
    
    for i in range(1, max_length + 1):  # 1부터 max_length까지의 길이로 서브시퀀스 생성
        for j in range(len(s1) - i + 1):  # s1의 서브시퀀스 시작 인덱스 선택
            for k in range(len(s2) - i + 1):  # s2의 서브시퀀스 시작 인덱스 선택
                # s1과 s2의 길이 i인 서브시퀀스의 해시 값 비교
                if optimal_hash(s1[j:j+i]) == optimal_hash(s2[k:k+i]):  
                    # 현재까지의 결과보다 긴 서브시퀀스가 발견되면 업데이트
                    result = s1[j:j+i]  
    return result
