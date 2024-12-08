def BoongoPpang(N, M):
    dp = [[[0] * (M + 1) for _ in range(10)] for _ in range(N + 1)]
    
    #0번째 날부터가 아니라 첫번째 날부터 초기화를 해 준다.
    for i in range(min(10,M+1)):
        dp[1][i][i] = 1

    for day in range(2, N + 1):
        for yesterday_mukbang in range(10):  # 전날 먹은 양 (최대 10까지)
            for today_mukbang in range(max(0, yesterday_mukbang - 1), min(9, yesterday_mukbang + 1) + 1):  # 오늘 먹는 양
                if today_mukbang == 0 and yesterday_mukbang == 0:
                    continue  # 이틀 연속 굶는 경우는 제외해야 한다!!
                for i in range(len(dp[day-1][yesterday_mukbang])):
                    if (dp[day-1][yesterday_mukbang][i] != 0):
                        previous_total = i
                        if (previous_total + today_mukbang <= M):
                            dp[day][today_mukbang][previous_total + today_mukbang] += dp[day - 1][yesterday_mukbang][previous_total]

    result = 0
    for today_mukbang in range(10):
        for total in range(M + 1):
            result += dp[N][today_mukbang][total]
    return result

N, M = map(int, input().split())
print(BoongoPpang(N, M))