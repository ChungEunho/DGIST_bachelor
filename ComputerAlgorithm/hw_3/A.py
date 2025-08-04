def DALGU_POWER(initial_sleeping):
    sleep_time_info = [initial_sleeping]  # sleeping만 리스트에 저장하고 time은 따로 관리
    time = 0

    while sleep_time_info:
        next_sleep_time_info = []

        for sleeping in sleep_time_info:
            if sleeping == "11111111":
                return time

            if (time + 1) % 4 != 0:
                move_rightmost_sleeping = sleeping[-1] + sleeping[:-1]
                next_sleep_time_info.append(move_rightmost_sleeping)

                move_leftmost_sleeping = sleeping[1:] + sleeping[0]
                next_sleep_time_info.append(move_leftmost_sleeping)

                if '0' in sleeping:
                    idx = sleeping.rfind('0')
                    wakeup_sleeping = sleeping[:idx] + '1' + sleeping[idx+1:]
                    next_sleep_time_info.append(wakeup_sleeping)
                superscience_sleeping = ''.join('1' if ch == '0' else '0' for ch in sleeping)
                next_sleep_time_info.append(superscience_sleeping)
            else:
                # 휴식 상태로 변경
                nyam_nyam_sleeping = '0' + sleeping[1:-1] + '0'
                next_sleep_time_info.append(nyam_nyam_sleeping)

        sleep_time_info = next_sleep_time_info
        time += 1

    return -1

initial_sleeping = input().strip()
print(DALGU_POWER(initial_sleeping))
