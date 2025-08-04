def whatkindofgameisthis(n, k, cards):
    reach_progress = 0
    additional_cards_num = 0
    cards.sort() 

    for card in cards:
        while reach_progress + 1 < card:
            additional_cards_num += 1
            card_num_to_add = reach_progress + 1
            reach_progress += card_num_to_add 
            if reach_progress >= k:
                return additional_cards_num

        reach_progress += card
        if reach_progress >= k:
            return additional_cards_num

    while reach_progress < k:
        additional_cards_num += 1
        reach_progress += (reach_progress + 1)

    return additional_cards_num

N, K = map(int, input().split())
initial_cards = list(map(int, input().split()))

print(whatkindofgameisthis(N, K, initial_cards))
