import random

from poker.action import Action
from poker.card import Card, Rank, Suit
from poker.game_state import GameState
from poker.hand import evaluate


# How many imagined deals to try per decision
SIMULATIONS = 500

FULL_DECK = [Card(rank, suit) for rank in Rank for suit in Suit]

rng = random.Random()


RANKING = (
    "AA KK QQ JJ TT 99 AKs 88 AQs AJs AKo ATs KQs AQo KJs 77 AJo KTs A9s QJs KQo ATo KJo 66 "
    "A8s QTs K9s A7s JTs KTo A9o QJo Q9s A5s A6s A4s K8s A8o QTo 55 A3s J9s K9o JTo K7s A7o "
    "Q8s K6s A5o T9s A2s A6o Q9o J8s K8o 44 A4o K5s K4s T8s K7o A3o J9o Q7s T9o 98s Q6s Q8o "
    "J7s K3s A2o Q5s K2s 97s T7s K6o Q4s K5o J8o J6s 87s T8o 33 Q3s Q7o T6s K4o J5s 98o Q2s "
    "Q6o J7o 96s K3o J4s 76s 86s J3s Q5o K2o T7o T5s 22 97o J2s 65s Q4o T4s 87o 75s 85s 95s "
    "J6o T3s Q3o J5o 54s T6o T2s 96o 86o Q2o 64s 74s 76o 84s J4o 94s 93s J3o 53s 92s T5o 75o "
    "65o 95o 85o 63s 43s J2o T4o 73s 83s 82s T3o 54o 52s 62s 64o 84o 42s T2o 72s 74o 32s 94o "
    "93o 53o 92o 43o 63o 73o 83o 82o 52o 42o 62o 72o 32o"
)

# One character per rank, lowest first
RANK_CHAR = dict(zip(Rank, "23456789TJQKA"))
RANK_ORDER = {rank: i for i, rank in enumerate(Rank)}


def hand_kind(card1: Card, card2: Card) -> str:
    # Turn two cards into their kind, e.g. K♥ A♥ -> "AKs", 7♣ 7♦ -> "77"
    high, low = sorted((card1, card2), key=lambda c: RANK_ORDER[c.rank], reverse=True)
    if high.rank == low.rank:
        return RANK_CHAR[high.rank] * 2
    return RANK_CHAR[high.rank] + RANK_CHAR[low.rank] + ("s" if high.suit == low.suit else "o")


def build_top_share() -> dict:
    share, hands_so_far = {}, 0
    for kind in RANKING.split():
        hands_so_far += 6 if len(kind) == 2 else 4 if kind.endswith("s") else 12
        share[kind] = hands_so_far / 1326
    return share


TOP_SHARE = build_top_share()

memory = {"hand": None, "shares": {}, "raised": False}

def start_new_hand_if_needed(my_cards: set):
    # Our private cards change every hand, so new cards mean a new hand: forget the old reads
    if frozenset(my_cards) != memory["hand"]:
        memory["hand"] = frozenset(my_cards)
        memory["shares"] = {} # opponent's position in state.players -> share
        memory["raised"] = False # has anyone (including me) raised before the shared cards?


def read_opponents(state: GameState) -> list:
    """Return one share for each opponent still in the hand."""
    me = state.my_player
    big_blind = max(1, state.big_blind)
    first_round = len(state.community_cards) == 0

    # Every different bet above the big blind is a raise. Two different levels means
    # someone raised and then someone else raised again on top.
    raise_levels = sorted({p.chips_bet for p in state.players if p is not me and p.chips_bet > big_blind})
    if first_round and raise_levels:
        memory["raised"] = True

    shares = []
    for position, player in enumerate(state.players):
        if player is me or player.has_folded:
            continue

        if first_round:
            bet = player.chips_bet
            if bet > big_blind and player.chips_in_stack == 0:
                # Went all in. Players short of chips go all in with weaker hands,
                # because waiting means the forced bets slowly eat their chips.
                share = 0.45 if bet <= 10 * big_blind else 0.30 if bet <= 25 * big_blind else 0.20
            elif bet > big_blind:
                share = 0.08 if len(raise_levels) >= 2 and bet == raise_levels[-1] else 0.20
            elif bet == big_blind and player.blind is None:
                share = 0.55 # paid just the minimum to stay in: a middling hand
            else:
                share = 1.0 # hasn't chosen anything yet (or is the big blind, who was forced to bet)

            # Keep the strongest read we've seen this hand
            memory["shares"][position] = min(memory["shares"].get(position, 1.0), share)

        share = memory["shares"].get(position, 1.0)
        if not first_round and share == 1.0:
            # We never saw them choose before the shared cards, but they're still here,
            # so they paid to continue: likely a raise if anyone raised, otherwise not much.
            share = 0.35 if memory["raised"] else 0.65
        shares.append(share)

    return shares


def deal_from_range(deck: list, share: float) -> list:
    """Take two cards from `deck` that make a hand in the top `share`, and remove them."""
    for attempt in range(50):
        i, j = rng.sample(range(len(deck)), 2)
        if TOP_SHARE[hand_kind(deck[i], deck[j])] <= share:
            break

    cards = [deck[i], deck[j]]
    for position in sorted((i, j), reverse=True):
        deck.pop(position)
    return cards


def win_chance(my_cards: set, shared_cards: set, opponent_shares: list) -> float:
    """
    Estimate how often our cards beat the opponents, by playing the rest of the
    hand out at random `SIMULATIONS` times. Each opponent is dealt only hands
    that fit their share.
    """
    # The cards we can't see: everything not in our hand or on the table
    unseen = [card for card in FULL_DECK if card not in my_cards and card not in shared_cards]
    shared = list(shared_cards)
    mine = list(my_cards)

    wins = 0.0
    for _ in range(SIMULATIONS):
        deck = unseen[:] # a fresh copy each time, so dealing doesn't use up the real list

        opponent_hands = [deal_from_range(deck, share) for share in opponent_shares]

        # Finish the table with random cards from what's left
        final_shared = shared + rng.sample(deck, 5 - len(shared))

        # evaluate() scores the best 5 cards out of the 7 given. Higher score = better hand.
        my_score = evaluate(mine + final_shared)
        best_other = max(evaluate(hand + final_shared) for hand in opponent_hands)

        if my_score > best_other:
            wins += 1
        elif my_score == best_other:
            wins += 0.5 # a tie: count it as half a win

    return wins / SIMULATIONS


def decide(state: GameState) -> Action:
    me = state.my_player
    start_new_hand_if_needed(me.hole_cards)

    # How many chips we must add to stay in: the biggest bet this round minus what we've put in.
    biggest_bet = max(player.chips_bet for player in state.players)
    to_call = biggest_bet - me.chips_bet

    # Opponents still in this hand (everyone except us who hasn't folded)
    opponents = sum(1 for player in state.players if player is not me and not player.has_folded)

    if opponents == 0:
        return Action.check() if to_call == 0 else Action.call()

    chance = win_chance(me.hole_cards, state.community_cards, read_opponents(state))

    # The price of calling, as a share of the pot we'd end up playing for.
    price = to_call / (state.chips_in_pot + to_call) if to_call > 0 else 0.0

    # If everyone had equally good cards, we'd each win 1 / (number of players).
    # Clearly better than that means we're strong, so we bet and make others pay to stay in.
    fair_share = 1 / (opponents + 1)
    if chance > 1.5 * fair_share:
        if not state.community_cards:
            memory["raised"] = True   # remember that we raised, for reading callers later
        # raise_by() first matches the biggest bet, then adds this much on top
        return Action.raise_by(max(1, state.chips_in_pot // 2))

    if to_call == 0:
        return Action.check() # free to stay in, so never fold

    if chance > price:
        return Action.call() # we win often enough to pay this price

    return Action.fold()


def myAI(state: GameState) -> Action:
    try:
        return decide(state)
    except Exception:
        return defaultAI(state)


def defaultAI(state: GameState) -> Action:
    return Action.check() if state.my_player.chips_bet == max(player.chips_bet for player in state.players) else Action.fold()