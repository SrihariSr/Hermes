import random

from poker.action import Action
from poker.card import Card, Rank, Suit
from poker.game_state import GameState
from poker.hand import evaluate


# How many imagined deals to try per decision
SIMULATIONS = 300

# All 52 cards, built once when the file is loaded rather than every decision
FULL_DECK = [Card(rank, suit) for rank in Rank for suit in Suit]

# Our own random number generator, so we never disturb anyone else's.
rng = random.Random()


def win_chance(my_cards: set, shared_cards: set, opponents: int) -> float:
    """
    Estimate how often our cards beat `opponents` other players, by playing the
    rest of the hand out at random `SIMULATIONS` times.
    """
    # The cards we can't see: everything not in our hand or on the table
    unseen = [card for card in FULL_DECK if card not in my_cards and card not in shared_cards]
    shared = list(shared_cards)
    mine = list(my_cards)

    # Each imagined deal needs 2 cards per opponent, plus whatever shared cards are still to come
    cards_needed = 2 * opponents + (5 - len(shared))

    wins = 0.0
    for _ in range(SIMULATIONS):
        dealt = rng.sample(unseen, cards_needed)

        # The first 2 * opponents cards are the opponents' hands, the rest finish the table
        final_shared = shared + dealt[2 * opponents:]

        # evaluate() scores the best 5 cards out of the 7 given. Higher score = better hand.
        my_score = evaluate(mine + final_shared)
        best_other = max(evaluate(dealt[2 * k: 2 * k + 2] + final_shared) for k in range(opponents))

        if my_score > best_other:
            wins += 1
        elif my_score == best_other:
            wins += 0.5   # a tie: count it as half a win (good enough for now)

    return wins / SIMULATIONS


def decide(state: GameState) -> Action:
    me = state.my_player

    # How many chips we must add to stay in: the biggest bet this round minus what we've put in.
    biggest_bet = max(player.chips_bet for player in state.players)
    to_call = biggest_bet - me.chips_bet

    # Opponents still in this hand (everyone except us who hasn't folded)
    opponents = sum(1 for player in state.players if player is not me and not player.has_folded)

    # If everyone else has folded there's nobody to beat. The real game never asks us
    # in this case, but the scorer's test situations do, so handle it rather than crash.
    if opponents == 0:
        return Action.check() if to_call == 0 else Action.call()

    chance = win_chance(me.hole_cards, state.community_cards, opponents)

    # The price of calling, as a share of the pot we'd end up playing for.
    price = to_call / (state.chips_in_pot + to_call) if to_call > 0 else 0.0

    # If everyone had equally good cards, we'd each win 1 / (number of players).
    # Clearly better than that means we're strong, so we bet and make others pay to stay in.
    fair_share = 1 / (opponents + 1)
    if chance > 1.5 * fair_share:
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