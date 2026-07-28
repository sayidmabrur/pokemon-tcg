from dataclasses import dataclass
from enum import Enum

from pandas import DataFrame, Series


class Category(Enum):
    BASIC_POKEMON = "Basic Pokémon"
    POKEMON_STAGE1 = "Stage 1 Pokémon"
    POKEMON_STAGE2 = "Stage 2 Pokémon"
    ITEM = "Item"
    SUPPORTER = "Supporter"
    TOOL = "Pokémon Tool"
    STADIUM = "Stadium"
    ENERGY = "Special Energy"
    BASIC_ENERGY = "Basic Energy"


@dataclass(frozen=True)
class Card:
    id: int
    count: int
    category: Category


class Deck:

    def __init__(self):
        self.decks = {}
        pass

    def add_card(self, id: int, count: int, category: Category):
        self.decks[id] = Card(id=id, count=count, category=category)


def merge_card(deck_df: Series, card_source_df: DataFrame):
    counts = deck_df.value_counts().rename_axis("Card ID").reset_index(name="count")
    cards_in_decks = counts.merge(
        card_source_df[
            [
                "Card ID",
                "Card Name",
                "Category",
                "Stage (Pokémon)/Type (Energy and Trainer)",
                "Rule",
                "Type",
            ]
        ],
        on="Card ID",
        how="left",
    )

    return cards_in_decks


def build_deck(deck_csv_path):
    import pandas as pd

    card_source_df = pd.read_csv("EN_Card_Data.csv")
    deck_df = pd.read_csv(deck_csv_path, header=None).squeeze("columns")

    # merge card
    cards_in_decks = merge_card(deck_df, card_source_df)

    # define the decks
    deck = Deck()
    for _, card in cards_in_decks.iterrows():
        deck.add_card(
            id=card["Card ID"],
            count=card["count"],
            category=Category(card["Stage (Pokémon)/Type (Energy and Trainer)"]),
        )

    return deck

