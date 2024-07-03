#!/usr/bin/python

import argparse
from dataclasses import replace
from itertools import chain, product
import math
from pathlib import Path
from typing import Iterable, Self, Sequence

import networkx as nx
#from matplotlib import pyplot as plt
import pandas as pd

from game import Game, Cartridge, Hardware, Collection, GAMES, GAMEIDS, GAMEINPUTS
from result import MultiResult, Result
from structs import Gender, Pokemon, Req, PokemonReq, Entry, PokemonEntry, ItemEntry, ChoiceEntry, DexEntry, SpeciesDexEntry, FormDexEntry, Rule

class Generation():
    breed: pd.DataFrame | None
    buy: pd.DataFrame | None
    evolve: pd.DataFrame | None
    forms_change: pd.DataFrame | None
    forms_ingame: pd.DataFrame | None
    forms_transfer: pd.DataFrame | None
    fossil: pd.DataFrame | None
    friend_safari: pd.DataFrame | None
    gift: pd.DataFrame | None
    misc: pd.DataFrame | None
    pickup_item: pd.DataFrame | None
    trade: pd.DataFrame | None
    wild: pd.DataFrame | None
    wild_item: pd.DataFrame | None

    DATA = ["breed", "buy", "evolve", "forms_change", "forms_ingame", "forms_transfer",
            "fossil", "friend_safari", "gift", "misc", "pickup_item", "trade", "wild", "wild_item"]

    def __init__(self, gen: int):
        self.id = gen
        self.dir = Path("data") / f"gen{self.id}"
        self.games = set(GAMES.loc[GAMES.GENERATION == self.id, "GAMEID"])

        dex = pd.read_csv(self.dir / "dex.csv").fillna(False)
        dex['FORM_IDX'] = dex.groupby('SPECIES').cumcount()
        dex['POKEMON']: Series = dex.apply(lambda r: Pokemon.new(r.SPECIES, r.FORM, r.IDX, form_idx=r.FORM_IDX), axis=1) # type: ignore
        self.pokemon_list = dex.set_index('POKEMON', drop=True)
        self.pokemon_list = self.pokemon_list.drop(['SPECIES', 'FORM', 'IDX', 'FORM_IDX'], axis=1)
        self.pokemon_list.loc[self.pokemon_list.GENDER == False, 'GENDER'] = 'BOTH'

        self.items = pd.read_csv(self.dir / "item.csv").fillna(False).set_index("ITEM").convert_dtypes()
 
        for datum in Generation.DATA:
            fname = self.dir / f"{datum}.csv"
            if not fname.exists():
                setattr(self, datum, None)
                continue
            df = pd.read_csv(fname)
            for game in self.games:
                if game in df.columns:
                    df[game] = df[game].fillna(0)
            for col in df.columns:
                fill = False
                if col.split('.')[0] in self.games:
                    fill = 0
                df[col] = df[col].fillna(fill)
            setattr(self, datum, df)


    def genders(self, pokemon: Pokemon) -> list[Gender]:
        '''Return the list of possible gender suffixes.'''
        gender: str = self.pokemon_list.loc[pokemon, "GENDER"] # type: ignore
        if gender == "BOTH":
            return [Gender.MALE, Gender.FEMALE]
        elif gender == "MALE":
            return [Gender.MALE]
        elif gender == "FEMALE":
            return [Gender.FEMALE]
        elif gender == "UNKNOWN":
            return [Gender.UNKNOWN]
        else:
            raise ValueError(f"Unknown species gender label {gender}")


class GameSave():
    def __init__(self, cartridge: Cartridge, generation: Generation):
        self.cartridge = cartridge
        self.game = cartridge.game
        self.generation = generation

        self.has_gender = self.game.gen >= 2 and "NO_BREED" not in self.game.props

        self.pokemon_list = self.generation.pokemon_list.copy()
        self.pokemon_list = self.pokemon_list[self.pokemon_list.apply(lambda r: self.game.match(r.GAME), axis=1)].drop('GAME', axis=1)
        self.all_species: set[str] = {p.species for p in self.pokemon_list.index}
        self.transferable_pokemon: set[Pokemon] = {p for p in self.pokemon_list.index if self.pokemon_list.loc[p, 'TAG'] != "NOTRANSFER"}
        self.tradeable_pokemon: set[Pokemon] = {p for p in self.transferable_pokemon if self.pokemon_list.loc[p, 'TAG'] != "NOTRADE"} # type: ignore
        items = self.generation.items
        if "GAME" in items:
            items = items[items.apply(lambda r: self.game.match(r.GAME), axis=1)].drop(labels='GAME', axis=1)
        self.items = set(items.index)
        self.tradeable_items = set(items[~items.KEY].index)

        # Initialized later - indicates what other game states this one can communicate with
        self.transfers: dict[str, set[GameSave]] = {
            "TRADE": set(), # Trade Pokemon (possible w/held items)
            "POKEMON": set(), # Transfer Pokemon w/o trading
            "ITEMS": set(), # Transfer items
            "MYSTERYGIFT": set(), # Mystery gift
            "RECORDMIX": set(), # Gen 3 record mixing
            "CONNECT": set(), # Miscellaneous
        }

        # These are used to initialize what will be the final rules.
        self.evolutions: dict[Pokemon, dict[PokemonReq, set[PokemonEntry]]] = {pokemon: {} for pokemon in self.pokemon_list.index}
        self.breeding: dict[Pokemon, dict[PokemonReq, set[PokemonEntry]]] = {pokemon: {} for pokemon in self.pokemon_list.index}
        self.unique_pokemon: dict[Pokemon, set[PokemonEntry]] = {pokemon: set() for pokemon in self.pokemon_list.index}
        self.sf2pokemon: dict[tuple[str, str | None], Pokemon] = {(p.species, p.form): p for p in self.pokemon_list.index}
        self.transfer_forms: dict[Pokemon, Pokemon] = {}

    def get_pokemon(self, misc: PokemonEntry | FormDexEntry | PokemonReq | tuple[str, str | None] | str) -> Pokemon:
        if isinstance(misc, PokemonEntry) or isinstance(misc, FormDexEntry) or isinstance(misc, PokemonReq):
            return self.sf2pokemon[(misc.species, misc.form)]
        if isinstance(misc, tuple):
            return self.sf2pokemon[misc]
        if isinstance(misc, str):
            split = misc.split('_')
            species = split[0]
            form = split[1] if len(split) > 1 else None
            return self.sf2pokemon[(species, form)]

    def transfer(self, other_cartridge, category):
        '''Register another cartridge as able to receive certain communications'''
        self.transfers[category].add(other_cartridge)

    def init_evolutions(self):
        if self.generation.evolve is None or "NO_EVOLVE" in self.game.props or "NO_DEX" in self.game.props:
            return

        for _, row in self.generation.evolve.iterrows():
            if "GAME" in row and not self.game.match(row.GAME):
                continue
            self._init_single_evo(row.FROM, row.TO, {"NOEVOLVE"})
        
        # Including form changes in evolutions
        if self.generation.forms_change is None:
            return
        for _, row in self.generation.forms_change.iterrows():
            if "GAME" in row and not self.game.match(row.GAME):
                continue
            in_str = row.SPECIES
            out_str = row.SPECIES
            if row.IN_FORM:
                in_str += f'_{row.IN_FORM}'
            if row.OUT_FORM:
                out_str += f'_{row.OUT_FORM}'
            self._init_single_evo(in_str, out_str)
    
    def _init_single_evo(self, pre_str: str, post_str: str, forbidden_props={}):
        pre = self.parse_pokemon_input(pre_str, forbidden=forbidden_props)
        post = self.parse_pokemon_output(post_str, use_gender=False)[0]
        pre_pokemon = self.get_pokemon(pre)
        if pre not in self.evolutions[pre_pokemon]:
            self.evolutions[pre_pokemon][pre] = set()
        for p in post:
            self.evolutions[pre_pokemon][pre].add(p)

    def init_breeding(self):
        if self.generation.breed is None or "NO_BREED" in self.game.props or "NO_DEX" in self.game.props:
            return

        for _, row in self.generation.breed.iterrows():
            parent = self.parse_pokemon_input(row.PARENT, forbidden={"NOBREED"})
            parent_pokemon = self.get_pokemon(parent)
            if parent not in self.breeding[parent_pokemon]:
                self.breeding[parent_pokemon][parent] = set()
            for child in self.parse_pokemon_output(row.CHILD):
                self.breeding[parent_pokemon][parent].add(child[0])
    
    def init_transfer_forms(self) -> None:
        if self.generation.forms_transfer is None or "NO_DEX" in self.game.props:
            return
        
        for _, row in self.generation.forms_transfer.iterrows():
            if not self.game.match(row.GAME):
                continue
            get_split: list[str] = row.GET.split('_')
            get_species = get_split[0]
            get_form = None if len(get_split) == 1 else get_split[1]
            becomes_split: list[str] = row.BECOMES.split('_')
            becomes_species = becomes_split[0]
            assert get_species == becomes_species
            becomes_form = None if len(becomes_split) == 1 else becomes_split[1]
            becomes_pokemon = self.get_pokemon((becomes_species, becomes_form))
            get_pokemon = replace(becomes_pokemon, form=get_form)
            self.transfer_forms[get_pokemon] = becomes_pokemon

    def init_unique_pokemon(self):
        if self.generation.buy is not None:
            for _, row in self.generation.buy.iterrows():
                if not self.game.match(row.GAME):
                    continue
                for os in self.parse_output(row.POKEMON_OR_ITEM):
                    for o in os:
                        if isinstance(o, PokemonEntry):
                            self.add_unique(o)

        if self.generation.fossil is not None:
            for _, row in self.generation.fossil.iterrows():
                if not self.game.match(row.GAME):
                    continue
                for os in self.parse_pokemon_output(row.POKEMON):
                    for o in os:
                        self.add_unique(o)

        if self.generation.gift is not None:
            for _, row in self.generation.gift.iterrows():
                if not self.game.match(row.GAME):
                    continue
                for choice in self.parse_gift_entry(row.POKEMON_OR_ITEM):
                    for o in choice:
                        if isinstance(o, PokemonEntry):
                            self.add_unique(o)

        if self.generation.trade is not None:
            for _, row in self.generation.trade.iterrows():
                for os in self.parse_pokemon_output(row.GET):
                    for o in os:
                        self.add_unique(o)
        
        if self.generation.wild is not None:
            cols = [c for c in self.generation.wild.columns if self.game.match(c)]
            if len(cols) > 1:
                raise ValueError("Multiple column entries matched game {self.game}: {cols}")
            if len(cols) == 1:
                gamecol = cols[0]
                for _, row in self.generation.wild.iterrows():
                    if row[gamecol]:
                        for os in self.parse_output(row.SPECIES):
                            for o in os:
                                if isinstance(o, PokemonEntry):
                                    self.add_unique(o)
                        
    def add_unique(self, pokemon_entry: PokemonEntry):
        pokemon_entry = replace(pokemon_entry, cart_id=self.cartridge.id)
        pokemon = self.get_pokemon(pokemon_entry)
        if pokemon_entry in self.unique_pokemon[pokemon]:
            return

        self.unique_pokemon[pokemon].add(pokemon_entry)

        for preq, posts in self.evolutions[pokemon].items():
            if preq.matches(pokemon_entry):
                for post in posts:
                    if self.has_gender:
                        post_genders = self.generation.genders(self.get_pokemon(post))
                        if pokemon_entry.gender in post_genders:
                            post_gender = pokemon_entry.gender
                        elif len(post_genders) > 1:
                            raise ValueError(f"Gender mismatch between '{preq}' and '{post}' evolution")
                        else: # Shedinja
                            post_gender = post_genders[0]
                    else:
                        post_gender = None
                    new_p = replace(post, gender=post_gender, props=pokemon_entry.props.union(post.props))
                    self.add_unique(new_p)

        for preq, ps in self.breeding[pokemon].items():
            if preq.matches(pokemon_entry):
                for p in ps:
                    self.add_unique(p)

        if "NOTRANSFER" not in pokemon_entry.props:
            transfer_to = self.transfers["TRADE"].union(self.transfers["POKEMON"])
            if "TRANSFERRESET" in pokemon_entry.props:
                pokemon_entry = replace(pokemon_entry, props=frozenset())
            for gs in transfer_to:
                if pokemon in gs.unique_pokemon:
                    gs.add_unique(pokemon_entry)
                elif pokemon in gs.transfer_forms:
                    transfer_pokemon = gs.transfer_forms[pokemon]
                    gs.add_unique(replace(pokemon_entry, species=transfer_pokemon.species, form=transfer_pokemon.form))

    def get_rules(self) -> list[tuple["Rule", float]]:
        global _choice_idx
        rules = []

        if self.generation.breed is not None and "NO_BREED" not in self.game.props and "NO_DEX" not in self.game.props:
            ditto = self.parse_pokemon_input('Ditto', forbidden={"NOBREED"})
            for _, row in self.generation.breed.iterrows():
                pr = self.parse_pokemon_input(row.PARENT, forbidden={"NOBREED"})
                for gender in self.generation.genders(self.get_pokemon(pr)): 
                    pg = replace(pr, gender=gender)
                    for c in self.parse_output(row.CHILD):
                        if len(c) > 1:
                            raise ValueError(f"Invalid child entry {c}")
                        c = c[0]
                        required: set[Req] = {pg}
                        if gender != Gender.FEMALE:
                            required.add(ditto)
                        if "ITEM" in row and row.ITEM:
                            required.add(ItemEntry(self.cartridge.id, row.ITEM))
                        rules += self._multi_rules(set(), required, {c})

        if self.generation.buy is not None:
            for _, row in self.generation.buy.iterrows():
                if not self.game.match(row.GAME):
                    continue
                exchange, _ = self.parse_input(row.get("EXCHANGE"))
                req, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                for o in self.parse_output(row.POKEMON_OR_ITEM):
                    if len(o) != 1:
                        raise ValueError(f"Invalid buy entry {o}")
                    buy_consumed = set(exchange) if exchange else set()
                    buy_required = set(req) if req else set()
                    rules += self._multi_rules(buy_consumed, buy_required, {o[0]})

        trade_evos: dict[Pokemon, dict[str | None, Pokemon]] = {}
        trade_evos_by_item: dict[str | None, dict[Pokemon, Pokemon]] = {}
        trade_evo_pairs: dict[Pokemon, tuple[Pokemon, Pokemon]] = {}

        if self.generation.evolve is not None and "NO_EVOLVE" not in self.game.props and "NO_DEX" not in self.game.props:
            for _, row in self.generation.evolve.iterrows():
                if "GAME" in row and not self.game.match(row.GAME):
                    continue
                other = self.parse_pokemon_input(row.OTHER_POKEMON) if row.get("OTHER_POKEMON") else None
                if row.get("TRADE"):
                    item_str = row.get("ITEM") or None
                    from_pokemon = self.get_pokemon(self.parse_pokemon_output(row.FROM)[0][0])
                    to_pokemon = self.get_pokemon(self.parse_pokemon_output(row.TO)[0][0])
                    if other:
                        other_pokemon = self.get_pokemon(other)
                        if item_str:
                            raise ValueError("Not handled")
                        trade_evo_pairs[from_pokemon] = (to_pokemon, other_pokemon)
                        continue
                    if from_pokemon not in trade_evos:
                        trade_evos[from_pokemon] = {}
                    trade_evos[from_pokemon][item_str] = to_pokemon
                    if item_str not in trade_evos_by_item:
                        trade_evos_by_item[item_str] = {}
                    trade_evos_by_item[item_str][from_pokemon] = to_pokemon
                    continue
                
                pre = self.parse_pokemon_input(row.FROM, forbidden={"NOEVOLVE"})
                post = self.parse_pokemon_output(row.TO, use_gender=False)
                item = ItemEntry(self.cartridge.id, row.ITEM) if row.ITEM else None
                if len(post) != 1:
                    raise ValueError(f"Invalid evolution entry {post}")
                reqs = set()
                if other is not None:
                    reqs.add(other)
                rules += self._evo_rules(pre, post[0], item, reqs)

        if self.generation.forms_change is not None:
            for _, row in self.generation.forms_change.iterrows():
                if not self.game.match(row.GAME):
                    continue
                reqs, _ = self.parse_input(row.get("REQUIRED"))
                species: str = row.SPECIES
                in_form: str | None = row.IN_FORM or None
                out_form: str | None = row.OUT_FORM or None
                in_pokemon_req = PokemonReq(species, in_form)
                out_pokemon_entry = PokemonEntry(self.cartridge.id, species, out_form, None)
                rules += self._evo_rules(in_pokemon_req, [out_pokemon_entry], None, reqs)

        if self.generation.forms_ingame is not None:
            for _, row in self.generation.forms_ingame.iterrows():
                if not self.game.match(row.GAME):
                    continue
                reqs, _ = self.parse_input(row.REQUIRED)
                out_pokemon = self.get_pokemon(row.FORM)
                out = {
                    SpeciesDexEntry.new(self.cartridge.id, out_pokemon),
                    FormDexEntry.new(self.cartridge.id, out_pokemon),
                }
                rules += self._multi_rules([], reqs, out)

        if self.generation.fossil is not None:
            for _, row in self.generation.fossil.iterrows():
                if not self.game.match(row.GAME):
                    continue
                reqs, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                for p in self.parse_output(row.POKEMON):
                    if len(p) != 1:
                        raise ValueError(f"Unexpected fossil {p}")
                    rules += self._multi_rules({ItemEntry(self.cartridge.id, row.ITEM)}, reqs, p)

        if self.generation.gift is not None:
            for idx, row in self.generation.gift.iterrows():
                if not self.game.match(row.GAME):
                    continue
                reqs, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                choices = self.parse_gift_entry(row.POKEMON_OR_ITEM)
                if len(choices) == 1:
                    rules += self._multi_rules(set(), reqs, set(choices[0]), 1)
                else:
                    _choice_idx += 1
                    choice = ChoiceEntry(self.cartridge.id, f"gift:{_choice_idx}")
                    rules += self._multi_rules(set(), reqs, {choice}, 1)
                    for pokemon_or_items in choices:
                        rules += self._multi_rules({choice}, set(), pokemon_or_items, 1)

        if self.generation.misc is not None:
            for _, row in self.generation.misc.iterrows():
                if not self.game.match(row.GAME):
                    continue
                cons, _ = self.parse_input(row.get("CONSUMED"))
                reqs, _ = self.parse_input(row.get("REQUIRED"))
                output = self.parse_output(row.OUTPUT)
                repeats = int(row.REPEATS)
                rules.append((Rule(
                    frozenset(cons) if cons else frozenset(), # type: ignore
                    frozenset(reqs) if reqs else frozenset(), # type: ignore
                    frozenset(output[0])), repeats))

        if self.generation.pickup_item is not None and "NO_DEX" not in self.game.props:
            # Assumes it's possible to obtain a Pokemon with Pickup in any game w/abilities
            for _, row in self.generation.pickup_item.iterrows():
                if not self.game.match(row.GAME):
                    continue
                rules += self._multi_rules(set(), set(), {ItemEntry(self.cartridge.id, row.ITEM)})
        
        if self.generation.trade is not None:
            for _, row in self.generation.trade.iterrows():
                if not self.game.match(row.GAME):
                    continue
                reqs, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                if row.GIVE == "ANY":
                    gives = [None]
                    give_pokemon = None
                else:
                    give = self.parse_pokemon_input(row.GIVE, forbidden={"NOTRANSFER"}) 
                    give_pokemon = self.get_pokemon(give)
                    gives = [p for p in self.unique_pokemon[give_pokemon] if give.matches(p)]
                gets = self.parse_pokemon_output(row.GET)

                get_pokemon = self.get_pokemon(gets[0][0])
                item = row.get("ITEM") or None
                choice = None
                if len(gets) * len(gives) > 1:
                    _choice_idx += 1
                    choice = ChoiceEntry(self.cartridge.id, f"trade_{row.GET}:{_choice_idx}")
                    rules += self._multi_rules(set(), reqs, {choice}, 1)
                    reqs = set()

                if get_pokemon in trade_evo_pairs:
                    _, other = trade_evo_pairs[get_pokemon]
                    if (give_pokemon in [None, other]) and item != "Everstone":
                        raise ValueError("Not handled")

                evolution = None
                if get_pokemon in trade_evos:
                    if item and item in trade_evos[get_pokemon]:
                        evolution = trade_evos[get_pokemon][item]
                        item = None
                    elif None in trade_evos[get_pokemon]:
                        if self.game.gen == 3 and self.game.core:
                            # Held item loss glitch - item is lost/ignored, including Everstone
                            item = None
                        # Everstone doesn't stop Kadabra from evolving since gen 4
                        if item != "Everstone" or (self.game.gen >= 4 and get_pokemon.species == "Kadabra"):
                            evolution = trade_evos[get_pokemon][None]

                for give in gives:
                    for get in gets:
                        if len(get) != 1:
                            raise ValueError(f"Invalid trade entry {get}")
                        get = get[0]
                        get_p = self.get_pokemon(get)

                        evo_consumed: set[Entry] = {give} if give else set()
                        if choice:
                            evo_consumed.add(choice)
                        evo_output: set[Entry] = {
                            PokemonEntry.new(self.cartridge.id, evolution, get.gender, get.props),
                            SpeciesDexEntry.new(self.cartridge.id, get_p),
                            FormDexEntry.new(self.cartridge.id, get_p),
                        } if evolution else {get}
                        if item:
                            evo_output.add(ItemEntry(self.cartridge.id, item))
                        
                        rules += self._multi_rules(evo_consumed, reqs, evo_output, repeats=1)

        wild_items = {}
        if self.generation.wild_item is not None:
            for _, row in self.generation.wild_item.iterrows():
                if not self.game.match(row.GAME):
                    continue
                if row.SPECIES not in wild_items:
                    wild_items[row.SPECIES] = []
                wild_items[row.SPECIES].append(row.ITEM)

        if self.generation.wild is not None:
            cols = [c for c in self.generation.wild.columns if self.game.match(c)]
            if len(cols) > 1:
                raise ValueError(f"Multiple column entries matched game {self.game}: {cols}")
            if len(cols) == 1:
                gamecol = cols[0]
                for _, row in self.generation.wild.iterrows():
                    pokemon_or_item = row.SPECIES
                    items = wild_items.get(pokemon_or_item) or [None]
                    p_or_is: list[tuple[Entry]] = []
                    for consumed, reqs, count in self.parse_wild_entry(row[gamecol]):
                        if count == 0:
                            continue
                        if count != math.inf and len(items) * len(p_or_is) > 1 and not consumed:
                            _choice_idx += 1
                            choice = ChoiceEntry(self.cartridge.id, f"wild_{pokemon_or_item}:{_choice_idx}")
                            rules += self._multi_rules(consumed, reqs, {choice}, count)
                            consumed = {choice}
                            reqs = set()
                        if not p_or_is:
                            p_or_is = self.parse_output(pokemon_or_item)
                        for p_or_i in p_or_is:
                            if len(p_or_i) != 1:
                                raise ValueError("Confused")
                            for item in items:
                                out = {p_or_i[0]}
                                if item:
                                    out.add(ItemEntry(self.cartridge.id, item))
                                rules += self._multi_rules(consumed, reqs, out, count)

        for pokemon in self.unique_pokemon:
            if pokemon not in self.transferable_pokemon:
                continue
            for pokemon_entry in self.unique_pokemon[pokemon]:
                if "NOTRANSFER" in pokemon_entry.props:
                    continue
                for gs in self.transfers["POKEMON"]:
                    out_pokemon_entry = replace(pokemon_entry, cart_id=gs.cartridge.id)
                    if pokemon in gs.transfer_forms:
                        changed = gs.transfer_forms[pokemon]
                        out_pokemon_entry = replace(out_pokemon_entry, species=changed.species, form=changed.form)
                    elif pokemon not in gs.unique_pokemon:
                        continue

                    if "TRANSFERRESET" in pokemon_entry.props:
                        out_pokemon_entry = replace(out_pokemon_entry, props=frozenset())
                    rules.append((self._transfer_rule(gs.cartridge.id, {pokemon_entry}, {out_pokemon_entry}), math.inf))

                has_noitem_evo = False
                if pokemon not in self.tradeable_pokemon:
                    continue
                for gs in self.transfers["TRADE"]:
                    out_pokemon_entry = replace(pokemon_entry, cart_id=gs.cartridge.id)
                    if pokemon in gs.transfer_forms:
                        changed = gs.transfer_forms[pokemon]
                        out_pokemon_entry = replace(out_pokemon_entry, species=changed.species, form=changed.form)
                    elif pokemon not in gs.unique_pokemon:
                        continue
                    if "TRANSFERRESET" in pokemon_entry.props:
                        out_pokemon_entry = replace(out_pokemon_entry, props=frozenset())
                    for item, evo_pokemon in trade_evos.get(pokemon, {}).items():
                        if item is None and "NOEVOLVE" not in out_pokemon_entry.props:
                            has_noitem_evo = True
                        assert evo_pokemon in self.tradeable_pokemon, "Evolution should also be tradeable"
                        if evo_pokemon not in gs.unique_pokemon:
                            continue
                        if "NOEVOLVE" in out_pokemon_entry.props:
                            continue
                        evo_pokemon_entry = replace(out_pokemon_entry, cart_id=gs.cartridge.id, species=evo_pokemon.species, form=evo_pokemon.form)
                        if item:
                            rules.append((self._transfer_rule(gs.cartridge.id, {pokemon_entry, ItemEntry(self.cartridge.id, item)}, {evo_pokemon_entry}), math.inf))
                        else:
                            rules.append((self._transfer_rule(gs.cartridge.id, {pokemon_entry}, {evo_pokemon_entry}), math.inf))
                            if (pokemon.species != "Kadabra" or self.generation.id == 2) and self.generation.id != 3:
                                rules.append((self._transfer_rule(gs.cartridge.id, {pokemon_entry, ItemEntry(self.cartridge.id, "Everstone")}, {out_pokemon_entry}), math.inf))
                    if pokemon in trade_evo_pairs and "NOEVOLVE" not in out_pokemon_entry.props:
                        evo_pokemon, pokemon2 = trade_evo_pairs[pokemon]
                        evo_pokemon2 = trade_evo_pairs[pokemon2][0]
                        evo_pokemon_entry = replace(out_pokemon_entry, cart_id=gs.cartridge.id, species=evo_pokemon.species)
                        for pokemon2_entry in self.unique_pokemon[pokemon2]:
                            if "NOEVOLVE" in pokemon2_entry.props:
                                continue
                            evo_pokemon2_entry = replace(pokemon2_entry, cart_id=gs.cartridge.id, species=evo_pokemon2.species)
                            if "TRANSFERRESET" in pokemon2_entry.props:
                                evo_pokemon2_entry = replace(evo_pokemon2_entry, props=frozenset())
                            rules.append((self._transfer_rule(gs.cartridge.id, {pokemon_entry, pokemon2_entry}, {evo_pokemon_entry, evo_pokemon2_entry}), math.inf))
                    if not has_noitem_evo:
                        rules.append((self._transfer_rule(gs.cartridge.id, {pokemon_entry}, {out_pokemon_entry}), math.inf))
                

        for item in self.tradeable_items:
            for gs in self.transfers["ITEMS"]:
                if item not in gs.tradeable_items:
                    continue
                rules.append((self._transfer_rule(gs.cartridge.id, {ItemEntry(self.cartridge.id, item)}, {ItemEntry(gs.cartridge.id, item)}), math.inf))

            if "NO_TRADE_ITEMS" not in self.game.props:
                for gs in self.transfers["TRADE"]:
                    if "NO_TRADE_ITEMS" in gs.game.props:
                        continue
                    if item not in gs.tradeable_items:
                        continue
                    rules.append((self._transfer_rule(gs.cartridge.id, {ItemEntry(self.cartridge.id, item)}, {ItemEntry(gs.cartridge.id, item)}), math.inf))

            for gs in self.transfers["MYSTERYGIFT"]:
                rules.append((self._transfer_rule(gs.cartridge.id, {ChoiceEntry(self.cartridge.id, f"MG_{item}")}, {ItemEntry(gs.cartridge.id, item)}), math.inf))

        for gs in self.transfers["RECORDMIX"]:
            rules.append((self._transfer_rule(gs.cartridge.id, {ChoiceEntry(self.cartridge.id, "RM_Eon Ticket")}, {ItemEntry(gs.cartridge.id, "Eon Ticket")}), math.inf))
        
        return rules

    def _multi_rules(self, consumed: Iterable[Req], required: Iterable[Req], output: Iterable[Entry], repeats=math.inf) -> set[tuple["Rule", float]]:
        '''
        PokemonReq may appear in consumed/required. In this case, create a rule for all combinations of valid PokemonEntries
        '''
        rules: set[tuple["Rule", float]] = set()
        new_consumed = []
        new_required = []
        input_pokemon = set()
        input_species = set()
        for inp, new_inp in [(consumed, new_consumed), (required, new_required)]:
            for i in inp:
                if isinstance(i, PokemonReq):
                    ipokemon = self.get_pokemon(i)
                    input_pokemon.add((self.cartridge.id, ipokemon))
                    input_species.add((self.cartridge.id, ipokemon.species))
                    matches = [p for p in self.unique_pokemon[ipokemon] if i.matches(p)]
                    if not matches:
                        return rules
                    new_inp.append(matches)
                elif isinstance(i, PokemonEntry):
                    input_pokemon.add((i.cart_id, self.get_pokemon(i)))
                    input_species.add((i.cart_id, i.species))
                    new_inp.append([i])
                else:
                    new_inp.append([i])
        new_consumed = list(product(*new_consumed))
        new_required = list(product(*new_required))

        full_output = set(output)
        for o in output:
            if isinstance(o, PokemonEntry):
                o_pokemon = self.get_pokemon(o)
                if (o.cart_id, o.species) not in input_species:
                    full_output.add(SpeciesDexEntry.new(o.cart_id, o_pokemon))
                if (o.cart_id, o_pokemon) not in input_pokemon:
                    full_output.add(FormDexEntry.new(o.cart_id, o_pokemon))
        full_output = frozenset(full_output)

        combos = list(product(new_consumed, new_required))
        choice = None
        if len(combos) > 1 and repeats != math.inf:
            global _choice_idx
            _choice_idx += 1
            choice = ChoiceEntry(self.cartridge.id, f"choice:{_choice_idx}")
            rules.add((Rule(frozenset(), frozenset(), frozenset({choice})), repeats))
            repeats = math.inf
        for consumed, required in combos:
            c = set(consumed)
            if choice is not None:
                c.add(choice)
            c = frozenset(c)
            r = frozenset(required)
            rules.add((Rule(c, r, full_output), repeats))
        return rules

    def _evo_rules(self, pre_req: PokemonReq, post_entries: Iterable[PokemonEntry], item: ItemEntry | None=None, reqs: Iterable[Req] = set()) -> set[tuple["Rule", float]]:
        rules: set[tuple["Rule", float]] = set()
        pre_pokemon = self.get_pokemon(pre_req)
        for pre_entry in self.unique_pokemon[pre_pokemon]:
            if not pre_req.matches(pre_entry):
                continue
            out: list[Entry] = []
            for post_entry in post_entries:
                post_pokemon = self.get_pokemon(post_entry)
                if pre_entry.gender is None:
                    post_gender = None
                else:
                    post_genders = self.generation.genders(post_pokemon)
                    if pre_entry.gender in post_genders:
                        post_gender = pre_entry.gender
                    elif post_pokemon.species == "Shedinja":
                        post_gender = post_genders[0]
                    else:
                        raise ValueError(f"Gender mismatch between '{pre_pokemon}' and '{post_pokemon}' evolution")
                out.append(replace(post_entry, gender=post_gender, props=pre_entry.props.union(post_entry.props)))
            consumed: set[Entry] = {pre_entry}
            if item is not None:
                consumed.add(item)
            # Need to call _multi_rules since required might contain a PokemonReq
            rules |= self._multi_rules(consumed, set(reqs), out)
        return rules
    
    def _transfer_rule(self, out_cart_id: str, inputs: Iterable[Entry], outputs: Iterable[Entry]) -> "Rule":
        new_outputs = set(outputs)
        for entry in chain(inputs, outputs):
            if isinstance(entry, PokemonEntry):
                pokemon = self.get_pokemon(entry)
                new_outputs.add(SpeciesDexEntry.new(out_cart_id, pokemon))
                new_outputs.add(FormDexEntry.new(out_cart_id, pokemon))
        
        return Rule(frozenset(), frozenset(inputs), frozenset(new_outputs), is_transfer=True)

    
    def parse_pokemon_input(self, entry, forbidden: set[str] | None = None) -> PokemonReq:
        split = entry.split('_')
        species = split[0]
        gender = None
        props = set(split[1:])
        forms = set(p for p in props if p.upper() != p or len(p) == 1 or '%' in p) or {None}
        if len(forms) != 1:
            raise ValueError(f"Multiple forms {forms} parsed from {entry}")
        props = props - forms
        form = forms.pop()
        try:
            pokemon = self.get_pokemon((species, form))
        except KeyError:
            form_str = "" if form is None else f" ({form})"
            raise ValueError(f"Invalid Pokemon {species}{form_str}")
        if "MALE" in props:
            props.remove("MALE")
            gender = Gender.MALE
        elif "FEMALE" in props:
            props.remove("FEMALE")
            gender = Gender.FEMALE

        if "ONE" in props:
            props.remove("ONE")
            props.add(f"ONE_{self.cartridge.id}")

        return PokemonReq(species, form, gender, frozenset(props), frozenset(forbidden or []))

    def parse_input(self, entry: str | None, forbidden: set[str] | None = None) -> tuple[list[Req], bool]:
        out: list[Req] = []
        valid = True
        if not entry:
            return out, True

        for e in entry.split(','):
            try:
                pr = self.parse_pokemon_input(e, forbidden)
                out.append(pr)
                continue
            except ValueError:
                pass
            if e in self.items:
                out.append(ItemEntry(self.cartridge.id, e))
            elif e.split('.')[0] in GAMEIDS:
                if not any([gs.cartridge != self.cartridge and gs.game.match(e) for gs in self.transfers["CONNECT"].union(self.transfers["RECORDMIX"])]): 
                    valid = False
            elif e.startswith("DEX_"):
                out += [SpeciesDexEntry.new(self.cartridge.id, p) for p in self.pokemon_list.index if self.pokemon_list.loc[p, e]]
            elif e.startswith("$"):
                out.append(ChoiceEntry(self.cartridge.id, e[1:]))
            else:
                raise ValueError(f"Unrecognized entry {e}")
        return out, valid

    def parse_output(self, entry: str, use_gender=True) -> list[tuple[Entry]]:
        out = []
        for e in entry.split(','):
            if e.split('_')[0] in self.all_species:
                out.append(self._parse_pokemon_output_entry(e, use_gender))
            elif e in self.items:
                out.append([ItemEntry(self.cartridge.id, e)])
            elif e.startswith("$"):
                out.append([ChoiceEntry(self.cartridge.id, e[1:])])
            else:
                raise ValueError(f"Unrecognized entry {e}")
        return list(product(*out))
    
    def parse_pokemon_output(self, entry: str, use_gender=True) -> list[tuple[PokemonEntry]]:
        out = []
        for e in entry.split(','):
            out.append(self._parse_pokemon_output_entry(e, use_gender))
        return list(product(*out))
    
    def _parse_pokemon_output_entry(self, entry: str, use_gender=True) -> list[PokemonEntry]:
        use_gender = use_gender and self.has_gender
        split = entry.split('_')
        species = split[0]
        assert species in self.all_species, f"Invalid species {species}"
        props = set(split[1:])
        forms = set(p for p in props if p.upper() != p or len(p) == 1 or '%' in p) or {None}
        props = props - forms
        if len(forms) != 1:
            raise ValueError(f"Multiple forms: {forms}")
        form = forms.pop()
        try:
            pokemon = self.get_pokemon((species, form))
        except KeyError:
            raise ValueError(f"Unrecognized form {form} for species {species}")

        genders: Sequence[Gender | None] = [None]
        if "MALE" in props:
            props.remove("MALE")
            if use_gender:
                genders = [Gender.MALE]
        elif "FEMALE" in props:
            props.remove("FEMALE")
            if use_gender:
                genders = [Gender.FEMALE]
        elif use_gender:
            genders = self.generation.genders(pokemon)
        props = frozenset(props)

        return [PokemonEntry(self.cartridge.id, species, form, g, props) for g in genders]

    
    def parse_gift_entry(self, entry: str, use_gender=True) -> list[tuple[Entry]]:
        out: list[tuple[Entry]] = []
        for e in entry.split(';'):
            out += self.parse_output(e, use_gender)
        return out

    def parse_wild_entry(self, entry: float | int | str) -> list[tuple[list[Req], list[Req], float]]:
        if isinstance(entry, float) or isinstance(entry, int):
            return [([], [], float(entry))]
        out = []
        for e in entry.split(';'):
            count, _, con_entry = e.partition('{')
            if con_entry:
                con, _ = self.parse_input(con_entry[:-1])
                out.append((con, [], float(count)))
            else:
                count, _, req_entry = e.partition('[')
                if req_entry:
                    req, req_valid = self.parse_input(req_entry[:-1])
                    if not req_valid:
                        continue
                    out.append(([], req, float(count)))
                else:
                    out.append(([], [], float(count)))
        return out


class CollectionSaves():
    def __init__(self, collection, mode, generations=None):
        if generations is None:
            generations = {gen: Generation(gen) for gen in set(c.game.gen for c in collection.cartridges)}
        self.generations = generations

        self.game_saves = {c: GameSave(c, generations[c.game.gen]) for c in collection.cartridges}
        self.collection = collection
        self.main_cartridge = collection.main_cartridge

        self.mode = mode

        for kind, cart_pairs in collection.interactions.items():
            for cart1, cart2 in cart_pairs:
                self.game_saves[cart1].transfer(self.game_saves[cart2], kind)

        if any([gs.has_gender for gs in self.game_saves.values()]):
            for gs in self.game_saves.values():
                gs.has_gender = True
        self.pokemon_list = self.game_saves[self.main_cartridge].pokemon_list

        for gs in self.game_saves.values():
            gs.init_evolutions()
            gs.init_breeding()
            gs.init_transfer_forms()
        for gs in self.game_saves.values():
            gs.init_unique_pokemon()

    def calc_dexes(self, flatten=False) -> Result:
        idx2pokemon: dict[int, str] = {}
        for pokemon in self.pokemon_list.index:
            idx2pokemon[pokemon.idx] = pokemon.species
        rule_graph = RuleGraph.make(self._get_rules(), self._get_targets())
        rule_graph.explore()
        branches = rule_graph.try_branches_and_update()
        branches = [{frozenset(entry.to_pokemon() for entry in choice) for choice in branch} for branch in branches]
        present = {k.to_pokemon() for k, v in rule_graph.acquired.items() if v}
        if flatten:
            for branch in branches:
                present = present.union(*branch)
            branches = []
        all_records = {entry.to_pokemon() for entry in self._get_targets()}
        return Result.new(all_records, present, branches)

    def _get_rules(self) -> list[tuple["Rule", float]]:
        rules = []
        for gs in self.game_saves.values():
            gs_rules = gs.get_rules()
            # We create a duplicate of the main cartridge with id 'main_reset', and set can_explore
            # to False for the main cartridge's internal rules.
            # This allows us to simulate resetting the main cartridge in the RuleGraph
            reset_rules = \
                [(r.replace_in_cart_ids(self.main_cartridge.id, "main_reset"), reps) for r, reps in gs_rules if self.main_cartridge.id in r.in_cart_ids() and r.is_transfer] + \
                [(r.replace_out_cart_ids(self.main_cartridge.id, "main_reset"), reps) for r, reps in gs_rules if self.main_cartridge.id in r.out_cart_ids() and r.is_transfer]
            if gs.cartridge == self.main_cartridge:
                reset_rules += [(r.replace_in_cart_ids(self.main_cartridge.id, "main_reset").replace_out_cart_ids(self.main_cartridge.id, "main_reset"), reps) for r, reps in gs_rules if not r.is_transfer]
                gs_rules = [(r if r.is_transfer else replace(r, can_explore=False), reps) for r, reps in gs_rules]
            rules += gs_rules
            rules += reset_rules
        rules += self._friend_safari_rules()
        
        rules = self._handle_console_resets(rules)
        return rules


    def _friend_safari_rules(self) -> list[tuple["Rule", float]]:
        rules = []
        fs_consoles = self.collection.friend_safari_consoles()
        if not fs_consoles or not self.generations.get(6):
            return rules

        fs_pokemon: pd.DataFrame = self.generations[6].friend_safari # type: ignore

        for console, carts in fs_consoles.items():
            cid = f"3DS_{console.id}"
            cart_ids = {cart.id: slot for cart, slot in carts.items()}
            if self.main_cartridge.id in cart_ids:
                cart_ids["main_reset"] = cart_ids[self.main_cartridge.id]
            fs_base = ChoiceEntry(cid, "FS")
            rules.append((Rule(frozenset(), frozenset(), frozenset({fs_base})), 1))
            if console.model.name == "3DSr":
                noreset = ChoiceEntry(cid, "NORESET")
                reset = ChoiceEntry(cid, "RESET")
                rules.append((Rule(frozenset(), frozenset(), frozenset({noreset})), 1))
                rules.append((Rule(frozenset({noreset}), frozenset(), frozenset({reset}), can_explore=False), 1))
                rules.append((Rule(frozenset(), frozenset({reset}), frozenset({fs_base})), math.inf))
            else:
                noreset = None
            for pokemon_type in fs_pokemon['TYPE'].unique():
                rules.append((Rule(
                    frozenset({fs_base}),
                    frozenset(),
                    frozenset({ChoiceEntry(cid, f"FS_{pokemon_type}_{slot}") for slot in range(1, 4)}),
                    can_explore=False), math.inf))
            for _, row in fs_pokemon.iterrows():
                slot = int(row.SLOT)
                gc_in = ChoiceEntry(cid, f"FS_{row.TYPE}_{slot}")
                gc_out = ChoiceEntry(cid, f"FS_{row.TYPE}_{slot}_{row.SPECIES}")
                rules.append((Rule(frozenset({gc_in}), frozenset(), frozenset({gc_out})), math.inf))
                for cart_id, max_cart_slot in cart_ids.items():
                    cart_out = ChoiceEntry(cart_id, f"FS_{row.SPECIES}")

                    reqs = set({gc_out})
                    if slot == 3:
                        if max_cart_slot == 2:
                            continue
                        elif max_cart_slot == 2.5:
                            # 3 only until reset
                            assert(isinstance(noreset, ChoiceEntry))
                            reqs.add(noreset)
                    rules.append((Rule(frozenset(), frozenset(reqs), frozenset({cart_out})), math.inf))
        return rules

    def _handle_console_resets(self, rules: list[tuple["Rule", float]]) -> list[tuple["Rule", float]]:
        '''
        For every software cart that belongs to a console that can be reset, add a requirement to
        every rule involving that cart that the console has not (yet) been reset
        '''
        for console, carts in self.collection.reset_consoles().items():
            assert(console.model.name == "3DSr")
            cid = f"3DS_{console.id}"
            noreset = {ChoiceEntry(cid, "NORESET")}
            for cart in carts:
                rules = [(replace(r, required=(r.required | noreset)) if cart.id in (r.in_cart_ids() | r.out_cart_ids()) else r, repeats) for r, repeats in rules]
        return rules
        

    def _get_targets(self) -> set[DexEntry]:
        if self.mode == 'species':
            cls = SpeciesDexEntry
        else:
            cls = FormDexEntry
        return {cls.new(self.main_cartridge.id, pokemon) for pokemon in self.pokemon_list.index}


class RuleGraph():
    def __init__(self, digraph: nx.DiGraph, acquired: dict[DexEntry, bool], explore_parent: "RuleGraph | None" = None, spaces=0):
        self.G = digraph
        self.acquired = acquired
        self.explore_parent = explore_parent
        self.spaces = spaces

    @classmethod
    def make(cls, rules: Iterable[tuple["Rule", float]], targets: Iterable[DexEntry]) -> Self:
        G = nx.DiGraph()
        G.add_edge("START", "INF")
        G.add_node("END")
        for target in targets:
            if not isinstance(target, DexEntry):
                raise ValueError("Only DexEntry can be targets")
            G.add_edge(target, "END")
            G.nodes[target]['target'] = True
        acquired = {target: False for target in targets}
        rg = cls(G, acquired)
        for rule, repeats in rules:
            rg._add_rule(rule, repeats)
        rg._prune(handle_cycle=False)
        rg._label_cycles()
        return rg

    def copy(self, explore_parent=None) -> "RuleGraph":
        return RuleGraph(self.G.copy(), self.acquired.copy(), explore_parent=explore_parent, spaces=(self.spaces + 2)) # type: ignore

    def _add_rule(self, rule: "Rule", repeats: float):
        G = self.G
        if rule in G:
            G.nodes[rule]['repeats'] += repeats
            return

        for c in rule.consumed:
            G.add_edge(c, rule, consumed=True)
            if 'count' not in self.G.nodes[c]:
                nx.set_node_attributes(G, {c: {'count': 0}})
        for r in rule.required:
            G.add_edge(r, rule, consumed=False)
            if not isinstance(r, DexEntry) and 'count' not in G.nodes[r]:
                nx.set_node_attributes(G, {r: {'count': 0}})
        if not rule.consumed and not rule.required:
            G.add_edge("INF", rule)
        for o in rule.output:
            G.add_edge(rule, o)
            if not isinstance(o, DexEntry) and 'count' not in G.nodes[o]:
                nx.set_node_attributes(G, {o: {'count': 0}})

        G.nodes[rule]['repeats'] = repeats

    def _replace_rule(self, old_rule: "Rule", new_rule: "Rule", handle_cycle=True):
        G = self.G
        if not new_rule.output:
            raise ValueError("Shouldn't happen")
        if new_rule in G.nodes:
            if handle_cycle:
                for entry in G.predecessors(old_rule):
                    if isinstance(entry, str):
                        continue
                    if (entry, new_rule) in G.edges and G.edges[(entry, old_rule)]['cycle']:
                        G.edges[(entry, new_rule)]['cycle'] = True
                for entry in G.successors(old_rule):
                    if isinstance(entry, str):
                        continue
                    if (new_rule, entry) in G.edges and G.edges[(old_rule, entry)]['cycle']:
                        G.edges[(new_rule, entry)]['cycle'] = True
            G.nodes[new_rule]['repeats'] += G.nodes[old_rule]['repeats']
            G.remove_node(old_rule)
        else:
            for c in new_rule.consumed:
                G.add_edge(c, new_rule)
                G.edges[(c, new_rule)]['consumed'] = True
                if handle_cycle:
                    G.edges[(c, new_rule)]['cycle'] = G.edges[(c, old_rule)]['cycle']
            for r in new_rule.required:
                G.add_edge(r, new_rule)
                G.edges[(r, new_rule)]['consumed'] = False
                if handle_cycle:
                    G.edges[(r, new_rule)]['cycle'] = G.edges[(r, old_rule)]['cycle']
            for o in new_rule.output:
                G.add_edge(new_rule, o)
                if handle_cycle:
                    G.edges[(new_rule, o)]['cycle'] = G.edges[(old_rule, o)]['cycle']
            if not new_rule.consumed and not new_rule.required:
                G.add_edge("INF", new_rule)
            G.nodes[new_rule]['repeats'] = G.nodes[old_rule]['repeats']
            G.remove_node(old_rule)


    def explore(self):
        '''
        Make a child and have the child explore among valid rules aside from those in the main cart
        '''
        while True:
            any_transfer_rule = False
            self.apply_safe_rules()
            child = self.copy(explore_parent=self)
            while True:
                rules = list(child._get_valid_rules(explore=True))
                any_rule = False
                for rule in rules:
                    if rule not in child.G.nodes:
                        continue
                    child._apply_rule(rule)
                    any_rule = True
                    if rule.is_transfer:
                        any_transfer_rule = True
                if not any_rule:
                    break
            if not any_transfer_rule:
                break

    def apply_safe_rules(self):
        G = self.G
        while True:
            while True:
                no_consumed = list(G.successors("INF"))
                if len(no_consumed) == 0:
                    break

                for rule in no_consumed:
                    if rule in G.nodes:
                        self._apply_rule(rule)

            any_applied = False
            for rule in list(self._get_safe_rules()):
                if rule in G:
                    any_applied = True
                    self._apply_rule(rule)
            if not any_applied:
                break

    def try_branches_and_update(self) -> list[set[frozenset[DexEntry]]]:
        '''
        Try out all branching paths. For any branching path with only one outcome, simply add the
        outcomes to acquired.
        '''
        out_branches = []
        branches = self.try_branches()
        for branch in branches:
            all_present = frozenset.intersection(*branch)
            if all_present:
                for entry in all_present:
                    self.acquired[entry] = True
                branch = {b - all_present for b in branch}
                branch = {b for b in branch if b}
            if len(branch) == 0:
                continue
            elif len(branch) == 1:
                for entry in branch.pop():
                    self.acquired[entry] = True
                continue
            out_branches.append(branch)
        return out_branches

    def try_branches(self, ignore_special=False) -> list[set[frozenset[DexEntry]]]:
        outcomes = []
        for rg in self._get_components():
            outcomes.append(rg._try_branches_single_component(ignore_special))
        return outcomes

    def _apply_rule(self, rule: "Rule"):
        '''
        Apply rule once if it has any consumed, or as many times as possible if it doesn't.
        '''
        G = self.G
        zero_entries = set()
        zero_repeat_rules = set()
        has_consumed = False

        outputs = list(G.successors(rule))
        inputs = [r for r in G.predecessors(rule) if r != "INF"]
        for i in inputs:
            if G.edges[(i, rule)]['consumed']:
                has_consumed = True
                G.nodes[i]['count'] -= 1
                if G.nodes[i]['count'] == 0:
                    zero_entries.add(i)
                    G.remove_edge("START", i)

        repeats = 1 if has_consumed else G.nodes[rule]['repeats']
        G.nodes[rule]['repeats'] -= repeats
        if G.nodes[rule]['repeats'] == 0:
            zero_repeat_rules.add(rule)

        # Acquire outputs
        # If it's a transfer rule, the parent (if it exists) also gets the outputs
        changes_by_rg: dict["RuleGraph", tuple[set[Entry], set[Entry] | None, set[Rule] | None]] = {self: (set(), zero_entries, zero_repeat_rules)}
        if rule.is_transfer and self.explore_parent is not None:
            changes_by_rg[self.explore_parent] = (set(), None, None)

        for rg, changes in changes_by_rg.items():
            for out in outputs:
                if out not in rg.G.nodes:
                    continue
                if rg.G.nodes[out].get('target', False):
                    rg.acquired[out] = True
                if isinstance(out, DexEntry):
                    changes[0].add(out)
                else:
                    if rg.G.nodes[out]['count'] == 0:
                        rg.G.add_edge("START", out)
                    rg.G.nodes[out]['count'] += repeats
                    if repeats == math.inf or self._not_consumed(out):
                        changes[0].add(out)
            rg._update_maintaining_invariant(*changes)

    def _update_maintaining_invariant(self, inf_entries: Iterable[Entry] | None = None, zero_entries: Iterable[Entry] | None = None, zero_repeat_rules: Iterable["Rule"] | None = None):
        G = self.G
        maybe_unreachable_entries = set()
        maybe_useless_entries = set()
        rules_with_removed_inputs = {}
        rules_with_removed_outputs = {}

        for entry in inf_entries or set():
            for rule in G.successors(entry):
                if not isinstance(rule, Rule):
                    continue
                if rule not in rules_with_removed_inputs:
                    rules_with_removed_inputs[rule] = set()
                rules_with_removed_inputs[rule].add(entry)
            for rule in G.predecessors(entry):
                if not isinstance(rule, Rule):
                    continue
                if rule not in rules_with_removed_outputs:
                    rules_with_removed_outputs[rule] = set()
                rules_with_removed_outputs[rule].add(entry)
            G.remove_node(entry)

        for entry in zero_entries or set():
            maybe_unreachable_entries.add(entry)

        for rule in zero_repeat_rules or set():
            for entry in G.predecessors(rule):
                if not isinstance(entry, Entry):
                    continue
                maybe_useless_entries.add(entry)
            if rule in rules_with_removed_outputs:
                del rules_with_removed_outputs[rule]
            if rule in rules_with_removed_inputs:
                del rules_with_removed_inputs[rule]
            G.remove_node(rule)

        while maybe_unreachable_entries or maybe_useless_entries or rules_with_removed_outputs or rules_with_removed_inputs:
            if rules_with_removed_inputs:
                rule = next(iter(rules_with_removed_inputs))
                repeats = G.nodes[rule]['repeats']
                removed_inputs = rules_with_removed_inputs.pop(rule)
                new_rule = replace(
                    rule,
                    consumed=(rule.consumed - removed_inputs),
                    required=(rule.required - removed_inputs))
                removed_outputs = rules_with_removed_outputs.pop(rule, set())
                if removed_outputs:
                    # Copied from below
                    if not self._is_potentially_useful(rule):
                        for entry in G.predecessors(rule):
                            if isinstance(entry, str):
                                continue
                            maybe_useless_entries.add(entry)
                        G.remove_node(rule)
                        continue
                    else:
                        new_rule = replace(new_rule, output=(new_rule.output - removed_outputs))
                self._replace_rule(rule, new_rule)
                if new_rule in rules_with_removed_outputs:
                    assert rules_with_removed_outputs[new_rule].issubset(removed_outputs)
                    del rules_with_removed_outputs[new_rule]
            elif maybe_unreachable_entries:
                entry = maybe_unreachable_entries.pop()
                if not self._is_probably_reachable(entry):
                #if len(list(G.predecessors(entry))) == 0:
                    for rule in list(G.successors(entry)):
                        if not isinstance(rule, Rule):
                            continue
                        for entry2 in G.successors(rule):
                            if entry2 != entry:
                                maybe_unreachable_entries.add(entry2)
                        G.remove_node(rule)
                        if rule in rules_with_removed_inputs:
                            del rules_with_removed_inputs[rule]
                        if rule in rules_with_removed_outputs:
                            del rules_with_removed_outputs[rule]
                    G.remove_node(entry)
                    if entry in maybe_useless_entries:
                        maybe_useless_entries.remove(entry)
            elif maybe_useless_entries:
                entry = maybe_useless_entries.pop()
                if not self._is_potentially_useful(entry):
                    for rule in G.predecessors(entry):
                        if not isinstance(rule, Rule):
                            continue
                        if rule not in rules_with_removed_outputs:
                            rules_with_removed_outputs[rule] = set()
                        rules_with_removed_outputs[rule].add(entry)
                    G.remove_node(entry)
            elif rules_with_removed_outputs:
                rule = next(iter(rules_with_removed_outputs))
                removed_outputs = rules_with_removed_outputs.pop(rule)
                if not self._is_potentially_useful(rule):
                    for entry in G.predecessors(rule):
                        if isinstance(entry, str):
                            continue
                        maybe_useless_entries.add(entry)
                    G.remove_node(rule)
                else:
                    new_rule = replace(rule, output=(rule.output - removed_outputs))
                    self._replace_rule(rule, new_rule)

    def _is_probably_reachable(self, entry: Entry) -> bool:
        G = self.G
        return nx.has_path(G, "START", entry)

    def _is_potentially_useful(self, entry_or_rule: "Entry | Rule") -> bool:
        G = self.G
        visited_nodes = set()
        to_process = {entry_or_rule}
        while to_process:
            node = to_process.pop()
            for _, node2, data in G.edges(node, data=True):
                if node2 == "END":
                    return True
                if isinstance(node2, str) or node2 in visited_nodes:
                    continue
                if not data['cycle']:
                    return True
                to_process.add(node2)
            visited_nodes.add(node)
        return False

    def _prune(self, handle_cycle=True):
        G = self.G

        nx.set_node_attributes(G, False, "start")
        nx.set_node_attributes(G, {"START": {"start": True}})
        visited_entries = set()
        entries = set()
        rules = {"START"}
        while entries or rules:
            if entries:
                entry = entries.pop()
                visited_entries.add(entry)
                G.nodes[entry]["start"] = True
                for rule in G.successors(entry):
                    rules.add(rule)
            else:
                rule = rules.pop()
                if all(G.nodes[entry]["start"] for entry in G.predecessors(rule)):
                    G.nodes[rule]["start"] = True
                    for entry in G.successors(rule):
                        if entry not in visited_entries:
                            entries.add(entry)
        G.remove_nodes_from([n for n, a in G.nodes(data=True) if not a["start"] and not isinstance(n, str)])

        good_nodes = nx.ancestors(G, "END") | {"START", "INF", "END"}
        # Delete the useless rules first
        G.remove_nodes_from([n for n in G.nodes if isinstance(n, Rule) and n not in good_nodes])
        # Now delete the useless entries
        for entry in list(G.nodes):
            if entry in good_nodes:
                continue
            for rule in list(G.predecessors(entry)):
                new_rule = replace(rule, output=(rule.output - {entry}))
                self._replace_rule(rule, new_rule, handle_cycle=handle_cycle)
            G.remove_node(entry)

    def _label_cycles(self):
        G = self.G
        for n1, n2 in G.edges:
            if isinstance(n1, str) or isinstance(n2, str):
                continue
            if nx.has_path(G, n2, n1):
                G.edges[(n1, n2)]['cycle'] = True
            else:
                G.edges[(n1, n2)]['cycle'] = False

    def _get_valid_rules(self, explore=False):
        '''
        Return iterator of valid rules, i.e. rules whose inputs are satisfied

        If explore is True, limit to rules with can_explore set.
        '''
        G = self.G
        return (r for r in nx.single_source_shortest_path(G, "START", cutoff=2).keys()
                if isinstance(r, Rule) and all(G.has_edge("START", p) for p in G.predecessors(r)) and (r.can_explore or not explore))

    def _get_safe_rules(self):
        '''
        Return iterator of safe rules, i.e. valid rules whose consumed inputs aren't used elsewhere
        '''
        G = self.G
        return (r for r in self._get_valid_rules()
                if all(not G.edges[(p, r)]['consumed'] or len(list(G.successors(p))) == 1
                       for p in G.predecessors(r)))

    def _get_components(self):
        G = self.G
        subgraph = nx.subgraph_view(G, filter_node=(lambda n: not isinstance(n, str)))
        for component in nx.connected_components(subgraph.to_undirected(as_view=True)):
            Gcomponent: nx.DiGraph = G.copy() # type: ignore
            Gcomponent.remove_nodes_from([n for n in G.nodes if not isinstance(n, str) and n not in component])
            acquired = {target: False for target in Gcomponent.predecessors("END")}
            yield RuleGraph(Gcomponent, acquired, spaces=self.spaces+2)

    def _try_branches_single_component(self, ignore_special=False) -> set[frozenset[DexEntry]]:
        '''
        Assumes the graph is made up of only one component!
        '''
        if not ignore_special:
            special = self._handle_special_component()
            if special is not None:
                return special
        outcomes = set()
        rules = list(self._get_valid_rules())
        if not rules:
            return {frozenset()}
        for rule in rules:
            copy = self.copy()
            #print(f"{' '*copy.spaces}{rule}")
            copy._apply_rule(rule)
            copy.apply_safe_rules()
            if all(copy.acquired.values()):
                return {frozenset(copy.acquired.keys())}
            for os in product({frozenset(s for s, v in copy.acquired.items() if v)}, *copy.try_branches()):
                unioned = frozenset.union(*os)
                if frozenset(copy.acquired.keys()) == unioned:
                    return {unioned}
                outcomes.add(unioned)

        return RuleGraph._filter_outcomes(outcomes)

    @staticmethod
    def _filter_outcomes(outcomes):
        '''Filter out outcomes that are strictly worse than other paths'''
        to_remove = set()
        for outcome1 in outcomes:
            if outcome1 in to_remove:
                continue
            for outcome2 in outcomes:
                if outcome2 != outcome1 and outcome2.issubset(outcome1):
                    to_remove.add(outcome2)
                    break
        return outcomes.difference(to_remove)

    def _handle_special_component(self) -> set[frozenset[DexEntry]] | None:
        G = self.G
        # TODO: this is a possibly expensive check
        wf = [n for n in G.nodes if isinstance(n, ChoiceEntry) and n.choice.startswith("WF")]
        if len(wf) == 31:
            return self._handle_wf_component()
        else:
            # If transfer from WHITE to the main cartridge is possible, then you can obtain all of
            # them and this would have been handled by explore(). If it's not possible, then either
            # all trainers are of interest (single WHITE cartridge only), or fewer than 10 are
            # (Each Gen 4 game is missing at most 3 Pokemon families available in White Forest).
            assert len(wf) <= 11

        fs = [n for n in G.nodes if isinstance(n, ChoiceEntry) and n.choice == "FS"]
        if fs:
            return self._handle_fs_component(fs)
        return None
   
    def _handle_wf_component(self) -> set[frozenset[DexEntry]]:
        '''
        Should only be called in the case in which all WF trainers are useful
        '''
        G = self.G
        wf_node = [n for n in G.nodes if isinstance(n, ChoiceEntry) and n.choice == "WF"][0]
        white_id = wf_node.cart_id
        # Every White Forest Pokemon is present in at least one of these groups, and
        # missing from at least one of these groups.
        #
        # The first group is one of the optimal combinations.
        trainer_groups = [
            ['BRITNEY', 'CARLOS', 'DOUG', 'ELIZA', 'EMI', 'FREDERIC', 'JACQUES', 'LENA', 'LYNETTE', 'SILVIA'],
            ['DAVE', 'GENE', 'LEO', 'LYNETTE', 'MIHO', 'MIKI', 'PIERCE', 'PIPER', 'ROBBIE', 'SHANE'],
            ['COLLIN', 'GRACE', 'HERMAN', 'KARENNA', 'KEN', 'MARIE', 'MOLLY', 'ROSA', 'RYDER', 'VINCENT'],
        ]
        outcomes = set()
        for trainer_group in trainer_groups:
            copy = self.copy()
            for trainer in trainer_group:
                rule = Rule(frozenset({ChoiceEntry(white_id, "WF")}), frozenset(), frozenset({ChoiceEntry(white_id, f"WF_{trainer}")}), can_explore=False)
                copy._apply_rule(rule)
            copy.apply_safe_rules()
            for os in product({frozenset(s for s, v in copy.acquired.items() if v)}, *copy.try_branches()):
                outcomes.add(frozenset.union(*os))

        return RuleGraph._filter_outcomes(outcomes)

    def _handle_fs_component(self, fs_nodes: list[ChoiceEntry]):
        G = self.G
        console_possibilities = {}

        copy = self.copy()
        reset_gains = set()
        noreset_entries = set()
        for fs_node in fs_nodes:
            copy.G.nodes[fs_node]['count'] = 0
            if ("START", fs_node) in copy.G.edges:
                copy.G.remove_edge("START", fs_node)
            noreset = ChoiceEntry(fs_node.cart_id, "NORESET")
            if noreset in copy.G.nodes:
                assert(copy.G.nodes[noreset]['count'] == 1)
                noreset_entries.add(noreset)
        
        copy._update_maintaining_invariant(zero_entries=set(fs_nodes))
        if noreset_entries:
            reset_copy = copy.copy()
            for noreset in noreset_entries:
                copy.G.nodes[noreset]['count'] = 0
                copy.G.remove_edge("START", noreset)
            copy._update_maintaining_invariant(zero_entries=noreset_entries)
        other_branches = copy.try_branches()
        best_reset = frozenset()
        if noreset_entries:
            assert(not other_branches)
        for noreset in noreset_entries:
            console_id = noreset.cart_id
            reset = ChoiceEntry(console_id, "RESET")
            reset_rule = Rule(frozenset({noreset}), frozenset(), frozenset({reset}), can_explore=False)
            if reset_rule not in G.nodes:
                continue
            copy = reset_copy.copy() # type: ignore
            other_noresets = noreset_entries - {noreset}
            for other_noreset in other_noresets:
                copy.G.nodes[other_noreset]['count'] = 0
                copy.G.remove_edge("START", other_noreset)
            copy._update_maintaining_invariant(zero_entries=other_noresets)
            copy._apply_rule(reset_rule)
            reset_branches = copy.try_branches(ignore_special=True)
            assert(len(reset_branches) == 1)
            assert(len(reset_branches[0]) == 1)
            rb = reset_branches[0].pop()
            if best_reset.issubset(rb):
                best_reset = rb
            else:
                assert rb.issubset(best_reset)

        if best_reset:
            for dex_entry in best_reset:
                assert(dex_entry in G)
                self.acquired[dex_entry] = True
            self._update_maintaining_invariant(inf_entries=best_reset)
            
        for fs_node in fs_nodes:
            console_id = fs_node.cart_id
            downstreams = {}
            for pokemon_type in [
                "Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting", "Fire", "Flying", "Ghost",
                "Grass", "Ground", "Ice", "Normal", "Poison", "Psychic", "Rock", "Steel", "Water",
            ]:
                downstreams[pokemon_type] = {}
                for i in range(1, 4):
                    entry = ChoiceEntry(console_id, f"FS_{pokemon_type}_{i}")
                    if entry in G.nodes:
                        for rule in G.successors(entry):
                            useful, valid = self._all_downstream(rule)
                            if not valid:
                                raise ValueError("Not handled")
                            useful = useful - best_reset
                            if useful:
                                if i not in downstreams[pokemon_type]:
                                    downstreams[pokemon_type][i] = set()
                                downstreams[pokemon_type][i].add(frozenset(useful))
            console_possibilities[console_id] = set()
            for pokemon_type, idxs in downstreams.items():
                if not idxs:
                    continue
                for tup in product(*idxs.values()):
                    console_possibilities[console_id].add(frozenset.union(*tup))
        all_fs_entries = frozenset.union(*[frozenset.union(*ps) for ps in console_possibilities.values()])
        for ob in other_branches:
            all_in_branch = frozenset.union(*ob)
            assert(all_in_branch & all_fs_entries)
        to_cross_multiply = other_branches or [{best_reset}]
        for possibilities in console_possibilities.values():
            if possibilities:
                to_cross_multiply.append(possibilities)
        return {frozenset.union(*tup) for tup in product(*to_cross_multiply)}


    def _not_consumed(self, entry: Entry):
        '''
        Return true if no rule consumes (as opposed to requires) this entry
        '''
        G = self.G
        for _, rule, consumed in list(G.out_edges(entry, data='consumed')): # type: ignore
            if consumed:
                return False
        return True

    def _all_downstream(self, rule: "Rule") -> tuple[set[DexEntry], bool]:
        '''
        Return the set of useful nodes downstream of the rule, and whether they can all be acquired
        simply by applying the rule and then other safe rules
        '''
        G = self.G
        useful = {n for n in nx.descendants(G, rule) if (n, "END") in G.edges}
        copy = self.copy()
        for entry in copy.G.predecessors(rule):
            if copy.G.nodes[entry]['count'] != math.inf:
                copy.G.nodes[entry]['count'] = 1
                copy.G.add_edge("START", entry)
        copy._apply_rule(rule)
        copy.apply_safe_rules()
        return useful, all(copy.acquired[u] for u in useful)


_choice_idx = 0


def main(args: argparse.Namespace):
    pd.set_option('future.no_silent_downcasting', True)
    all_games: list[list[tuple[Game, str, Hardware | None]]] = [[]]
    all_hardware: list[list[Hardware]] = [[]]
    num_collections = 1
    for item in args.game_or_hardware:
        if item == '.':
            num_collections += 1
            all_games.append([])
            all_hardware.append([])
        elif item.split('.')[0] in GAMEINPUTS:
            game = Game.parse(item)
            all_games[-1].append((game, item, None))
        else:
            h, _, gs = item.partition('[')
            hardware = Hardware.parse(h)
            all_hardware[-1].append(hardware)
            if gs:
                for game_name in gs[:-1].split(','):
                    game = Game.parse(game_name, hardware)
                    all_games[-1].append((game, game_name, hardware))

    if num_collections == 1:
        games = all_games[0]
        hardware = set(all_hardware[0])
        _, result = calc_dexes(games, hardware, args.mode, args.flatten)

        if args.missing:
            result.print_missing()
        else:
            result.print_obtainable()
        print("\n---\n")
        print(f"TOTAL: {result.count()}")
    elif num_collections == 2:
        raise ValueError("Games/hardware before the first '.' are universal, so there should be 0 or 2+ instances of '.'")
    else:
        collection_saves: list[CollectionSaves] = []
        results = []
        for idx in range(1, num_collections):
            games = all_games[idx] + all_games[0]
            hardware = set(all_hardware[idx] + all_hardware[0])
            c, r = calc_dexes(games, hardware, args.mode, args.flatten)
            collection_saves.append(c)
            results.append(r)
        pokemon2idx: dict[str, int] = {}
        for cs in collection_saves:
            for pokemon in cs.pokemon_list.index:
                pokemon2idx[pokemon.species] = pokemon.idx

        ve_pairs: list[tuple[Pokemon, Pokemon]] = []
        if args.version_exclusive:
            games = {idx: {gs.game for gs in cs.game_saves.values()} for idx, cs in enumerate(collection_saves)}
            ve_table = pd.read_csv(Path("data") / "version_exclusives.csv")
            all_pokemon: set[Pokemon] = set.union(*(set(cs.pokemon_list.index) for cs in collection_saves)) # type: ignore
            sf2pokemon = {(p.species, p.form): p for p in all_pokemon}
            def parse_pokemon(pokemon_str: str) -> Pokemon | None:
                split = pokemon_str.split('_')
                species = split[0]
                form = split[1] if len(split) > 1 else None
                return sf2pokemon.get((species, form))
            for _, row in ve_table.iterrows():
                matches_first = {idx for idx, gameset in games.items() if any(g.match(row.GAME1) for g in gameset)}
                matches_second = {idx for idx, gameset in games.items() if any(g.match(row.GAME2) for g in gameset)}
                if len(matches_first) == 0 or len(matches_second) == 0:
                    continue
                if len(matches_first) == 1 and matches_first == matches_second:
                    continue
                p1 = parse_pokemon(row.SPECIES1)
                p2 = parse_pokemon(row.SPECIES2)
                if p1 is not None and p2 is not None:
                    ve_pairs.append((p1, p2))

        result = MultiResult(results, args.mode == "form", ve_pairs)
        if args.all_present:
            result.print_all_present()
        elif args.compact:
            result.print_compact(not args.missing, not args.full)
        else:
            result.print(not args.missing, not args.full)


def calc_dexes(games: list[tuple[Game, str, Hardware | None]], hardware: set[Hardware], mode: str, flatten=False) -> tuple[CollectionSaves, Result]:
    cartridges = [Cartridge(g, cl, c) for g, cl, c in games]

    collection = Collection(cartridges, hardware)
    new_collection_saves = CollectionSaves(collection, mode)
    new_dexes = new_collection_saves.calc_dexes(flatten)

    return new_collection_saves, new_dexes
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_or_hardware', nargs='+', default=[])
    parser.add_argument('--full', '-f', action='store_true')
    parser.add_argument('--all-present', '-a', action='store_true')
    parser.add_argument('--flatten', action='store_true')
    parser.add_argument('--compact', '-c', action='store_true')
    parser.add_argument('--version-exclusive', '-v', action='store_true')
    parser.add_argument('--missing', '-m', action='store_true')
    parser.add_argument('--mode', choices=["species", "form"], default="species")
    args = parser.parse_args()

    if args.full and args.all_present:
        raise ValueError("--full and --all-present are incompatible")
    if args.all_present and args.compact:
        raise ValueError("--all-present and --compact are incompatible")
    if args.all_present and args.version_exclusive:
        raise ValueError("--all-present and --version-exclusive are incompatible")
    if args.all_present and args.missing:
        raise ValueError("--all-present and --missing are incompatible")
    if args.compact and args.version_exclusive:
        raise ValueError("--compact and --version_exclusive are incompatible")
    main(args)
