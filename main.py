#!/usr/bin/python

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from itertools import chain, combinations, product, zip_longest
import math
from pathlib import Path
from typing import Optional

import networkx as nx
#from matplotlib import pyplot as plt
import pandas as pd

from game import Game, Cartridge, Hardware, Collection, GAMES, GAMEIDS, GAMEINPUTS
from result import MultiResult, Result

class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DexEntry():
    species: str
    
    def __repr__(self):
        return f"DexEntry({self.species})"

@dataclass(frozen=True)
class Item():
    item: str

@dataclass(frozen=True)
class GameChoice():
    choice: str

    def __repr__(self):
        return f"GameChoice({self.choice})"

@dataclass(frozen=True)
class PokemonReq():
    species: str
    gender: Optional[Gender] = None
    required: frozenset = frozenset()
    forbidden: frozenset = frozenset([])

    def matches(self, pokemon):
        if self.species != pokemon.species:
            return False
        if self.gender is not None and pokemon.gender != self.gender:
            return False
        if not self.required.issubset(pokemon.props):
            return False
        if self.forbidden.intersection(pokemon.props):
            return False
        return True
 
    def __str__(self):
        out = self.species
        if self.gender == Gender.MALE:
            out += '(♂)'
        elif self.gender == Gender.FEMALE:
            out += '(♀)'

        if self.required:
            out += ' {'
            out += ', '.join(self.required)
            out += '}'
        return out


@dataclass(frozen=True)
class Pokemon():
    species: str
    gender: Optional[Gender]
    props: frozenset = frozenset([])

    def __str__(self):
        out = self.species
        if self.gender == Gender.MALE:
            out += '(♂)'
        elif self.gender == Gender.FEMALE:
            out += '(♀)'

        if self.props:
            out += ' {'
            out += ', '.join(self.props)
            out += '}'
        return out

    __repr__ = __str__


class Generation():
    DATA = ["breed", "buy", "environment", "evolve", "fossil", "friend_safari", "gift", "misc",
            "pickup_item", "pickup_pokemon", "trade", "wild", "wild_item"]

    def __init__(self, gen):
        self.id = gen
        self.dir = Path("data") / f"gen{self.id}"
        self.games = set(GAMES.loc[GAMES.GENERATION == self.id, "GAMEID"])

        dex = pd.read_csv(self.dir / "dex.csv").fillna(False)
        self.pokemon_list = dex.set_index(dex.index + 1).reset_index().set_index('SPECIES')
        self.pokemon_list.loc[self.pokemon_list.GENDER == False, 'GENDER'] = 'BOTH'

        self.items = pd.read_csv(self.dir / "item.csv").fillna(False).set_index("ITEM")

        self.tradeable_pokemon = set(self.pokemon_list.index)
        self.tradeable_items = set([item for item in self.items.index if not self.items.loc[item]["KEY"]])
 
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


    def genders(self, pokemon):
        '''Return the list of possible gender suffixes.'''
        gender = self.pokemon_list.loc[pokemon, "GENDER"]
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
    def __init__(self, cartridge, generation):
        self.cartridge = cartridge
        self.game = cartridge.game
        self.generation = generation

        self.has_gender = self.game.gen >= 2 and "NO_BREED" not in self.game.props

        # Initialized later - indicates what other game states this one can communicate with
        self.transfers = {
            "TRADE": set(), # Trade Pokemon (possible w/held items)
            "POKEMON": set(), # Transfer Pokemon w/o trading
            "ITEMS": set(), # Transfer items
            "MYSTERYGIFT": set(), # Mystery gift
            "RECORDMIX": set(), # Gen 3 record mixing
            "CONNECT": set(), # Miscellaneous
        }

        self.env = set()
        if self.generation.environment is not None:
            for _, row in self.generation.environment.iterrows():
                if self.game.match(row.GAME):
                    self.env.add(row.ENVIRONMENT)

        # These are used to initialize what will be the final rules.
        self.evolutions = {species: {} for species in self.generation.pokemon_list.index}
        self.breeding = {species: {} for species in self.generation.pokemon_list.index}
        self.unique_pokemon = {species: set() for species in self.generation.pokemon_list.index}


    def transfer(self, other_cartridge, category):
        '''Register another cartridge as able to receive certain communications'''
        self.transfers[category].add(other_cartridge)
    
    def init_evolutions(self):
        if self.generation.evolve is None or "NO_EVOLVE" in self.game.props or "NO_DEX" in self.game.props:
            return

        for _, row in self.generation.evolve.iterrows():
            env = row.get("ENVIRONMENT")
            if env and env not in self.env:
                continue
            pre = self.parse_pokemon_input(row.FROM, forbidden={"NOEVOLVE"})
            post = self.parse_output(row.TO, use_gender=False)[0]
            if pre not in self.evolutions[pre.species]:
                self.evolutions[pre.species][pre] = set()
            for p in post:
                self.evolutions[pre.species][pre].add(p)

    def init_breeding(self):
        if self.generation.breed is None or "NO_BREED" in self.game.props or "NO_DEX" in self.game.props:
            return

        for _, row in self.generation.breed.iterrows():
            parent = self.parse_pokemon_input(row.PARENT, forbidden={"NOBREED"})
            if parent not in self.breeding[parent.species]:
                self.breeding[parent.species][parent] = set()
            for child in self.parse_output(row.CHILD, self.has_gender):
                self.breeding[parent.species][parent].add(child[0])

    def init_unique_pokemon(self):
        if self.generation.buy is not None:
            for _, row in self.generation.buy.iterrows():
                if not self.game.match(row.GAME):
                    continue
                for os in self.parse_output(row.POKEMON_OR_ITEM, self.has_gender):
                    for o in os:
                        if isinstance(o, Pokemon):
                            self.add_unique(o)

        if self.generation.fossil is not None:
            for _, row in self.generation.fossil.iterrows():
                if not self.game.match(row.GAME):
                    continue
                for os in self.parse_output(row.POKEMON, self.has_gender):
                    for o in os:
                        self.add_unique(o)

        if self.generation.gift is not None:
            for _, row in self.generation.gift.iterrows():
                if not self.game.match(row.GAME):
                    continue
                for choice in self.parse_gift_entry(row.POKEMON_OR_ITEM):
                    for o in choice:
                        if isinstance(o, Pokemon):
                            self.add_unique(o)

        if self.generation.trade is not None:
            for _, row in self.generation.trade.iterrows():
                for os in self.parse_output(row.GET, self.has_gender):
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
                        for os in self.parse_output(row.SPECIES, self.has_gender):
                            for o in os:
                                if isinstance(o, Pokemon):
                                    self.add_unique(o)
                        
    def add_unique(self, pokemon):
        if pokemon in self.unique_pokemon[pokemon.species]:
            return

        self.unique_pokemon[pokemon.species].add(pokemon)

        for preq, posts in self.evolutions[pokemon.species].items():
            if preq.matches(pokemon):
                for post in posts:
                    if self.has_gender:
                        post_genders = self.generation.genders(post.species)
                        if pokemon.gender in post_genders:
                            post_gender = pokemon.gender
                        elif len(post_genders) > 1:
                            raise ValueError(f"Gender mismatch between '{preq}' and '{post}' evolution")
                        else: # Shedinja
                            post_gender = post_genders[0]
                    else:
                        post_gender = None
                    new_p = Pokemon(post.species, post_gender, pokemon.props.union(post.props))
                    self.add_unique(new_p)

        for preq, ps in self.breeding[pokemon.species].items():
            if preq.matches(pokemon):
                for p in ps:
                    self.add_unique(p)

        if "NOTRANSFER" not in pokemon.props:
            transfer_to = self.transfers["TRADE"].union(self.transfers["POKEMON"])
            if "TRANSFERRESET" in pokemon.props:
                pokemon = Pokemon(pokemon.species, pokemon.gender, frozenset())
            for gs in transfer_to:
                if pokemon.species in gs.unique_pokemon:
                    gs.add_unique(pokemon)

    def get_rules(self):
        global _choice_idx
        rules = []

        if self.generation.breed is not None and "NO_BREED" not in self.game.props and "NO_DEX" not in self.game.props:
            ditto = self.parse_pokemon_input('Ditto', forbidden={"NOBREED"})
            for _, row in self.generation.breed.iterrows():
                pr = self.parse_pokemon_input(row.PARENT, forbidden={"NOBREED"})
                for gender in self.generation.genders(pr.species): 
                    pg = replace(pr, gender=gender)
                    for c in self.parse_output(row.CHILD):
                        if len(c) > 1:
                            raise ValueError(f"Invalid child entry {c}")
                        c = c[0]
                        required = {pg}
                        if gender != Gender.FEMALE:
                            required.add(ditto)
                        if "ITEM" in row and row.ITEM:
                            required.add(Item(row.ITEM))
                        rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, {}, required, {c})

        if self.generation.buy is not None:
            for _, row in self.generation.buy.iterrows():
                if not self.game.match(row.GAME):
                    continue
                exchange, _ = self.parse_input(row.get("EXCHANGE"))
                required, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                for o in self.parse_output(row.POKEMON_OR_ITEM, self.has_gender):
                    if len(o) != 1:
                        raise ValueError(f"Invalid buy entry {o}")
                    consumed = set(exchange) if exchange else set()
                    required = set(required) if required else set()
                    rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, consumed, required, {o[0]})

        trade_evos = {}
        trade_evos_by_item = {}
        trade_evo_pairs = {}

        if self.generation.evolve is not None and "NO_EVOLVE" not in self.game.props and "NO_DEX" not in self.game.props:
            for _, row in self.generation.evolve.iterrows():
                env = row.get("ENVIRONMENT")
                if env and env not in self.env:
                    continue
                other = self.parse_pokemon_input(row.OTHER_POKEMON) if row.get("OTHER_POKEMON") else None
                if row.get("TRADE"):
                    item = row.get("ITEM", False)
                    if other:
                        if item:
                            raise ValueError("Not handled")
                        trade_evo_pairs[row.FROM] = (row.TO, other.species)
                        continue
                    if row.FROM not in trade_evos:
                        trade_evos[row.FROM] = {}
                    trade_evos[row.FROM][item] = row.TO
                    if item not in trade_evos_by_item:
                        trade_evos_by_item[item] = {}
                    trade_evos_by_item[item][row.FROM] = row.TO
                    continue
                
                pre = self.parse_pokemon_input(row.FROM, forbidden={"NOEVOLVE"})
                post = self.parse_output(row.TO, use_gender=False)
                item = Item(row.ITEM) if row.ITEM else None
                if len(post) != 1:
                    raise ValueError(f"Invalid evolution entry {post}")
                rules += Rule.evolution_init(self.cartridge.id, self.unique_pokemon, self.generation.genders, pre, post[0], item, other)

        if self.generation.fossil is not None:
            for _, row in self.generation.fossil.iterrows():
                if not self.game.match(row.GAME):
                    continue
                reqs, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                for p in self.parse_output(row.POKEMON, self.has_gender):
                    if len(p) != 1:
                        raise ValueError(f"Unexpected fossil {p}")
                    rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, {Item(row.ITEM)}, reqs, p)

        if self.generation.gift is not None:
            for idx, row in self.generation.gift.iterrows():
                if not self.game.match(row.GAME):
                    continue
                reqs, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                choices = self.parse_gift_entry(row.POKEMON_OR_ITEM)
                if len(choices) == 1:
                    rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, set(), reqs, set(choices[0]), 1)
                else:
                    _choice_idx += 1
                    choice = GameChoice(f"gift:{_choice_idx}")
                    rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, set(), reqs, {choice}, 1)
                    for pokemon_or_items in choices:
                        rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, {choice}, set(), pokemon_or_items, 1)

        if self.generation.misc is not None:
            for _, row in self.generation.misc.iterrows():
                if not self.game.match(row.GAME):
                    continue
                consumed, _ = self.parse_input(row.get("CONSUMED"))
                required, _ = self.parse_input(row.get("REQUIRED"))
                output = self.parse_output(row.get("OUTPUT"))
                repeats = int(row["REPEATS"])
                rules.append(Rule(
                    self.cartridge.id, self.cartridge.id,
                    frozenset(consumed) if consumed else frozenset(),
                    frozenset(required) if required else frozenset(),
                    frozenset(output[0]), repeats))

        if self.generation.pickup_item is not None and "NO_DEX" not in self.game.props:
            for _, row in self.generation.pickup_item.iterrows():
                if not self.game.match(row.GAME):
                    continue
                for pokemon in self.generation.pickup_pokemon["SPECIES"]:
                    req = self.parse_pokemon_input(pokemon)
                    rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, set(), {req}, {Item(row.ITEM)})
        
        if self.generation.trade is not None:
            for idx, row in self.generation.trade.iterrows():
                if not self.game.match(row.GAME):
                    continue
                reqs, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                if row.GIVE == "ANY":
                    gives = [None]
                    give_species = None
                else:
                    give = self.parse_pokemon_input(row.GIVE, forbidden={"NOTRANSFER"}) 
                    give_species = give.species
                    gives = [p for p in self.unique_pokemon[give_species] if give.matches(p)]
                gets = self.parse_output(row.GET)
                get_species = gets[0][0].species
                item = row.get("ITEM")
                choice = None
                if len(gets) * len(gives) > 1:
                    _choice_idx += 1
                    choice = GameChoice(f"trade_{row.GET}:{_choice_idx}")
                    rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, set(), reqs, {choice}, 1)
                    reqs = set()

                if get_species in trade_evo_pairs:
                    _, other = trade_evo_pairs[get_species]
                    if (give_species in [None, other]) and item != "Everstone":
                        raise ValueError("Not handled")

                evolution = None
                if get_species in trade_evos:
                    if item and item in trade_evos[get_species]:
                        evolution = trade_evos[get_species][item]
                        item = False
                    elif False in trade_evos[get_species]:
                        if self.game.gen == 3 and self.game.core:
                            # Held item loss glitch - item is lost/ignored, including Everstone
                            item = False
                        # Everstone doesn't stop Kadabra from evolving since gen 4
                        if item != "Everstone" or (self.game.gen >= 4 and get_species == "Kadabra"):
                            evolution = trade_evos[get_species][False]

                for give in gives:
                    for get in gets:
                        if len(get) != 1:
                            raise ValueError(f"Invalid trade entry {get}")
                        get = get[0]

                        consumed = {give} if give else set()
                        if choice:
                            consumed.add(choice)
                        output = {Pokemon(evolution, get.gender, get.props), DexEntry(get.species)} if evolution else {get}
                        if item:
                            output.add(Item(item))
                        
                        rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, consumed, reqs, output, repeats=1)

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
                    p_or_is = self.parse_output(pokemon_or_item, self.has_gender)
                    for idx, (consumed, reqs, count) in enumerate(self.parse_wild_entry(row[gamecol])):
                        if count == 0:
                            continue
                        if count != math.inf and len(items) * len(p_or_is) > 1 and not consumed:
                            _choice_idx += 1
                            choice = GameChoice(f"wild_{pokemon_or_item}:{_choice_idx}")
                            rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, consumed, reqs, {choice}, count)
                            consumed = {choice}
                            reqs = set()
                        for p_or_i in p_or_is:
                            if len(p_or_i) != 1:
                                raise ValueError("Confused")
                            for item in items:
                                out = {p_or_i[0]}
                                if item:
                                    out.add(Item(item))
                                rules += Rule.multi_init(self.cartridge.id, self.unique_pokemon, consumed, reqs, out, count)

        for species in self.unique_pokemon:
            for pokemon in self.unique_pokemon[species]:
                if "NOTRANSFER" in pokemon.props:
                    continue
                out_pokemon = pokemon
                if "TRANSFERRESET" in pokemon.props:
                    out_pokemon = Pokemon(pokemon.species, pokemon.gender)
                for gs in self.transfers["POKEMON"]:
                    if species not in gs.generation.tradeable_pokemon:
                        continue
                    rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {pokemon}, {out_pokemon}))

                has_noitem_evo = False
                for gs in self.transfers["TRADE"]:
                    if species not in gs.generation.tradeable_pokemon:
                        continue
                    for item, evo_species in trade_evos.get(species, {}).items():
                        if not item and "NOEVOLVE" not in out_pokemon.props:
                            has_noitem_evo = True
                        if evo_species not in gs.generation.tradeable_pokemon:
                            continue
                        if "NOEVOLVE" in out_pokemon.props:
                            continue
                        evo_pokemon = replace(out_pokemon, species=evo_species)
                        if item:
                            rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {pokemon, Item(item)}, {evo_pokemon}))
                        else:
                            rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {pokemon}, {evo_pokemon}))
                            if (species != "Kadabra" or self.generation.id == 2) and self.generation.id != 3:
                                rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {pokemon, Item("Everstone")}, {out_pokemon}))
                    if species in trade_evo_pairs and "NOEVOLVE" not in out_pokemon.props:
                        evo_species, species2 = trade_evo_pairs[species]
                        evo_species2 = trade_evo_pairs[species2][0]
                        evo_pokemon = replace(out_pokemon, species=evo_species)
                        for pokemon2 in self.unique_pokemon[species2]:
                            out_pokemon2 = pokemon2
                            if "TRANSFERRESET" in out_pokemon2.props:
                                out_pokemon2 = Pokemon(pokemon2.species, pokemon2.gender)
                            if "NOEVOLVE" in out_pokemon2.props:
                                continue
                            evo_pokemon2 = replace(out_pokemon2, species=evo_species2)
                            rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {pokemon, pokemon2}, {evo_pokemon, evo_pokemon2}))
                    if not has_noitem_evo:
                        rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {pokemon}, {out_pokemon}))
                

        for item in self.generation.tradeable_items:
            for gs in self.transfers["ITEMS"]:
                if item not in gs.generation.tradeable_items:
                    continue
                rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {Item(item)}, {Item(item)}))

            if "NO_TRADE_ITEMS" not in self.game.props:
                for gs in self.transfers["TRADE"]:
                    if "NO_TRADE_ITEMS" in gs.game.props:
                        continue
                    if item not in gs.generation.tradeable_items:
                        continue
                    rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {Item(item)}, {Item(item)}))

            for gs in self.transfers["MYSTERYGIFT"]:
                rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {GameChoice(f"MG_{item}")}, {Item(item)}))

        for gs in self.transfers["RECORDMIX"]:
            rules.append(Rule.transfer(self.cartridge.id, gs.cartridge.id, {GameChoice("RM_Eon Ticket")}, {Item("Eon Ticket")}))
        
        return rules
    
    def parse_pokemon_input(self, entry, forbidden=None):
        split = entry.split('_')
        species = split[0]
        gender = None
        if species not in self.generation.pokemon_list.index:
            raise ValueError(f"Invalid Pokemon {species}")
        props = set(split[1:])
        if "MALE" in props:
            props.remove("MALE")
            gender = Gender.MALE
        elif "FEMALE" in props:
            props.remove("FEMALE")
            gender = Gender.FEMALE

        if "ONE" in props:
            props.remove("ONE")
            props.add(f"ONE_{self.cartridge.id}")
        return PokemonReq(species, gender, frozenset(props), frozenset(forbidden or []))

    def parse_input(self, entry, forbidden=None):
        out = []
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
            if e in self.generation.items.index:
                out.append(Item(e))
            elif e.split('.')[0] in GAMEIDS:
                if not any([gs.cartridge != self.cartridge and gs.game.match(e) for gs in self.transfers["CONNECT"].union(self.transfers["RECORDMIX"])]): 
                    valid = False
            elif e.startswith("DEX_"):
                out += [DexEntry(p) for p in self.generation.pokemon_list.index if self.generation.pokemon_list.loc[p, e]]
            elif e.startswith("$"):
                out.append(GameChoice(e[1:]))
            else:
                raise ValueError(f"Unrecognized entry {e}")
        return out, valid

    def parse_output(self, entry, use_gender=True):
        out = []
        for e in entry.split(','):
            split = e.split('_')
            species = split[0]
            if species in self.generation.pokemon_list.index:
                props = set(split[1:])
                if "MALE" in props:
                    props.remove("MALE")
                    genders = [Gender.MALE]
                elif "FEMALE" in props:
                    props.remove("FEMALE")
                    genders = [Gender.FEMALE]
                elif self.has_gender and use_gender:
                    genders = self.generation.genders(species)

                if not (self.has_gender and use_gender):
                    genders = [None]
                props = frozenset(props)

                out.append([Pokemon(species, g, props) for g in genders])
            elif e in self.generation.items.index:
                out.append([Item(e)])
            elif e.startswith("$"):
                out.append([GameChoice(e[1:])])
            else:
                raise ValueError(f"Unrecognized entry {e}")
        return list(product(*out))
    
    def parse_gift_entry(self, entry, use_gender=True):
        out = []
        for e in entry.split(';'):
            out += self.parse_output(e, use_gender)
        return out

    def parse_wild_entry(self, entry, use_gender=True):
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
    def __init__(self, collection, generations=None):
        if generations is None:
            generations = {gen: Generation(gen) for gen in set(c.game.gen for c in collection.cartridges)}
        self.generations = generations

        self.game_saves = {c: GameSave(c, generations[c.game.gen]) for c in collection.cartridges}
        self.collection = collection
        self.main_cartridge = collection.main_cartridge

        for kind, cart_pairs in collection.interactions.items():
            for cart1, cart2 in cart_pairs:
                self.game_saves[cart1].transfer(self.game_saves[cart2], kind)

        if any([gs.has_gender for gs in self.game_saves.values()]):
            for gs in self.game_saves.values():
                gs.has_gender = True
        self.pokemon_list = self.game_saves[self.main_cartridge].generation.pokemon_list

        for gs in self.game_saves.values():
            gs.init_evolutions()
            gs.init_breeding()
        for gs in self.game_saves.values():
            gs.init_unique_pokemon()

    def calc_dexes(self, flatten=False):
        idx2pokemon = {}
        for pokemon, row in self.pokemon_list.iterrows():
            idx = row['index']
            idx2pokemon[idx] = pokemon
        rules = []
        for gs in self.game_saves.values():
            rules += gs.get_rules()
        rules += self._friend_safari_rules()
        self.rule_graph = RuleGraph(self.main_cartridge.id, {p: False for p in self.pokemon_list.index}, rules)

        self.rule_graph.explore()
        branches = self.rule_graph.try_branches_and_update()
        present = {k for k, v in self.rule_graph.dex.items() if v}
        if flatten:
            for branch in branches:
                present = present.union(*branch)
            branches = []
        return Result.new(idx2pokemon, present, branches)

    def _friend_safari_rules(self):
        rules = []
        fs_consoles = self.collection.friend_safari_consoles()
        if not fs_consoles or not self.generations.get(6):
            return rules

        fs_pokemon = self.generations[6].friend_safari

        for console, carts in fs_consoles.items():
            cid = f"3DS_{console.id}"
            rules.append(Rule(cid, cid, frozenset(), frozenset(), frozenset({GameChoice("FS")}), repeats=1))
            for pokemon_type in fs_pokemon['TYPE'].unique():
                rules.append(Rule(cid, cid, frozenset({GameChoice("FS")}), frozenset(),
                    frozenset({GameChoice(f"FS_{pokemon_type}_{slot}") for slot in range(1, 4)})))
            for _, row in fs_pokemon.iterrows():
                slot = int(row.SLOT)
                gc_in = GameChoice(f"FS_{row.TYPE}_{slot}")
                gc_out = GameChoice(f"FS_{row.TYPE}_{slot}_{row.SPECIES}")
                cart_out = GameChoice(f"FS_{row.SPECIES}")
                rules.append(Rule(cid, cid, frozenset({gc_in}), frozenset(), frozenset({gc_out})))
                for cart, max_cart_slot in carts.items():
                    if max_cart_slot < slot:
                        continue
                    rules.append(Rule(cid, cart.id, frozenset(), frozenset({gc_out}), frozenset({cart_out})))
        return rules


class RuleGraph():
    def __init__(self, main_cart_id, dex, rules_or_digraph, parent=None, spaces=0):
        self.main_cart_id = main_cart_id
        self.dex = dex
        self.spaces = spaces
        if isinstance(rules_or_digraph, nx.DiGraph):
            self.G = rules_or_digraph
        else:
            self.G = nx.DiGraph()
            self.G.add_edge("START", "INF")
            self.G.add_node("END")
            for rule in rules_or_digraph:
                self._add_rule(rule, init=True)
            self._prune()

        self.parent = parent

    def copy(self, set_parent=True):
        return RuleGraph(self.main_cart_id, self.dex.copy(), self.G.copy(), parent=(self if set_parent else None), spaces=self.spaces+2)

    def _add_rule(self, rule, init=False):
        if isinstance(rule.in_cart_id, int):
            import pdb; pdb.set_trace()
        G = self.G
        for c in rule.consumed:
            entry = (rule.in_cart_id, c)
            G.add_edge(entry, rule, consumed=True)
            if 'count' not in self.G.nodes[entry]:
                nx.set_node_attributes(G, {entry: {'count': 0}})
        for r in rule.required:
            entry = (rule.in_cart_id, r)
            G.add_edge(entry, rule, consumed=False)
            if not isinstance(r, DexEntry) and 'count' not in G.nodes[entry]:
                nx.set_node_attributes(G, {entry: {'count': 0}})
        if not rule.consumed and not rule.required:
            G.add_edge("INF", rule)
        for o in rule.output:
            entry = (rule.out_cart_id, o)
            G.add_edge(rule, entry)
            if not isinstance(o, DexEntry) and 'count' not in G.nodes[entry]:
                nx.set_node_attributes(G, {entry: {'count': 0}})
            elif isinstance(o, DexEntry) and rule.out_cart_id == self.main_cart_id:
                self.G.add_edge(entry, "END")

        if init:
            # We add an additional cart equivalent to the main cart but with a custom cart ID
            # This makes it simpler to handle resetting the main cartridge.
            if rule.in_cart_id == self.main_cart_id and rule.out_cart_id == self.main_cart_id:
                self._add_rule(replace(rule, in_cart_id='main_reset', out_cart_id='main_reset'))
            if rule.in_cart_id == self.main_cart_id and rule.is_transfer:
                self._add_rule(replace(rule, in_cart_id='main_reset'))
            if rule.out_cart_id == self.main_cart_id and rule.is_transfer:
                self._add_rule(replace(rule, out_cart_id='main_reset'))

    def explore(self):
        '''
        Make a child and have the child explore among valid rules aside from those in the main cart
        '''
        while True:
            any_transfer_rule = False
            self.apply_safe_rules()
            child = self.copy()
            while True:
                rules = list(
                    r for r in child._get_valid_rules()
                    if r.in_cart_id != self.main_cart_id and not r.in_cart_id.startswith("3DS")
                )
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

    def try_branches_and_update(self):
        '''
        Try out all branching paths. For any branching path with only one outcome, simply add the
        outcomes to dex.
        '''
        out_branches = []
        branches = self.try_branches()
        for branch in branches:
            if len(branch) == 1:
                for species in branch.pop():
                    self.dex[species] = True
            elif len(branch) > 12:
                out_branches.append(self._curtail_branch_output(branch))
            else:
                out_branches.append(branch)
        return out_branches

    def try_branches(self):
        outcomes = []
        for rg in self._get_components():
            outcomes.append(rg._try_branches_single_component())
        return outcomes

    def _apply_rule(self, rule):
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

        repeats = 1 if has_consumed else rule.repeats
        new_rule = replace(rule, repeats=(rule.repeats - repeats))
        G.remove_node(rule)
        self._add_rule(new_rule)
        if new_rule.repeats == 0:
            zero_repeat_rules.add(new_rule)

        # Acquire outputs
        # If it's a transfer rule, the parent (if it exists) also gets the outputs
        changes_by_rg = {self: [set(), zero_entries, zero_repeat_rules]}
        if rule.is_transfer and self.parent is not None:
            changes_by_rg[self.parent] = [set(), None, None]

        for rg, changes in changes_by_rg.items():
            for out in outputs:
                if out not in rg.G.nodes:
                    continue
                if isinstance(out[1], DexEntry):
                    rg.dex[out[1].species] = True
                    changes[0].add(out)
                else:
                    if rg.G.nodes[out]['count'] == 0:
                        rg.G.add_edge("START", out)
                    rg.G.nodes[out]['count'] += repeats
                    if repeats == math.inf or self._not_consumed(out):
                        changes[0].add(out)
            rg._update_maintaining_invariant(*changes)

    def _update_maintaining_invariant(self, inf_entries=None, zero_entries=None, zero_repeat_rules=None):
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
                rules_with_removed_inputs[rule].add(entry[1])
            for rule in G.predecessors(entry):
                if not isinstance(rule, Rule):
                    continue
                if rule not in rules_with_removed_outputs:
                    rules_with_removed_outputs[rule] = set()
                rules_with_removed_outputs[rule].add(entry[1])
            G.remove_node(entry)

        for entry in zero_entries or set():
            maybe_unreachable_entries.add(entry)

        for rule in zero_repeat_rules or set():
            for entry in G.predecessors(rule):
                if isinstance(entry, str):
                    continue
                maybe_useless_entries.add(entry)
            if rule in rules_with_removed_outputs:
                del rules_with_removed_outputs[rule]
            if rule in rules_with_removed_inputs:
                del rules_with_removed_inputs[rule]
            G.remove_node(rule)

        while maybe_unreachable_entries or maybe_useless_entries or rules_with_removed_outputs or rules_with_removed_inputs:
            if maybe_unreachable_entries:
                entry = maybe_unreachable_entries.pop()
                if len(list(G.predecessors(entry))) == 0:
                    for rule in list(G.successors(entry)):
                        if not isinstance(rule, Rule):
                            continue
                        for entry2 in G.successors(rule):
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
                if not nx.has_path(G, entry, "END"):
                    for rule in G.predecessors(entry):
                        if not isinstance(rule, Rule):
                            continue
                        if rule not in rules_with_removed_outputs:
                            rules_with_removed_outputs[rule] = set()
                        rules_with_removed_outputs[rule].add(entry[1])
                    G.remove_node(entry)
            elif rules_with_removed_outputs:
                rule = next(iter(rules_with_removed_outputs))
                removed_outputs = rules_with_removed_outputs.pop(rule)
                removed_inputs = rules_with_removed_inputs.pop(rule, set())
                if rule not in G.nodes:
                    import pdb; pdb.set_trace()
                if "END" not in G.nodes:
                    import pdb; pdb.set_trace()
                if not nx.has_path(G, rule, "END"):
                    for entry in G.predecessors(rule):
                        if isinstance(entry, str):
                            continue
                        maybe_useless_entries.add(entry)
                    G.remove_node(rule)
                else:
                    new_rule = replace(
                        rule,
                        consumed=(rule.consumed - removed_inputs),
                        required=(rule.required - removed_inputs),
                        output=(rule.output - removed_outputs))
                    G.remove_node(rule)
                    self._add_rule(new_rule)
            elif rules_with_removed_inputs:
                rule = next(iter(rules_with_removed_inputs))
                removed_inputs = rules_with_removed_inputs.pop(rule)
                new_rule = replace(
                    rule,
                    consumed=(rule.consumed - removed_inputs),
                    required=(rule.required - removed_inputs))
                G.remove_node(rule)
                self._add_rule(new_rule)

    def _prune(self):
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
                new_rule = replace(rule, output=(rule.output - {entry[1]}))
                G.remove_node(rule)
                self._add_rule(new_rule)
            G.remove_node(entry)

    def _get_valid_rules(self):
        '''
        Return iterator of valid rules, i.e. rules whose inputs are satisfied
        '''
        G = self.G
        return (r for r in nx.single_source_shortest_path(G, "START", cutoff=2).keys()
                if isinstance(r, Rule) and all(G.has_edge("START", p) for p in G.predecessors(r)))

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
            Gcomponent = G.copy()
            Gcomponent.remove_nodes_from([n for n in G.nodes if not isinstance(n, str) and n not in component])
            dex = {entry[1].species: False for entry in Gcomponent.predecessors("END")}
            yield RuleGraph(self.main_cart_id, dex, Gcomponent, spaces=self.spaces+2)

    def _try_branches_single_component(self):
        '''
        Assumes the graph is made up of only one component!
        '''
        special = self._handle_special_component()
        if special is not None:
            return special
        outcomes = set()
        G = self.G
        rules = list(self._get_valid_rules())
        if not rules:
            return {frozenset()}
        for rule in rules:
            copy = self.copy(set_parent=False)
            #if self.spaces <= 12:
            #    print(f"{' '*copy.spaces}{rule}")
            
            copy._apply_rule(rule)
            copy.apply_safe_rules()
            if all(copy.dex.values()):
                return {frozenset(copy.dex.keys())}
            for os in product({frozenset(s for s, v in copy.dex.items() if v)}, *copy.try_branches()):
                outcomes.add(frozenset.union(*os))

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

    def _handle_special_component(self):
        G = self.G
        # TODO: this is a possibly expensive check
        wf = [n for n in G.nodes if isinstance(n, tuple) and isinstance(n[1], GameChoice) and n[1].choice.startswith("WF")]
        if len(wf) == 31:
            return self._handle_wf_component()
        else:
            # If transfer from WHITE to the main cartridge is possible, then you can obtain all of
            # them and this would have been handled by explore(). If it's not possible, then either
            # all trainers are of interest (single WHITE cartridge only), or fewer than 10 are
            # (Each Gen 4 game is missing at most 3 Pokemon families available in White Forest).
            assert len(wf) <= 11

        fs = [n for n in G.nodes if isinstance(n, tuple) and isinstance(n[1], GameChoice) and n[1].choice == "FS"]
        if fs:
            return self._handle_fs_component(fs)
        return None
   
    def _handle_wf_component(self):
        '''
        Should only be called in the case in which all WF trainers are useful
        '''
        G = self.G
        wf_node = [n for n in G.nodes if isinstance(n, tuple) and n[1] == GameChoice("WF")][0]
        white_id = wf_node[0]
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
            copy = self.copy(set_parent=False)
            for trainer in trainer_group:
                rule = Rule(white_id, white_id, frozenset({GameChoice("WF")}), frozenset(), frozenset({GameChoice(f"WF_{trainer}")}), 1)
                copy._apply_rule(rule)
            copy.apply_safe_rules()
            for os in product({frozenset(s for s, v in copy.dex.items() if v)}, *copy.try_branches()):
                outcomes.add(frozenset.union(*os))

        return RuleGraph._filter_outcomes(outcomes)

    def _handle_fs_component(self, fs_nodes):
        G = self.G
        console_possibilities = {}
        for fs_node in fs_nodes:
            G.nodes[fs_node]['count'] = 0
            if ("START", fs_node) in G.edges:
                G.remove_edge("START", fs_node)
            console_id = fs_node[0]
            downstreams = {}
            for pokemon_type in [
                "Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting", "Fire", "Flying", "Ghost",
                "Grass", "Ground", "Ice", "Normal", "Poison", "Psychic", "Rock", "Steel", "Water",
            ]:
                downstreams[pokemon_type] = {}
                for i in range(1, 4):
                    entry = (console_id, GameChoice(f"FS_{pokemon_type}_{i}"))
                    if entry in G.nodes:
                        downstreams[pokemon_type][i] = set()
                        for rule in G.successors(entry):
                            useful, valid = self._all_downstream(rule)
                            if not valid:
                                raise ValueError("Not handled")
                            downstreams[pokemon_type][i].add(frozenset(useful))
            console_possibilities[console_id] = set()
            for pokemon_type, idxs in downstreams.items():
                if not idxs:
                    continue
                for tup in product(*idxs.values()):
                    console_possibilities[console_id].add(frozenset.union(*tup))
        self._prune()
        other_branches = self.try_branches()
        all_fs_entries = frozenset.union(*[frozenset.union(*ps) for ps in console_possibilities.values()])
        for ob in other_branches:
            all_in_branch = frozenset.union(*ob)
            assert(all_in_branch & all_fs_entries)
        to_cross_multiply = other_branches
        for possibilities in console_possibilities.values():
            if possibilities:
                to_cross_multiply.append(possibilities)
        return {frozenset.union(*tup) for tup in product(*to_cross_multiply)}


    def _not_consumed(self, entry):
        '''
        Return true if no rule consumes (as opposed to requires) this entry
        '''
        G = self.G
        for _, rule, consumed in list(G.out_edges(entry, data='consumed')):
            if consumed:
                return False
        return True

    def _all_downstream(self, rule):
        '''
        Return the set of useful nodes downstream of the rule, and whether they can all be acquired
        simply by applying the rule and then other safe rules
        '''
        G = self.G
        useful = {n[1].species for n in nx.descendants(G, rule) if (n, "END") in G.edges}
        copy = self.copy()
        copy = self.copy(set_parent=False)
        for entry in copy.G.predecessors(rule):
            if copy.G.nodes[entry]['count'] != math.inf:
                copy.G.nodes[entry]['count'] = 1
                copy.G.add_edge("START", entry)
        copy._apply_rule(rule)
        copy.apply_safe_rules()
        return useful, all(copy.dex[u] for u in useful)

    def _curtail_branch_output(self, branch):
        '''
        Reduce a large number of possibilities to a smaller set where
        - every entry is in at least one branch
        - every entry is *missing* from at least one branch
        - the branch with the most entries is guaranteed to be present (if a tie, at least one will be)
        '''
        assert(not frozenset.intersection(*branch))
        orig_branch = branch.copy()
        # Remove paths that are a subset of another path
        to_delete = set()
        for b1 in orig_branch:
            found_superset = False
            for b2 in orig_branch:
                if b1 == b2:
                    continue
                if b1.issubset(b2):
                    found_superset = True
                    break
            if found_superset:
                to_delete.add(b1)
                break
        orig_branch = orig_branch - to_delete
        all_entries = frozenset.union(*orig_branch)
        not_present = all_entries.copy()
        not_missing = all_entries.copy()
        final_branch = set()
        branch = [(b, b) for b in orig_branch]
        while not_present:
            best = max(branch, key=lambda b: len(b[1]))[0]
            final_branch.add(best)
            not_present = not_present - best
            not_missing = not_missing & best
            branch = [(b[0], b[1] - best) for b in branch]
            branch = [b for b in branch if b[1]]
        if not_missing:
            branch = [(b, (all_entries - b) & not_missing) for b in orig_branch]
            branch = [b for b in branch if b[1]]
        while not_missing:
            best, missing = max(branch, key=lambda b: len(b[1]))
            final_branch.add(best)
            not_missing = not_missing & best
            branch = [(b[0], b[1] - missing) for b in branch]
            branch = [b for b in branch if b[1]]
        return final_branch


_choice_idx = 0

@dataclass(frozen=True)
class Rule():
    in_cart_id: str
    out_cart_id: str
    consumed: frozenset
    required: frozenset
    output: frozenset
    repeats: float = math.inf
    is_transfer: bool = False

    def __repr__(self):
        consumed = set(self.consumed) or '{}'
        required = set(self.required) or '{}'
        return f"{'T' if self.is_transfer else ''}{self.in_cart_id}{consumed}{required}->{self.out_cart_id}{set(self.output)}({self.repeats})"

    @staticmethod
    def multi_init(game_id, unique_pokemon, consumed, required, output, repeats=math.inf):
        rules = set()
        new_consumed = []
        new_required = []
        input_species = set()
        for c in consumed:
            if isinstance(c, PokemonReq):
                input_species.add(c.species)
                matches = [p for p in unique_pokemon[c.species] if c.matches(p)]
                if not matches:
                    return rules
                new_consumed.append(matches)
            elif isinstance(c, Pokemon):
                input_species.add(c.species)
                new_consumed.append([c])
            else:
                new_consumed.append([c])
        for r in required:
            if isinstance(r, PokemonReq):
                input_species.add(r.species)
                matches = [p for p in unique_pokemon[r.species] if r.matches(p)]
                if not matches:
                    return rules
                new_required.append(matches)
            elif isinstance(r, Pokemon):
                input_species.add(r.species)
                new_consumed.append([r])
            else:
                new_required.append([r])
        new_consumed = list(product(*new_consumed))
        new_required = list(product(*new_required))

        full_output = set(output)
        for o in output:
            if isinstance(o, Pokemon) and o.species not in input_species:
                full_output.add(DexEntry(o.species))

        combos = list(product(new_consumed, new_required))
        choice = None
        if len(combos) > 1 and repeats != math.inf:
            global _choice_idx
            _choice_idx += 1
            choice = GameChoice(f"choice:{_choice_idx}")
            rules.add(Rule(game_id, game_id, frozenset(), frozenset(), frozenset({choice}), repeats=repeats))
            repeats = math.inf
        for consumed, required in combos:
            c = set(consumed)
            r = set(required)
            if choice is not None:
                c.add(choice)
            rules.add(Rule(game_id, game_id, frozenset(c), frozenset(r), frozenset(full_output), repeats))
        return rules

    @staticmethod
    def evolution_init(game_id, unique_pokemon, gender_func, pre, posts, item=None, other=None):
        rules = []
        for pre_pokemon in unique_pokemon[pre.species]:
            if not pre.matches(pre_pokemon):
                continue
            out = []
            for post_pokemon in posts:
                if pre_pokemon.gender is None:
                    post_gender = None
                else:
                    post_genders = gender_func(post_pokemon.species)
                    if pre_pokemon.gender in post_genders:
                        post_gender = pre_pokemon.gender
                    elif len(post_genders) > 1:
                        raise ValueError(f"Gender mismatch between '{pre_pokemon}' and '{post_pokemon}' evolution")
                    else: # Shedinja
                        post_gender = post_genders[0]
                out.append(Pokemon(post_pokemon.species, post_gender, pre_pokemon.props.union(post_pokemon.props)))
                out.append(DexEntry(post_pokemon.species))
            consumed = {pre_pokemon}
            required = set()
            if item is not None:
                consumed.add(item)
            if other is not None:
                required.add(other)
            # Need to call multi_init since required might contain a PokemonReq
            rules += Rule.multi_init(game_id, unique_pokemon, consumed, required, out)
        return rules

    @staticmethod
    def transfer(in_cart_id, out_cart_id, inputs, outputs):
        new_outputs = set(outputs)
        for i in inputs:
            if isinstance(i, Pokemon):
                new_outputs.add(DexEntry(i.species))
        for o in outputs:
            if isinstance(o, Pokemon):
                new_outputs.add(DexEntry(o.species))
        
        return Rule(in_cart_id, out_cart_id, frozenset(), frozenset(inputs), frozenset(new_outputs), is_transfer=True)


def main(args):
    pd.set_option('future.no_silent_downcasting', True)
    all_games = [[]]
    all_hardware = [[]]
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
        hardware = all_hardware[0]
        collection_saves, result = calc_dexes(games, hardware, args.flatten)

        if args.missing:
            result.print_missing()
        else:
            result.print_obtainable()
        print("\n---\n")
        print(f"TOTAL: {result.count()}")
        cart = collection_saves.main_cartridge
    elif num_collections == 2:
        raise ValueError("Games/hardware before the first '.' are universal, so there should be 0 or 2+ instances of '.'")
    else:
        collection_saves = []
        results = []
        for idx in range(1, num_collections):
            games = all_games[idx] + all_games[0]
            hardware = all_hardware[idx] + all_hardware[0]
            c, r = calc_dexes(games, hardware, args.flatten)
            collection_saves.append(c)
            results.append(r)
        pokemon2idx = {}
        for cs in collection_saves:
            for pokemon, row in cs.pokemon_list.iterrows():
                idx = row['index']
                pokemon2idx[pokemon] = idx
        result = MultiResult(results, pokemon2idx, args.version_exclusive)
        if args.all_present:
            result.print_all_present()
        elif args.compact:
            result.print_compact()
        else:
            result.print(obtainable=(not args.missing), skip_identical=(not args.full))


def calc_dexes(games, hardware, flatten=False):
    cartridges = [Cartridge(g, cl, c) for g, cl, c in games]

    collection = Collection(cartridges, hardware)
    new_collection_saves = CollectionSaves(collection)
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
    args = parser.parse_args()
    main(args)
