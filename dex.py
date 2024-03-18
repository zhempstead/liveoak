#!/usr/bin/python

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations, product, zip_longest
import math
from pathlib import Path
from typing import Optional

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pandas as pd

class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class DexReq():
    species: frozenset

@dataclass(frozen=True)
class Item():
    item: str

@dataclass(frozen=True)
class GameChoice():
    choice: str

@dataclass(frozen=True)
class PokemonReq():
    species: str
    gender: Optional[Gender] = None
    required: frozenset = frozenset([])
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

@dataclass(frozen=True)
class Game():
    name: str
    gen: int
    console: str
    core: bool
    props: frozenset = frozenset([])

GAMES = {g.name: g for g in [
    Game("RED", 1, "GB", True, frozenset({"NO_TRADE_ITEMS"})),
    Game("BLUE", 1, "GB", True, frozenset({"NO_TRADE_ITEMS"})),
    Game("YELLOW", 1, "GB", True, frozenset({"NO_TRADE_ITEMS"})),
    Game("STADIUM", 1, "N64", False, frozenset({"NO_EVOLVE"})),

    Game("GOLD", 2, "GB", True),
    Game("SILVER", 2, "GB", True),
    Game("CRYSTAL", 2, "GBC", True),
    Game("STADIUM2", 2, "N64", False, frozenset({"NO_BREED", "NO_EVOLVE"})),

    Game("RUBY", 3, "GBA", True),
    Game("SAPPHIRE", 3, "GBA", True),
    Game("EMERALD", 3, "GBA", True, frozenset({"GBA_WIRELESS"})),
    Game("FIRERED", 3, "GBA", True, frozenset({"GBA_WIRELESS"})),
    Game("LEAFGREEN", 3, "GBA", True, frozenset({"GBA_WIRELESS"})),
    Game("BOX", 3, "GCN", False, frozenset({"NO_BREED", "NO_EVOLVE"})),
    Game("COLOSSEUM", 3, "GCN", False, frozenset({"NO_BREED"})),
    Game("XD", 3, "GCN", False, frozenset({"NO_BREED"})),
    Game("BONUSDISC", 3, "GCN", False, frozenset({"NO_DEX"})),

    Game("DIAMOND", 4, "DS", True),
    Game("PEARL", 4, "DS", True),
    Game("PLATINUM", 4, "DS", True),
    Game("HEARTGOLD", 4, "DS", True),
    Game("SOULSILVER", 4, "DS", True),
    Game("POKEWALKER", 4, None, False, frozenset({"NO_DEX"})),
    Game("RANCH", 4, "Wii", False, frozenset({"NO_DEX", "SOFTWARE"})),
    Game("BATTLEREVOLUTION", 4, "Wii", False, frozenset({"NO_DEX"})),
    Game("RANGER", 4, "DS", False, frozenset({"NO_DEX"})),
    Game("SHADOWSOFALMIA", 4, "DS", False, frozenset({"NO_DEX"})),
    Game("GUARDIANSIGNS", 4, "DS", False, frozenset({"NO_DEX"})),

    Game("BLACK", 5, "DS", True),
    Game("WHITE", 5, "DS", True),
    Game("BLACK2", 5, "DS", True),
    Game("WHITE2", 5, "DS", True),
    Game("DREAMWORLD", 5, None, False, frozenset({"NO_DEX"})),
    Game("DREAMRADAR", 5, "3DS", False, frozenset({"NO_BREED", "NO_EVOLVE", "SOFTWARE"})),
]}

CONSOLES_WITH_SOFTWARE = [
    "Wii",
    "3DS",
]

BC_WF_TRAINERS = [
    "BRITNEY", "CARLOS", "COLLIN", "DAVE", "DOUG", "ELIZA", "EMI", "FREDERIC", "GENE", "GRACE",
    "HERMAN", "JACQUES", "KARENNA", "KEN", "LENA", "LEO", "LYNETTE", "MARIE", "MIHO", "MIKI",
    "MOLLY", "PIERCE", "PIPER", "RALPH", "ROBBIE", "ROSA", "RYDER", "SHANE", "SILVIA", "VINCENT",
]

class Generation():
    DATA = ["breed", "buy", "environment", "evolve", "fossil", "gift", "pickup_item", "pickup_pokemon", "trade", "wild", "wild_item"]

    def __init__(self, gen):
        self.id = gen
        self.dir = Path("data") / f"gen{self.id}"
        self.games = [g.name for g in GAMES.values() if g.gen == self.id]

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
            if "GAME" in df.columns:
                df["GAME"] = df["GAME"].apply(lambda game: parse_game(game, self.games))
            for game in self.games:
                if game in df.columns:
                    df[game] = df[game].fillna(0)
            for col in df.columns:
                df[col] = df[col].fillna(False)
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

class Cartridge():
    counter = {}

    def __init__(self, game, generation, gameid=None, console=None):
        self.game = game
        self.generation = generation
        self.console = console # Software games only
        self.id = gameid
        if self.id is None:      
            self.id = f"{self.game.name}_{Cartridge.counter.get(self.game.name, 1)}"
            Cartridge.counter[self.game.name] = Cartridge.counter.get(self.game.name, 1) + 1

        self.dex = {species: False for species in self.generation.pokemon_list.index}
        self.env = set()

        self.has_gender = self.game.gen >= 2 and "NO_BREED" not in self.game.props

        # Initialized later - indicates what other cartriges this one can communicate with
        self.transfers = {
            "POKEMON": [], # Transfer Pokemon w/o trading
            "TRADE": [], # Trade Pokemon (possible w/held items)
            "ITEMS": [], # Transfer items
            "MYSTERYGIFT": [], # Mystery gift
            "CONNECT": [], # Miscellaneous
        }

        if self.generation.environment is not None:
            for _, row in self.generation.environment.iterrows():
                if self.game.name in row.GAME:
                    self.env.add(row.ENVIRONMENT)

        self.pokemon = {}
        self.items = {}
        self.choices = {}

        self.path_child = None
        self.unique_pokemon = set()

    def copy(self):
        memo = {}
        c = Cartridge(self.game, self.generation, gameid=self.id)
        for species, pokemon in self.pokemon.items():
            c.pokemon[species] = pokemon.copy()
        c.items = self.items.copy()
        c.choices = self.choices.copy()
        c.rules = deepcopy(self.rules, memo)
        c.transfer_rules = {}
        for entry, rules in self.transfer_rules.items():
            c.transfer_rules[entry] = rules.copy()
        c.dex = self.dex.copy()
        c.transfers = self.transfers
        '''
        for k, transfers in self.transfers.items():
            c.transfers[k] = transfers.copy()
            if self in transfers:
                c.transfers[k].append(c)
        '''
        return c

    def transfer(self, other_cartridge, category):
        '''Register another cartridge as able to receive certain communications'''
        self.transfers[category].append(other_cartridge)
        if category != "CONNECT":
            self.transfers["CONNECT"].append(other_cartridge)

    def self_transfer(self, category):
        self.transfers[category].append(self)

    def init_rules(self):
        rules = []
        transfer_rules = []

        if self.generation.breed is not None and "NO_BREED" not in self.game.props and "NO_DEX" not in self.game.props:
            for _, row in self.generation.breed.iterrows():
                pr = self.parse_pokemon_input(row.PARENT)
                for gender in self.generation.genders(pr.species):
                    pg = replace(pr, gender=gender)
                    for c in self.parse_output(row.CHILD):
                        if len(c) > 1:
                            raise ValueError(f"Invalid child entry {c}")
                        item = Item(row.ITEM) if "ITEM" in row and row.ITEM else None
                        rules.append(Rule.breed(pg, c[0], gender != Gender.FEMALE, item))
                    
        if self.generation.buy is not None:
            for _, row in self.generation.buy.iterrows():
                if self.game.name in row.GAME:
                    exchange, valid = self.parse_input(row.get("EXCHANGE"))
                    if not valid:
                        continue
                    for o in self.parse_output(row.POKEMON_OR_ITEM, self.has_gender):
                        if len(o) != 1:
                            raise ValueError(f"Invalid buy entry {o}")
                        rules.append(Rule.buy(o[0], exchange))

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
                item = Item(row.ITEM) if row.ITEM else None
                pre = self.parse_pokemon_input(row.FROM, forbidden={"NOEVOLVE"})
                post = self.parse_output(row.TO, use_gender=False)
                if len(post) != 1:
                    raise ValueError("Confused")
                post = post[0]
                if self.has_gender:
                    if pre.gender is not None:
                        pre_genders = [pre.gender]
                    else:
                        pre_genders = self.generation.genders(pre.species)
                    for g in pre_genders:
                        post_with_g = []
                        for p in post:
                            post_genders = self.generation.genders(p.species)
                            if g in post_genders:
                                post_with_g.append(replace(p, gender=g))
                            elif len(post_genders) > 1:
                                raise ValueError(f"Gender mismatch between '{pre}' and '{p}' evolution")
                            else:
                                post_with_g.append(replace(p, gender=post_genders[0]))
                        rules.append(Rule.evolve(replace(pre, gender=g), post_with_g, item, other))
                else:
                    rules.append(Rule.evolve(pre, post, item, other))

        if self.generation.fossil is not None:
            for _, row in self.generation.fossil.iterrows():
                if self.game.name not in row.GAME:
                    continue
                reqs = None
                reqs, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                item = Item(row.ITEM)

                for p in self.parse_output(row.POKEMON, self.has_gender):
                    if len(p) != 1:
                        raise ValueError(f"Unexpected fossil {p}")
                    rules.append(Rule.fossil(item, p[0], reqs))

        if self.generation.gift is not None:
            for idx, row in self.generation.gift.iterrows():
                if self.game.name not in row.GAME:
                    continue
                choices = self.parse_gift_entry(row.POKEMON_OR_ITEM)
                reqs, valid = self.parse_input(row.REQUIRED)
                if not valid:
                    continue
                if len(choices) == 1:
                    rules.append(Rule.gift(choices[0], reqs))
                else:
                    rules.append(Rule.choice(str(idx), reqs))
                    for pokemon_or_items in choices:
                        rules.append(Rule.choose(str(idx), pokemon_or_items))

        if self.generation.pickup_item is not None and "NO_DEX" not in self.game.props:
            for _, row in self.generation.pickup_item.iterrows():
                if self.game.name not in row.GAME:
                    continue
                for pokemon in self.generation.pickup_pokemon["SPECIES"]:
                    req = self.parse_pokemon_input(pokemon, forbidden={"NOPICKUP"})
                    rules.append(Rule.pickup(Item(row.ITEM), req))

        if self.generation.trade is not None:
            for idx, row in self.generation.trade.iterrows():
                if self.game.name not in row.GAME:
                    continue
                if row.get("REQUIRED"):
                    reqs, valid = self.parse_input(row.REQUIRED)
                    if not valid:
                        continue
                else:
                    reqs = []
                if row.GIVE == "ANY":
                    give = None
                else:
                    give = self.parse_pokemon_input(row.GIVE) 
                gets = self.parse_output(row.GET)
                choice_id = None
                if len(gets) > 1:
                    choice_id = f"TRADE_GENDER_{row.GET}_{idx}"
                    rules.append(Rule.choice(choice_id, reqs, repeats=1))
                    reqs = None
                for get in gets:
                    if len(get) != 1:
                        raise ValueError(f"Invalid trade entry {get}")
                    get = get[0]
                    item = row.get("ITEM")
                    species = get.species
                    if species in trade_evo_pairs:
                        evolution, other = trade_evo_pairs(species) 
                        if (give is None or give.species == other) and item != "Everstone":
                            if choice_id is None:
                                choice_id = f"TRADE_EVOLVE_{row.GET}_{idx}"
                                rules.append(Rule.choice(choice_id, reqs, repeats=1))
                                reqs = None
                            if item:
                                item = Item(item)
                            # If your Pokemon holds an everstone, neither will evolve
                            rules.append(Rule.ingame_trade(give, get, item, None, choice_id, give_everstone=(give.species == other), reqs=reqs))
                            rules.append(Rule.ingame_trade(give, get, item, evolution, choice_id, reqs=reqs))
        
                    evolution = None
                    if species in trade_evos:
                        if item and item in trade_evos[species]:
                            evolution = trade_evos[species][item]
                            item = False
                        elif False in trade_evos[species] and (
                                item != "Everstone" or
                                # Held item loss glitch - Everstone is lost/ignored
                                self.generation.id == 3 or
                                # Everstone doesn't stop Kadabra from evolving since gen 4
                                (self.generation.id >= 4 and species == "Kadabra")
                        ):
                            evolution = trade_evos[species][False]
                            
                    if evolution is not None:
                        evolution = Pokemon(evolution, get.gender, get.props)
                    if item:
                        item = Item(item)
                    rules.append(Rule.ingame_trade(give, get, item, evolution, choice_id, reqs=reqs))


        wild_items = {}
        if self.generation.wild_item is not None and "NO_DEX" not in self.game.props:
            for _, row in self.generation.wild_item.iterrows():
                if self.game.name not in row.GAME:
                    continue
                if row.SPECIES not in wild_items:
                    wild_items[row.SPECIES] = []
                wild_items[row.SPECIES].append(row.ITEM)

        if self.generation.wild is not None and self.game.name in self.generation.wild.columns:
            for _, row in self.generation.wild.iterrows():
                pokemon_or_item = row.SPECIES
                items = [Item(i) for i in wild_items.get(pokemon_or_item, [])] or [None]
                p_or_is = self.parse_output(pokemon_or_item, self.has_gender)
                for idx, (consumed, reqs, count) in enumerate(self.parse_wild_entry(row[self.game.name])):
                    if count == 0:
                        continue
                    if count != math.inf and len(items) * len(p_or_is) > 1 and not consumed:
                        choice_id = f"WILD_{pokemon_or_item}_{idx}"
                        rules.append(Rule.choice(choice_id, reqs, repeats=count))
                        for p_or_i in p_or_is:
                            if len(p_or_i) != 1:
                                raise ValueError("Confused")
                            for item in items:
                                out = [p_or_i[0]]
                                if item:
                                    out.append(item)
                                rules.append(Rule.choose(choice_id, out))
                    else:
                        for p_or_i in p_or_is:
                            if len(p_or_i) != 1:
                                raise ValueError("Confused")
                            for item in items:
                                rules.append(Rule.wild(p_or_i[0], consumed, reqs, count, item))

        self.rules = Rules(rules)

        for pokemon in self.generation.tradeable_pokemon:
            for cart in self.transfers["POKEMON"]:
                if pokemon not in cart.generation.tradeable_pokemon:
                    continue
                transfer_rules.append(PokemonTransferRule(cart, pokemon, pokemon))
            for cart in self.transfers["TRADE"]:
                if pokemon not in cart.generation.tradeable_pokemon:
                    continue
                for item, evo in trade_evos.get(pokemon, {}).items():
                    if evo in cart.generation.tradeable_pokemon:
                        transfer_rules.append(PokemonTransferRule(cart, pokemon, evo, item or None))
                if pokemon in trade_evos_by_item.get(False, {}):
                    # Held item loss glitch - in gen 3, Everstone is lost/ignored
                    # Everstone doesn't stop Kadabra from evolving since gen 4
                    if self.generation.id != 3 and (pokemon != "Kadabra" or self.generation.id < 4):
                        transfer_rules.append(PokemonTransferRule(cart, pokemon, pokemon, "Everstone"))
                else:
                    transfer_rules.append(PokemonTransferRule(cart, pokemon, pokemon))
                if pokemon in trade_evo_pairs:
                    evo, other = trade_evo_pairs[pokemon]
                    other_evo = trade_evo_pairs[other][0]
                    if all(p in cart.generation.tradeable_pokemon for p in [evo, other, other_evo]):
                        transfer_rules.append(TradePairRule(cart, pokemon, evo, other, other_evo))
        for item in self.generation.tradeable_items:
            for cart in self.transfers["ITEMS"]:
                if item not in cart.generation.tradeable_items:
                    continue
                transfer_rules.append(ItemTransferRule(cart, item, None, item))
            if "NO_TRADE_ITEMS" in self.game.props:
                continue
            for cart in self.transfers["TRADE"]:
                if "NO_TRADE_ITEMS" in cart.game.props:
                    continue
                if item not in cart.generation.tradeable_items:
                    continue
                transfer_rules.append(ItemTransferRule(cart, item, None, item))

            for cart in self.transfers["MYSTERYGIFT"]:
                transfer_rules.append(ItemTransferRule(cart, None, f"MG_{item}", item))

        self.transfer_rules = {}
        for trule in transfer_rules:
            if isinstance(trule, PokemonTransferRule):
                if trule.in_species not in self.transfer_rules:
                    self.transfer_rules[trule.in_species] = []
                self.transfer_rules[trule.in_species].append(trule)

                if trule.item is not None:
                    if trule.item not in self.transfer_rules:
                        self.transfer_rules[trule.item] = []
                    self.transfer_rules[trule.item].append(trule)
            elif isinstance(trule, TradePairRule):
                if trule.in1 not in self.transfer_rules:
                    self.transfer_rules[trule.in1] = []
                self.transfer_rules[trule.in1].append(trule)
            elif isinstance(trule, ItemTransferRule):
                if trule.in_item is not None:
                    if trule.in_item not in self.transfer_rules:
                        self.transfer_rules[trule.in_item] = []
                    self.transfer_rules[trule.in_item].append(trule)
                if trule.in_choice is not None:
                    if trule.in_choice not in self.transfer_rules:
                        self.transfer_rules[trule.in_choice] = []
                    self.transfer_rules[trule.in_choice].append(trule)

    def init_uniques(self):
        for rule in self.rules.values():
            if rule.keep_props:
                continue
            for o in rule.output:
                if isinstance(o, Pokemon):
                    self.add_unique(o)

    def add_unique(self, pokemon):
        if pokemon in self.unique_pokemon:
            return
        self.unique_pokemon.add(pokemon)
        for cart in self.transfers["POKEMON"] + self.transfers["TRADE"]:
            cart.add_unique(pokemon)
        for out in self.outputs_of(pokemon):
            self.add_unique(out)

    def outputs_of(self, pokemon):
        '''Get the possible outputs using pokemon as an input among props-preserving rules'''
        out = set()
        for rule in self.rules.values():
            if not rule.keep_props:
                continue
            match = False
            for c in rule.consumed:
                if not isinstance(c, PokemonReq):
                    continue
                if c.matches(pokemon):
                    match = True
                    break
            if not match:
                continue
            for o in rule.output:
                if isinstance(o, Pokemon) and o.species != pokemon.species:
                    o = replace(o, props=pokemon.props)
                    out.add(o)

        for trule in self.transfer_rules.get(pokemon.species, []):
            if isinstance(trule, PokemonTransferRule):
                o = replace(pokemon, species=trule.out_species)
            else:
                o = replace(pokemon, species=trule.out1)
            if "TRANSFERRESET" in o.props:
                o = replace(o, props=frozenset())
            if "TRANSFERONE" in o.props:
                o = replace(o, props - set(["TRANSFERONE"]))
            out.add(o)
        return out

    def simplify_rules(self):
        unique_pokemon = {}
        for pokemon in self.unique_pokemon:
            if pokemon.species not in unique_pokemon:
                unique_pokemon[pokemon.species] = []
            unique_pokemon[pokemon.species].append(pokemon)

        to_delete = []
        to_add = []
        for rule_name, rule in self.rules.items():
            new_consumed = []
            new_required = []
            needs_change = False
            for c in rule.consumed:
                if isinstance(c, PokemonReq):
                    needs_change = True
                    matches = [p for p in unique_pokemon.get(c.species, []) if c.matches(p)]
                    for pokemon in matches:
                        for prop in pokemon.props:
                            if prop.startswith("ONE_"):
                                raise ValueError("Not handled")
                    new_consumed.append(matches)
                else:
                    new_consumed.append([c])
            for r in rule.required:
                if isinstance(r, PokemonReq):
                    needs_change = True
                    matches = [p for p in unique_pokemon.get(r.species, []) if r.matches(p)]
                    new_required.append(matches)
                else:
                    new_required.append([r])

            if needs_change:
                to_delete.append(rule_name)
                new_consumed = list(product(*new_consumed))
                new_required = list(product(*new_required))
                new_combos = list(product(new_consumed, new_required))
                choice_rule = None
                if len(new_combos) > 1 and rule.repeats != math.inf:
                    choice_rule = Rule.choice("choice:" + rule_name, [], repeats=rule.repeats)
                    to_add.append(choice_rule)

                for idx, (consumed, required) in enumerate(product(new_consumed, new_required)):
                    output = rule.output
                    if choice_rule is None:
                        repeats = rule.repeats
                    else:
                        consumed = list(consumed) + [GameChoice("choice:" + rule_name)]
                        repeats = math.inf
                    if len(new_combos) == 1:
                        new_name = rule_name
                    else:
                        new_name = f"{rule_name}_{idx}"
                    if rule.keep_props:
                        props = None
                        for c in consumed:
                            if isinstance(c, Pokemon):
                                props = c.props
                                break
                        new_output = []
                        for o in output:
                            if isinstance(o, Pokemon):
                                new_output.append(replace(o, props=props))
                            else:
                                new_output.append(o)
                        output = new_output
                    to_add.append(Rule(new_name, consumed, required, output, repeats, dex=rule.dex))

        for rule_name in to_delete:
            del self.rules[rule_name]
        for rule in to_add:
            self.rules[rule.name] = rule

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
            props.add(f"ONE_{self.game.id}")
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
            elif e in GAMES:
                if not any([c.id != self.id and c.game.name == e for c in self.transfers["CONNECT"]]): 
                    valid = False
            elif e.startswith("DEX_"):
                out.append(DexReq(frozenset(p for p in self.generation.pokemon_list.index if self.generation.pokemon_list.loc[p, e])))
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
            out += list(self.parse_output(e, use_gender))
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
        

    def acquire(self, entry, count, communicate=True):
        prev_count = self.get_count(entry)
        if prev_count == math.inf:
            return
        if self.path_child is not None:
            self.path_child.acquire(entry, count, False)
        if isinstance(entry, Pokemon):
            self.dex[entry.species] = True

            if entry.species not in self.pokemon:
                self.pokemon[entry.species] = {}
            self.pokemon[entry.species][entry] = self.pokemon[entry.species].get(entry, 0) + count

            if "NOTRANSFER" in entry.props or not communicate:
                return

            pokemon = entry
            send_count = math.inf
            if "TRANSFERRESET" in entry.props:
                # Pokemon that is only special to the given save
                pokemon = replace(pokemon, props=frozenset({}))
            if "TRANSFERONE" in entry.props:
                # Pokemon that can be received only once by a given save of the destination game
                pokemon = replace(pokemon, props=pokemon.props - set({"TRANSFERONE"}))
                send_count = 1
            for prop in entry.props:
                # Pokemon that can be obtained only once from a cartridge, with no possibility of a reset
                if prop.startswith("ONE_"):
                    if prev_count >= 1:
                        return
                    cartsets = self.unique_pokemon_transfers(entry.species)
                    if len(cartsets) > 1:
                        raise ValueError("Not handled")
                    cartset = gamesets[0]
                    global all_cartridges
                    for cart_id in gameset:
                        if cart_id == self.id:
                            continue
                        all_cartridges[cart_id].acquire(entry, 1, communicate=False)
                    return

            trules = self.transfer_rules.get(pokemon.species, [])
            for trule in trules:
                if not isinstance(trule, PokemonTransferRule):
                    continue
                if trule.item is not None and self.items.get(trule.item) == 0:
                    continue
                transfer_pokemon = replace(pokemon, species=trule.out_species)
                trule.cart.acquire(transfer_pokemon, count=send_count)
            for trule in trules:
                if not isinstance(trule, TradePairRule):
                    continue
                if send_count != math.inf:
                    raise ValueError("Not handled")
                matches2 = self.get_matches(PokemonReq(trule.in2, forbidden=frozenset(["NOTRANSFER"])))
                if matches2:
                    transfer1 = replace(pokemon, species=trule.out1)
                    trule.cart.acquire(transfer1, count=send_count)
                for in2, count in matches2:
                    if count != math.inf:
                        raise ValueError("Not handled")
                    transfer2 = replace(in2, species=trule.out2)
                    trule.cart.acquire(transfer2, count=send_count)

        elif isinstance(entry, Item):
            item = entry.item
            self.items[item] = self.items.get(item, 0) + count

            if not communicate:
                return

            for trule in self.transfer_rules.get(item, []):
                if isinstance(trule, PokemonTransferRule):
                    for in_pokemon, count in self.get_matches(PokemonReq(trule.in_species, forbidden=frozenset(["NOTRANSFER"]))):
                        out_pokemon = replace(in_pokemon, species=trule.out_species)
                        if "TRANSFERRESET" in out_pokemon.props:
                            out_pokemon = replace(out_pokemon, props=frozenset({}))
                        if "TRANSFERONE" in out_pokemon.props:
                            raise ValueError("Not handled")
                        trule.cart.acquire(out_pokemon, count=math.inf)
                elif isinstance(trule, ItemTransferRule):
                    if trule.in_choice is not None and self.choices.get(trule.in_choice, 0) == 0:
                        continue
                    trule.cart.acquire(Item(trule.out_item), count=math.inf)

        elif isinstance(entry, GameChoice):
            choice = entry.choice
            self.choices[choice] = self.choices.get(choice, 0) + count

            if not communicate:
                return

            for trule in self.transfer_rules.get(choice, []):
                if trule.in_item is not None and self.items.get(trule.in_item, 0) == 0:
                    continue
                trule.cart.acquire(Item(trule.out_item), count=math.inf)
    
    def apply_all_safe_rules(self):
        while True:
            changed = False
            for name in list(self.rules.keys()):
                changed |= self.apply_rule_if_safe(name)

            if not changed:
                break

    def check_rule_safe(self, rule):
        if not all(self.get_count(c) == math.inf for c in rule.consumed):
            return False
        for r in rule.required:
            if isinstance(r, DexReq):
                if not all(self.dex[p] for p in r.species):
                    return False
            elif self.get_count(r) == 0:
                return False
        return True

    def check_rule_useful(self, rule):
        if rule.repeats == 0:
            return False
        if any(self.get_count(o) != math.inf for o in rule.output):
            return True
        return False

    def check_rule_possible(self, rule):
        if rule.repeats == 0:
            return False
        if not all(self.get_count(c) >= 1 for c in rule.consumed):
            return False
        for r in rule.required:
            if isinstance(r, DexReq):
                if not all(self.dex[p] for p in r.species):
                    return False
            elif self.get_count(r) == 0:
                return False
        return True


    def apply_rule_if_possible(self, rule_name):
        rule = self.rules[rule_name]
        if not self.check_rule_possible(rule):
            return False
        for c in rule.consumed:
            if isinstance(c, Pokemon):
                self.pokemon[c.species][c] -= 1
                if self.pokemon[c.species][c] == 0:
                    del self.pokemon[c.species][c]
                    if not self.pokemon[c.species]:
                        del self.pokemon[c.species]
            elif isinstance(c, Item):
                self.items[c.item] -= 1
                if self.items[c.item] == 0:
                    del self.items[c.item]
            elif isinstance(c, GameChoice):
                self.choices[c.choice] -= 1
                if self.choices[c.choice] == 0:
                    del self.choices[c.choice]

        for o in rule.output:
            self.acquire(o, count=1)
        rule.repeats -= 1
        if rule.repeats == 0:
            del self.rules[rule_name]
        return True
 
    def apply_rule_if_safe(self, rule_name):
        rule = self.rules[rule_name]
        if not self.check_rule_useful(rule):
            del self.rules[rule_name]
            return False
        if not self.check_rule_safe(rule):
            return False
        if rule.keep_props:
            pokemon_consumed = [c for c in rule.consumed if isinstance(c, PokemonReq)]
            if len(pokemon_consumed) != 1:
                raise ValueError(f"Keep props rule with multiple consumed Pokemon: {pokemon_consumed}")
            pokemon_consumed = pokemon_consumed[0]
            matches = [m for m, c in self.get_matches(pokemon_consumed) if c == math.inf]
            if not matches:
                raise ValueError("Confused")
            changed = False
            for o in rule.output:
                o = replace(o, props=pokemon_consumed.props) 
                if self.get_count(o) == math.inf:
                    continue
                self.acquire(o, count=math.inf)
                changed = True
            for d in rule.dex:
                if not self.dex[d.species]:
                    changed = True
                    self.dex[d.species] = True
            return changed
        else:
            for o in rule.output:
                self.acquire(o, count=rule.repeats)
            for d in rule.dex:
                self.dex[d.species] = True
            del self.rules[rule_name]
            return True


    def rule_graph(self, draw=False):
        required = set()
        G = nx.DiGraph()
        G.add_node("START")
        for rule_name, rule in self.rules.items():
            for c in rule.consumed:
                count = self.get_count(c)
                if count == math.inf:
                    continue
                G.add_edge(c, rule_name, kind="consumed")
                if count > 0:
                    G.add_edge("START", c)
            for r in rule.required:
                if isinstance(r, DexReq):
                    G.add_edge(r, rule_name)
                    if all(self.dex[s] for s in r.species):
                        G.add_edge("START", r)
                else:
                    count = self.get_count(r)
                    if count == math.inf:
                        continue
                    G.add_edge(r, rule_name, kind="required")
                    if count > 0:
                        required.add(r)
                        G.add_edge("START", r)
            for o in rule.output:
                count = self.get_count(o)
                if count == math.inf:
                    continue
                G.add_edge(rule_name, o)
                if isinstance(o, Pokemon) and not self.dex[o.species]:
                    G.add_edge(rule_name, o.species)
            for d in rule.dex:
                if self.dex[species]:
                    continue
                G.add_edge(rule_name, d)
        for r in required:
            edges = G.edges(r, 'kind')
            if not any([edge[2] == 'consumed' for edge in edges]):
                to_remove = [(edge[0], edge[1]) for edge in edges if edge[2] == 'required']
                for edge in to_remove:
                    G.remove_edge(*edge)
        connected = nx.node_connected_component(G.to_undirected(), "START")
        G.remove_nodes_from([n for n in G if n not in connected])
        for node in self.get_useful(G.nodes):
                G.add_edge(node, "END")

        if "END" not in G.nodes:
            return G

        if draw:
            A = nx.nx_agraph.to_agraph(G)
            A.layout('dot')
            A.draw('graph.png')
        return G
 
    def get_useful(self, entries):
        '''
        Return the subset of entries that is directly useful for this cartridge, or might
        be useful for another cartridge
        '''
        useful = set()
        item_carts = set(self.transfers["ITEMS"])
        if "NO_TRADE_ITEMS" not in self.game.props:
            for trade_cart in self.transfers["TRADE"]:
                if "NO_TRADE_ITEMS" not in trade_cart.game.props:
                    item_carts.add(trade_cart)
        for entry in entries:
            if isinstance(entry, Item):
                item = entry.item
                if item in self.generation.tradeable_items:
                    for cart in item_carts:
                        if cart.items.get(item, 0) != math.inf:
                            useful.add(item)
                            break 
            elif isinstance(entry, GameChoice):
                choice = entry.choice
                if choice.startswith("MG_"):
                    _, _, item = choice.partition("_")
                    for cart in self.transfers["MYSTERYGIFT"]:
                        if cart.items.get(item, 0) != math.inf:
                            useful.add(entry)
                            break
            elif isinstance(entry, str):
                if entry in self.dex and not self.dex[entry]:
                    useful.add(entry)
        return useful


    def all_safe_paths(self):
        self.apply_all_safe_rules()
        G = self.rule_graph()

        next_steps = {}
        total_paths = {}

        for path in nx.all_simple_paths(G, "START", "END"):
            for idx in range(1, len(path) - 2, 2):
                entry = path[idx]
                rule_name = path[idx + 1]
                if entry not in next_steps:
                    next_steps[entry] = {}
                    total_paths[entry] = 0
                if rule_name not in next_steps[entry]:
                    next_steps[entry][rule_name] = 0
                total_paths[entry] += 1
                next_steps[entry][rule_name] += 1

        while True:
            any_changed = False
            for rule_name, rule in list(self.rules.items()):
                if rule_name not in G.nodes:
                    continue
                if not self.check_rule_possible(rule):
                    continue
                is_safe = True

                def consumed_safe(c):
                    pathcount = total_paths.get(c, 0)
                    if pathcount == 0 or pathcount > self.get_count(c):
                        return False
                    if next_steps.get(c, {}).get(rule_name, 0) == 0:
                        return False
                    return True

                consumed_choices = []
                if not all(consumed_safe(c) for c in rule.consumed):
                    continue
                if self.apply_rule_if_possible(rule_name):
                    any_changed = True
                    for c in rule.consumed:
                        total_paths[c] -= 1
                        next_steps[c][rule_name] -= 1
            if not any_changed:
                break
            else:
                self.apply_all_safe_rules()

    def try_paths(self, only_side_effects=False):
        start_end = set(["START", "END"])
        component_dexsets = []

        G = self.rule_graph()
        subG = G.subgraph([n for n in G.nodes if n not in start_end])
        for nodes in nx.connected_components(subG.to_undirected()):
            Gcomp = G.copy()
            Gcomp.remove_nodes_from([n for n in Gcomp if n not in nodes and n not in start_end])
            if only_side_effects:
                self.explore_component(Gcomp)
            else:
                component_dexsets.append(self.explore_component(Gcomp))

        if not only_side_effects:
            final_dexes = set()
            if not component_dexsets:
                component_dexsets.append(set([tuple(self.dex.items())]))
            for dexes in product(*component_dexsets):
                dex = tuple(
                    (p, any([d[idx][1] for d in dexes]))
                    for idx, (p, _) in enumerate(dexes[0])
                )
                final_dexes.add(dex)
            return final_dexes
        
    def explore_component(self, G, spaces=''):
        final_dexes = set()
        ruleset = set(G.nodes).intersection(self.rules.keys())

        for rule_name in ruleset:
            rule = self.rules[rule_name]
            if not self.check_rule_possible(rule):
                continue
            game_copy = self.copy()
            self.path_child = game_copy
            game_copy.apply_rule_if_possible(rule_name)
            game_copy.all_safe_paths()
            Gcopy = G.copy()
            dexes = game_copy.explore_component(Gcopy, spaces=spaces + '  ')
            self.path_child = None
            final_dexes = final_dexes.union(dexes)
        
        if not final_dexes:
            final_dexes.add(tuple(self.dex.items()))
        return final_dexes

    def handle_special(self, collection):
        if self.game.name == "BLACK":
            # Black City
            # Exactly 10 trainers have useful items for sale (evolution stones), but unlimited
            # amounts of all can be found in the wild.
            return False

        elif self.game.name == "WHITE":
            # White Forest
            if self.transfers["TRADE"]:
                # If we can trade, we can reset and get every trainer
                for trainer in BC_WF_TRAINERS:
                    self.acquire(GameChoice(f"WF_{trainer}"), 1)
                return True
            elif self.transfers["POKEMON"]:
                if self.id == collection.main_cartridge.id:
                    raise ValueError("Not handled")
                # If we can transfer, other cartridges can get every trainer.
                # The exact combinations shouldn't matter
                trainers = [GameChoice(f"WF_{trainer}") for trainer in BC_WF_TRAINERS]
                rules = [Rule.choice("WF", [])]
                rules.append(Rule.choose("WF", trainers[:10]))
                rules.append(Rule.choose("WF", trainers[10:20]))
                rules.append(Rule.choose("WF", trainers[20:30]))
                for rule in rules:
                    self.rules[rule.name] = rule
                return True
            else:
                if len(collection.cartridges) == 1:
                    # Every White Forest Pokemon is present in at least one of these groups, and
                    # missing from at least one of these groups.
                    #
                    # The first group is one of the optimum combinations.
                    tgroups = [
                        ['BRITNEY', 'CARLOS', 'DOUG', 'ELIZA', 'EMI', 'FREDERIC', 'JACQUES', 'LENA', 'LYNETTE', 'SILVIA'],
                        ['DAVE', 'GENE', 'LEO', 'LYNETTE', 'MIHO', 'MIKI', 'PIERCE', 'PIPER', 'ROBBIE', 'SHANE'],
                        ['COLLIN', 'GRACE', 'HERMAN', 'KARENNA', 'KEN', 'MARIE', 'MOLLY', 'ROSA', 'RYDER', 'VINCENT'],
                    ]
                    cgroups = [[GameChoice(f"WF_{t}") for t in tgroup] for tgroup in tgroups]
                    rules = [Rule.choice("WF", [])] + [Rule.choose("WF", cg) for cg in cgroups]
                    for rule in rules:
                        self.rules[rule.name] = rule
                    return True

                # If White is paired with Dream World or any 4th-gen game, all needed White
                # Forest Pokemon can be acquired with fewer than 10 trainers
                extra_pokemon = {}
                for trainer in BC_WF_TRAINERS:
                    game_copy = self.copy()
                    game_copy.acquire(GameChoice(f"WF_{trainer}"), 1)
                    game_copy.apply_all_safe_rules()
                    extras = {species for species in self.dex if game_copy.dex[species] and not self.dex[species]}
                    if extras:
                        extra_pokemon[trainer] = extras
                if len(extra_pokemon) <= 10:
                    for trainer in extra_pokemon.keys():
                        self.acquire(GameChoice(f"WF_{trainer}"), 1)
                    return True
                useful_trainers = set(extra_pokemon.keys())
                useful_pokemon = set().union(*extra_pokemon.values())
                num_useful = len(useful_pokemon)
                best_trainers = set()
                best_extra_count = 0
                for idx, trainerset in enumerate(combinations(useful_trainers, 10)):
                    extra = set().union(*(extra_pokemon[t] for t in trainerset))
                    if len(extra) == num_useful:
                        for trainer in trainerset:
                            self.acquire(GameChoice(f"WF_{trainer}"), 1)
                        return True
                raise ValueError("Shouldn't reach here")
        return False


    def get_matches(self, req):
        matches = []
        if isinstance(req, PokemonReq):
            return self.get_pokemon_matches(req)
        elif isinstance(req, Item):
            if self.items.get(req.item) >= 0:
                return [(req.item, self.items[req.item])]
        elif isinstance(req, GameChoice):
            if self.choices.get(req.choice) >= 0:
                return [(req.choice, self.choices[req.choice])]
        else:
            raise ValueError(f"Invalid requirement {req}")

    def get_pokemon_matches(self, preq):
        out = []
        for pokemon, count in self.pokemon.get(preq.species, {}).items():
            if count == 0:
                continue
            if not preq.matches(pokemon):
                continue
            out.append([pokemon, count])
        return out

    def get_count(self, entry):
        if isinstance(entry, Pokemon):
            return self.pokemon.get(entry.species, {}).get(entry, 0)
        elif isinstance(entry, Item):
            return self.items.get(entry.item, 0)
        elif isinstance(entry, GameChoice):
            return self.choices.get(entry.choice, 0)
        else:
            import pdb; pdb.set_trace()
            raise ValueError(f"Invalid entry {entry}")

    def unique_pokemon_transfers(self, species, already_visited=None):
        if already_visited is None:
            already_visited = frozenset()
        visited = already_visited.union(set([self.id]))
        possibilities = set()
        for cart in set(self.transfers["POKEMON"] + self.transfers["TRADE"]):
            if cart.id in already_visited:
                continue
            if species not in cart.generation.tradeable_pokemon:
                continue
            for poss in cart.unique_pokemon_transfers(species, visited):
                possibilities.add(poss)

        if not possibilities:
            return frozenset({visited})

        to_remove = set()
        for poss1 in possibilities:
            for poss2 in possibilities:
                if poss2 in to_remove:
                    continue
                if poss1 == poss2:
                    continue
                if poss1.issubset(poss2):
                    to_remove.add(poss1)
                    break
        for poss in to_remove:
            possibilities.remove(poss)
        return frozenset(possibilities)
                    


class Collection():
    def __init__(self, cartridges, hardware):
        if len(cartridges) == 0:
            raise ValueError("No games!")
        self.main_cartridge = cartridges[0]
        if "NO_DEX" in self.main_cartridge.game.props:
            raise ValueError(f"Can't use {self.main_cartridge.game.name} as main game")
        self.cartridges = cartridges
        if any([c.has_gender for c in self.cartridges]):
            for c in self.cartridges:
                c.has_gender = True

        self.hardware = hardware
        self.games = set([c.game for c in self.cartridges])
        self.pokemon_list = self.main_cartridge.generation.pokemon_list

        for game in self.games:
            if not self.can_play(game):
                raise ValueError(f"Game {game.name} cannot be played with specified hardware")
        self._init_interactions()

        if len(self.cartridges) > 1:
            G = nx.DiGraph()
            for cart in self.cartridges:
                G.add_node(cart.id)
                for kind, cart2s in cart.transfers.items():
                    if kind == "CONNECT":
                        continue
                    for cart2 in cart2s:
                        G.add_edge(cart.id, cart2.id)
            for cart in self.cartridges:
                for cart2 in cart.transfers["CONNECT"]:
                    if not G.has_edge(cart.id, cart2.id):
                        G.add_edge(cart2.id, cart.id)
            for cart in self.cartridges:
                if cart == self.main_cartridge:
                    continue
                if not nx.has_path(G, cart.id, self.main_cartridge.id):
                    raise ValueError(f"Game {cart.game.name} cannot interact with main game {self.main_cartridge.game.name} (possibly due to specified hardware)")

    def can_play(self, game):
        # Pokewalker doesn't require a console
        if game.console is None:
            return True

        # GEN I
        if game.console == "GB":
            if self.hardware.get("GB", 0) + self.hardware.get("GBP", 0) + self.hardware.get("GBC", 0) + self.hardware.get("GBA", 0) >= 1:
                return True
            if self.hardware.get("N64", 0) >= 1 and self.hardware.get("NUS-019", 0) >= 1:
                if "STADIUM2" in self.game_names:
                    return True
                if game.generation.id == 1 and "STADIUM" in self.game_names:
                    return True
        elif game.console == "N64":
            if self.hardware.get("N64", 0) >= 1:
                return True
        # GEN II
        elif game.console == "GBC":
            if self.hardware.get("GBC", 0) + self.hardware.get("GBA", 0) >= 1:
                return True
            if self.hardware.get("N64", 0) >= 1 and self.hardware.get("NUS-019", 0) >= 1 and "STADIUM2" in self.game_names:
                return True
        # GEN III
        elif game.console == "GBA":
            if self.hardware.get("GBA", 0) + self.hardware.get("GBM", 0) + self.hardware.get("DS", 0) >= 1:
                return True
        elif game.console == "GCN":
            if self.hardware.get("GCN", 0) + self.hardware.get("Wii", 0) >= 1:
                return True
        # GEN IV
        elif game.console == "DS":
            if self.hardware.get("DS", 0) + self.hardware.get("DSi", 0) + self.hardware.get("3DS", 0) >= 1:
                return True
        elif game.console == "Wii":
            if self.hardware.get("Wii", 0) + self.hardware.get("WiiU", 0) >= 1:
                return True

        # GEN V
        elif game.console == "3DS":
            if self.hardware.get("3DS", 0) >= 1:
                return True

        return False

    def _init_interactions(self):
        gb1s = self.hardware.get("GB", 0)
        gb2s = self.hardware.get("GBP", 0) + self.hardware.get("GBC", 0) + self.hardware.get("GBA", 0)
        gbcs = self.hardware.get("GBC", 0) + self.hardware.get("GBA", 0)
        gba1s = self.hardware.get("GBA", 0)
        gba2s = self.hardware.get("GBM", 0)
        dss = self.hardware.get("DS", 0) + self.hardware.get("DSi", 0) + self.hardware.get("3DS", 0)


        connect_gb1s = \
            self.hardware.get("DMG-04", 0) >= 1 or \
            (self.hardware.get("MGB-010", 0) >= 1 and self.hardware.get("DMG-14", 0) >= 1) or \
            (self.hardware.get("CGB-003", 0) >= 1 and self.hardware.get("DMG-14", 0) >= 2)
        connect_gb1_gb2 = \
            (self.hardware.get("DMG-04", 0) >= 1 and self.hardware.get("MGB-04", 0) >= 1) or \
            self.hardware.get("MGB-010", 0) >=1 or \
            (self.hardware.get("CGB-003", 0) >= 1 and self.hardware.get("DMG-14", 0) >= 1)
        connect_gb2s = \
            (self.hardware.get("DMG-04", 0) >= 1 and self.hardware.get("MGB-04", 0) >= 2) or \
            self.hardware.get("MGB-010", 0) >= 1 or \
            self.hardware.get("CGB-003", 0) >= 1

        connect_gbas_retro = connect_gb2s or (self.hardware.get("AGB-005", 0) >= 2)
        connect_gba1s = \
            self.hardware.get("AGB-005", 0) >= 1 or \
            (self.hardware.get("OXY-008", 0) >= 1 and self.hardware.get("OXY-009", 0) >= 2)
        connect_gba1_gba2 = self.hardware.get("OXY-008", 0) >= 1 and self.hardware.get("OXY-009", 0) >= 1
        connect_gba2s = self.hardware.get("OXY-008", 0) >= 1

        wireless_gba1s = self.hardware.get("AGB-015", 0) >= 2
        wireless_gba1_gba2 = self.hardware.get("AGB-015", 0) >= 1 and self.hardware.get("OXY-004", 0) >= 1
        wireless_gba2s = self.hardware.get("AGB-015", 0) >= 1


        gb_trade = \
            (gb1s >= 2 and connect_gb1s) or \
            (gb1s >= 1 and gb2s >= 1 and connect_gb1_gb2) or \
            (gb2s >= 1 and connect_gb2s) or \
            (gba1s >= 2 and connect_gbas_retro)

        gb_gbc_trade = \
            (gb1s >= 1 and gbcs >= 1 and connect_gb1_gb2) or \
            (gb2s >= 2 and gbcs >= 1 and connect_gb2s) or \
            (gba1s >= 2 and connect_gbas_retro)

        gbc_trade = \
            (gbcs >= 2 and connect_gb2s) or \
            (gba1s >= 2 and connect_gbas_retro)

        gbc_ir = self.hardware.get("GBC", 0) >= 2

        gba_trade = \
            (gba1s >= 2 and connect_gba1s) or \
            (gba1s >= 1 and gba2s >= 1 and connect_gba1_gba2) or \
            (gba2s >= 1 and connect_gba2s)

        gba_wireless = \
            (gba1s >= 2 and wireless_gba1s) or \
            (gba1s >= 1 and gba2s >= 2 and wireless_gba1_gba2) or \
            (gba2s >= 2 and wireless_gba2s)

        ps1_transfer = \
            self.hardware.get("N64", 0) >= 1 and \
            self.hardware.get("NUS-019", 0) >= 1 and \
            "STADIUM" in self.game_names
        ps1_trade = ps1_transfer and self.hardware["NUS-019"] >= 2

        ps2_transfer = \
            self.hardware.get("N64", 0) >= 1 and \
            self.hardware.get("NUS-019", 0) >= 1 and \
            "STADIUM2" in self.game_names
        ps2_trade = ps2_transfer and self.hardware["NUS-019"] >= 2

        gba_gcn = \
            gba1s >= 1 and \
            (self.hardware.get("GCN", 0) + self.hardware.get("Wii", 0) >= 1) and \
            self.hardware.get("DOL-011", 0) >= 1

        gba_ds = self.hardware.get("DS", 0) >= 1
        ds_trade = dss >= 2

        for cart1 in self.cartridges:
            game1 = cart1.game
            for cart2 in self.cartridges:
                game2 = cart2.game
                if cart1.id == cart2.id:
                    continue

                if game1.gen == 1 and game1.core:
                    if game2.gen == 1 and game2.core:
                        # Pokemon Stadium 2 is the only way to transfer items between Gen 1 games
                        if ps2_transfer:
                            cart1.transfer(cart2, "ITEMS")
                        if gb_trade or ps1_trade or ps2_trade:
                            cart1.transfer(cart2, "TRADE")
                    elif game2.name == "STADIUM":
                        if ps1_transfer:
                            cart1.transfer(cart2, "POKEMON")
                    elif game2.gen == 2 and game2.core:
                        trade = gb_trade
                        if game2.name == "CRYSTAL":
                            trade = gb_gbc_trade
                        if trade or ps2_trade:
                            cart1.transfer(cart2, "TRADE")
                    elif game2.name == "STADIUM2":
                        if ps2_transfer:
                            cart1.transfer(cart2, "POKEMON")
                elif game1.name == "STADIUM":
                    if game2.gen == 1 and game2.core:
                        if ps1_transfer:
                            cart1.transfer(cart2, "POKEMON")
                elif game1.gen == 2 and game1.core:
                    trades = [gb_trade, gb_gbc_trade, gbc_trade]
                    crystals = 0
                    if game1.name == "CRYSTAL":
                        crystals += 1

                    if game2.gen == 1 and game2.core:
                        if trades[crystals] or ps2_trade:
                            cart1.transfer(cart2, "TRADE")
                            # Pokemon holding items can be traded to gen 1. Gen 1 can't access the
                            # items, but they will be preserved if the Pokemon is later traded back
                            # to gen 2, including the same cartridge.
                            cart1.self_transfer("ITEMS")
                    elif game2.gen == 2 and game2.core:
                        if game2.name == "CRYSTAL":
                            crystals += 1
        
                        # Pokemon holding items can be traded to gen 1. Gen 1 can't access the
                        # items, but they will be preserved if the Pokemon is later traded back to
                        # gen 2. So two copies of Crystal can transfer items back and forth via gen
                        # 1 even with only one GBC.
                        if crystals == 2 and gb_gbc_trade and any([g.gen == 1 and g.core for g in self.games]):
                            cart1.transfer(cart2, "ITEMS")
                        if trades[crystals] or ps2_trade:
                            cart1.transfer(cart2, "TRADE")
                        if ps2_transfer:
                            cart1.transfer(cart2, "ITEMS")
                        if gbc_ir:
                            cart1.transfer(cart2, "MYSTERYGIFT")
                    elif cart2.name == "STADIUM2":
                        if ps2_transfer:
                            cart1.transfer(cart2, "POKEMON")
                elif game1.name == "STADIUM2":
                    if game2.gen == 1 and game2.core:
                        if ps2_transfer:
                            cart1.transfer(cart2, "POKEMON")
                    elif game2.gen == 2 and game2.core:
                        if ps2_transfer:
                            cart1.transfer(cart2, "POKEMON")
                            cart1.self_transfer("MYSTERYGIFT")
                elif game1.gen == 3 and game1.core:
                    if game2.gen == 3 and game2.core:
                        if gba_trade or \
                            (gba_wireless and "GBA_WIRELESS" in game1.props and "GBA_WIRELESS" in game2.props):
                            cart1.transfer(cart2, "TRADE")
                        if gba_wireless and "GBA_WIRELESS" in game1.props and "GBA_WIRELESS" in game2.props:
                            cart1.transfer(cart2, "TRADE")
                    elif game2.name == "BOX":
                        if gba_gcn:
                            cart1.transfer(cart2, "POKEMON")
                            cart1.transfer(cart2, "ITEMS")
                    elif game2.name in ["COLOSSEUM", "XD"]:
                        if gba_gcn:
                            cart1.transfer(cart2, "TRADE")
                    elif game2.name == "BONUSDISC":
                        if game1.name in ["RUBY", "SAPPHIRE"]:
                            cart1.transfer(cart2, "CONNECT")
                    elif game2.gen == 4 and game2.core:
                        if gba_ds:
                            cart1.transfer(cart2, "POKEMON")
                            cart1.transfer(cart2, "ITEMS")
                elif game1.name == "BOX":
                    if game2.gen == 3 and game2.core:
                        if gba_gcn:
                            cart1.transfer(cart2, "POKEMON")
                            cart1.transfer(cart2, "ITEMS")
                elif game1.name in ["COLOSSEUM", "XD"]:
                    if game2.gen == 3 and game2.core:
                        if gba_gcn:
                            cart1.transfer(cart2, "TRADE")
                elif game1.gen == 4 and game1.core:
                    if game2.gen == 4 and game2.core:
                        if ds_trade:
                            cart1.transfer(cart2, "TRADE")
                    elif game2.name == "POKEWALKER":
                        if game1.name in ["HEARTGOLD", "SOULSILVER"]:
                            cart1.transfer(cart2, "CONNECT")
                    elif game2.name == "RANCH":
                        if game1.name in ["DIAMOND", "PEARL"]:
                            cart1.transfer(cart2, "CONNECT")
                    elif game2.name in ["BATTLEREVOLUTION", "RANGER", "SHADOWOFALMIA", "GUARDIANSIGNS"]:
                        if ds_trade:
                            cart1.transfer(cart2, "CONNECT")
                    elif game2.gen == 5 and game2.core:
                        if ds_trade:
                            cart1.transfer(cart2, "POKEMON")
                elif game1.gen == 5 and game1.core:
                    if game2.gen == 5 and game2.core:
                        if ds_trade:
                            cart1.transfer(cart2, "TRADE")
                elif game1.name == "DREAMWORLD":
                    if game2.gen == 5 and game2.core:
                        cart1.transfer(cart2, "POKEMON")
                        cart1.transfer(cart2, "ITEMS")
                elif game1.name == "DREAMRADAR":
                    if game2.gen == 4 and game2.core:
                        cart1.transfer(cart2, "CONNECT")
                    if game2.name in ["BLACK2", "WHITE2"]:
                        cart1.transfer(cart2, "POKEMON")

    def calc_dexes(self):
        for cart in self.cartridges:
            cart.init_uniques()
        for cart in self.cartridges:
            cart.simplify_rules()

        if len(self.cartridges) == 1:
            self.main_cartridge.all_safe_paths()
            if self.main_cartridge.handle_special(self):
                self.main_cartridge.all_safe_paths()
        else:
            self.try_cart_paths()
            if any(cart.handle_special(self) for cart in self.cartridges):
                self.try_cart_paths()

        paths = self.main_cartridge.try_paths()
        dexes = pd.DataFrame([[p[1] for p in path] for path in paths], columns=self.pokemon_list.index)

        varying = [col for col in dexes.columns if not all(dexes[col]) and not all(~dexes[col])]
        groups = []
        pokemon_by_group_idx = {}
        for pokemon in varying:
            found_group = False
            for idx, group in enumerate(groups):
                if dexes[[pokemon, group[0]]].corr().round(8)[pokemon][group[0]] == 0:
                    continue
                found_group = True
                pokemon_by_group_idx[pokemon] = idx
                group.append(pokemon)
                break
            if not found_group:
                pokemon_by_group_idx[pokemon] = len(groups)
                groups.append([pokemon])
        
        group_combos = []
        for idx, pokemon in enumerate(groups):
            combos = set()
            for _, row in dexes[pokemon].iterrows():
                combos.add(tuple(row))
            subsets = set()
            for combo in combos:
                if is_subset(combo, combos):
                    subsets.add(combo)
            combos = combos - subsets
            group_combos.append(combos)

        group_min_idxs = [min([self.pokemon_list.loc[p, 'index'] for p in group]) for group in groups]

        present = {}
        missing = {}
        idx2gidx = {}
        idx2pokemon = {}

        for pokemon, row in self.pokemon_list.iterrows():
            idx = row['index']
            idx2pokemon[idx] = pokemon
            if all(dexes[pokemon]):
                present[idx] = [idx]
                continue
            elif all(~dexes[pokemon]):
                missing[idx] = [idx]
                continue

            group_idx = pokemon_by_group_idx[pokemon]
            idx2gidx[idx] = group_idx
            if idx == group_min_idxs[group_idx]:
                group = [self.pokemon_list.loc[p, 'index'] for p in groups[group_idx]]
                present_combos = sorted([
                    [idx for idx, present in zip(group, combo) if present]
                    for combo in group_combos[group_idx]
                ], key=lambda group: (-len(group), group))
                missing_combos = sorted([
                    [idx for idx, present in zip(group, combo) if not present]
                    for combo in group_combos[group_idx]
                ], key=lambda group: (-len(group), group))
                for line in zip_longest(*present_combos):
                    present[min([l for l in line if l is not None])] = list(line)
                for line in zip_longest(*missing_combos):
                    missing[min([l for l in line if l is not None])] = list(line)
                
        return Result(present, missing, idx2gidx, idx2pokemon)


    def try_cart_paths(self):
        state_copies = {cart.id: None for cart in self.cartridges}
        while True:
            updates = False
            for cart in self.cartridges:
                cart_state = (cart.pokemon, cart.items, cart.choices)
                if state_copies[cart.id] != cart_state:
                    updates = True
                    state_copies[cart.id] = deepcopy(cart_state)
                    cart.all_safe_paths()
            if not updates:
                break
        state_copies = {cart.id: None for cart in self.cartridges}
        while True:
            updates = False
            for cart in self.cartridges:
                cart_state = (cart.pokemon, cart.items, cart.choices)
                if state_copies[cart.id] != cart_state:
                    updates = True
                    state_copies[cart.id] = deepcopy(cart_state)
                    cart.try_paths(only_side_effects=True)
            if not updates:
                break

    def obtainable(self):
        groups_printed = set()
        out = []
        for idx, pokemon in enumerate(self.pokemon_list.index):
            if pokemon in self.pokemon_by_group_idx:
                group_idx = self.pokemon_by_group_idx[pokemon]
                if group_idx in groups_printed:
                    continue
                out += self._group_out(group_idx, obtainable=True)
                groups_printed.add(group_idx)
            elif all(self.dexes[pokemon]):
                out.append(f"{(idx + 1):03}. {pokemon}")
        return out

    def missing(self):
        groups_printed = set()
        out = []
        for idx, pokemon in enumerate(self.pokemon_list.index):
            if pokemon in self.pokemon_by_group_idx:
                group_idx = self.pokemon_by_group_idx[pokemon]
                if group_idx in groups_printed:
                    continue
                out += self._group_out(group_idx, obtainable=False)
                groups_printed.add(group_idx)
            elif all(~self.dexes[pokemon]):
                out.append(f"{(idx + 1):03}. {pokemon}")
        return out
                
    def _group_out(self, group_idx, obtainable):
        out = []
        group = self.groups[group_idx]
        combos = self.group_combos[group_idx]
        clusters = []
        if obtainable:
            for combo in combos:
                clusters.append([group[idx] for idx in range(len(group)) if combo[idx]])
        else:
            for combo in combos:
                clusters.append([group[idx] for idx in range(len(group)) if not combo[idx]])
        if all([not cluster for cluster in clusters]):
            return []

        out_clusters = []
        for cluster in clusters:
            out_cluster = sorted([f"{(self.pokemon_list.loc[p, 'index']):03}. {p}" for p in cluster])
            max_len = max([len(p) for p in out_cluster])
            out_cluster = [p + " "*(max_len - len(p)) for p in out_cluster]
            out_clusters.append(out_cluster)
        out_clusters = sorted(out_clusters)

        max_choice_len = max([len(cluster) for cluster in out_clusters])
        rows = []
        for idx in range(max_choice_len):
            row = []
            for cluster in out_clusters:
                if idx < len(cluster):
                    row.append(cluster[idx])
                else:
                    row.append("-" + " "*(len(cluster[0]) - 1))
            rows.append(" / ".join(row))
        return rows


class Rules(dict):
    def __init__(self, rules):
        tick = 0
        mapping = {}
        for rule in rules:
            if rule.name in mapping:
                tick += 1
                rule.name += f"_{tick}"
            mapping[rule.name] = rule
        super().__init__(mapping)

    def by_output(self):
        outputs = {}
        for rule_name, rule in self.items():
            for o in rule.output:
                if o not in outputs:
                    outputs[o] = set()
                outputs[o].add(rule_name)
        return outputs

class MultiResult():
    def __init__(self, results, pokemon2idx, match_version_exclusives):
        self.results = results
        self.max_idx = max([r.max_idx for r in results])
        self.pokemon2idx = pokemon2idx
        for result in self.results:
            if result.max_idx == self.max_idx:
                self.idx2pokemon = result.idx2pokemon
                break
        self.all_present = set()
        self.all_missing = set()
        for idx in range(1, self.max_idx + 1):
            line = self.results[0].line(idx)
            matches = True
            for result in self.results[1:]:
                if result.line(idx) != line:
                    matches = False
                    break
            if matches:
                if line is None:
                    self.all_missing.add(idx)
                else:
                    self.all_present.add(idx)

        
        self.version_exclusives = {}
        if match_version_exclusives:
            ve_table = pd.read_csv(Path("data") / "version_exclusives.csv")
            for _, row in ve_table.iterrows():
                idx1 = self.pokemon2idx.get(row.SPECIES1)
                idx2 = self.pokemon2idx.get(row.SPECIES2)
                if idx1 is None or idx2 is None:
                    continue
                present_patterns = set([(idx1 in r.present, idx2 in r.present) for r in self.results])
                if (True, False) in present_patterns and (False, True) in present_patterns and (True, True) not in present_patterns:
                    if idx1 not in self.version_exclusives:
                        self.version_exclusives[idx1] = set()
                    if idx2 not in self.version_exclusives:
                        self.version_exclusives[idx2] = set()
                    self.version_exclusives[idx1].add(idx2)
                    self.version_exclusives[idx2].add(idx1)
                

    def full_group(self, idx):
        processed = set()
        to_process = set([idx])
        while to_process:
            idx = to_process.pop()
            processed.add(idx)
            ve_opposites = self.version_exclusives.get(idx)
            if ve_opposites is not None:
                for opp_idx in ve_opposites:
                    if opp_idx not in processed:
                        to_process.add(opp_idx)
                
            for result in self.results:
                gidx = result.idx2gidx.get(idx)
                if gidx is None:
                    continue
                for sub_idx in result.gidx2idxs[gidx]:
                    if sub_idx not in processed:
                        to_process.add(sub_idx)
        return processed


    def _group_lines(self, idxs, obtainable=True):
        lines = [r._lines(r.present if obtainable else r.missing, idxs) for r in self.results]
        return list(zip_longest(*lines, fillvalue=""))

    def print(self, obtainable=True, skip_identical=True):
        vline = ["-" for _ in self.results]
        handled_idxs = set()
        lines = []
        for idx in range(1, self.max_idx + 1):
            if idx in handled_idxs:
                continue
            if idx in self.all_missing and skip_identical:
                continue
            if idx in self.all_present and skip_identical:
                continue
            
            sub_idxs = self.full_group(idx)
            if skip_identical:
                sub_idxs = sub_idxs.difference(self.all_missing).difference(self.all_present)
            sub_idxs = sorted(sub_idxs)
            group_lines = self._group_lines(sub_idxs, obtainable)
            if len(group_lines) > 1:
                if lines and lines[-1] != vline:
                    lines.append(vline)
                lines += group_lines
                lines.append(vline)
            else:
                lines += group_lines
            handled_idxs = handled_idxs.union(sub_idxs)
        if lines and lines[-1] != vline:
            lines.append(vline)
        lines.append([str(r.count()) for r in self.results])
        self._print_lines(lines)

    def print_compact(self, obtainable=True):
        if obtainable:
            lines = [r._lines(r.present, r.order()) for r in self.results]
        else:
            lines = [r._lines(r.missing, r.order()) for r in self.results]
        lines = list(zip_longest(*lines, fillvalue=''))
        lines.append(['-' for _ in self.results])
        lines.append([str(r.count()) for r in self.results])
        self._print_lines(lines)

    def print_all_present(self):
        r = self.results[0]
        print("\n".join(self.results[0]._lines(self.results[0].present, sorted(self.all_present))))

    def _print_lines(self, lines):
        max_lens = [max(*[len(l[j]) for l in lines]) for j in range(len(lines[0]))]
        lines = [[l[j] + ("-" if l[j] == "-" else " ")*(max_lens[j] - len(l[j])) for j in range(len(l))] for l in lines]
        lines = [('-|-' if l[0].startswith('--') else ' | ').join(l) for l in lines]
        print("\n".join(lines))
        


class Result():
    def __init__(self, present, missing, idx2gidx, idx2pokemon):
        self.present = present
        self.missing = missing
        self.idx2gidx = idx2gidx
        self.idx2pokemon = idx2pokemon
        self.max_idx = max(idx2pokemon.keys())
        self.gidx2idxs = {}
        for idx, gidx in self.idx2gidx.items():
            if gidx not in self.gidx2idxs:
                self.gidx2idxs[gidx] = set()
            self.gidx2idxs[gidx].add(idx)

    def obtainable_line(self, idx):
        if idx not in self.present:
            return None
        return " / ".join([self.entry(i) for i in self.present[idx]])

    def entry(self, idx):
        if idx is None:
            return "-"
        return f"{idx:04}. {self.idx2pokemon[idx]}"

    def line(self, idx):
        sub_idxs = self.present.get(idx)
        if not sub_idxs:
            return None
        return " / ".join([self.entry(sub_idx) for sub_idx in sub_idxs])

    def _lines(self, source, idxs):
        group_widths = {}
        for idx in idxs:
            gidx = self.idx2gidx.get(idx)
            if gidx is None:
                continue
            sub_idxs = source.get(idx)
            if not sub_idxs:
                continue
            lengths = [len(self.entry(sub_idx)) if sub_idx else 0 for sub_idx in sub_idxs]

            if gidx in group_widths:
                for j in range(len(lengths)):
                    group_widths[gidx][j] = max(group_widths[gidx][j], lengths[j])
            else:
                group_widths[gidx] = lengths

        out = []
        for idx in idxs:
            gidx = self.idx2gidx.get(idx)
            if gidx is None:
                if idx in source:
                    out.append(self.entry(idx))
                continue
            sub_idxs = source.get(idx)
            if not sub_idxs:
                continue
            line = zip([self.entry(sub_idx) for sub_idx in sub_idxs], group_widths[gidx])
            line = [text + " "*(w - len(text)) for text, w in line]
            out.append(" / ".join(line))

        return out

    def order(self):
        handled_groups = set()
        order = []
        for idx in range(1, self.max_idx + 1):
            gidx = self.idx2gidx.get(idx)
            if gidx is None:
                order.append(idx)
                continue
            if gidx in handled_groups:
                continue
            order += sorted(self.gidx2idxs[gidx])
            handled_groups.add(gidx)
        return order

    def print_obtainable(self):
        print("\n".join(self._lines(self.present, self.order())))

    def print_missing(self):
        print("\n".join(self._lines(self.missing, self.order())))

    def count(self):
        return len(self._lines(self.present, self.order()))



@dataclass
class PokemonTransferRule():
    cart: Cartridge
    in_species: str
    out_species: str
    item: Optional[str] = None

@dataclass
class TradePairRule():
    cart: Cartridge
    in1: str
    out1: str
    in2: str
    out2: str

@dataclass
class ItemTransferRule():
    cart: Cartridge
    in_item: Optional[str]
    in_choice: Optional[str]
    out_item: str
 

class TransferRule():
    def __init__(self, cart, required, output, dex=None):
        self.cart = cart
        self.required = required
        self.output = output
        self.dex = dex or []

    def __repr__(self):
        return '/'.join(self.required) + ' -> ' + '/'.join(self.output) + ' (' + self.cart.id + ')'

    def copy(self):
        return TransferRule(self.name, self.cart, self.required.copy(), self.output.copy(), self.repeats)

class Rule():
    def __init__(self, name, consumed, required, output, repeats, keep_props=False, dex=None):
        self.name = name
        self.consumed = consumed
        self.required = required
        self.output = output
        self.repeats = repeats
        self.keep_props = keep_props
        self.dex = dex or []

    def __repr__(self):
        return '/'.join([str(c) for c in self.consumed]) + ' -> ' + '/'.join([str(o) for o in self.output])

    def __copy__(self):
        return Rule(self.name, self.consumed, self.required, self.output, self.repeats)

    def __deepcopy__(self, memo):
        return Rule(
            self.name,
            tuple(self.consumed),
            tuple(self.required),
            tuple(self.output),
            self.repeats,
            self.keep_props,
            None if self.dex is None else tuple(self.dex)
        )

    @staticmethod
    def breed(parent, child, ditto=False, item=None):
        required = [parent]
        if ditto:
            required.append(PokemonReq("Ditto"))
        if item:
            required.append(item)
        return Rule(f"breed:{parent}->{child}", [], required, [child], math.inf)

    @staticmethod
    def buy(pokemon_or_item, exchange):
        consumed = exchange if exchange else []
        return Rule(f"buy:{pokemon_or_item}", consumed, [], [pokemon_or_item], math.inf)

    @staticmethod
    def choice(name, requirements, repeats=1):
        requirements = requirements or []
        gc = GameChoice(name)
        return Rule(name, [], requirements, [gc], repeats)

    @staticmethod
    def choose(name, pokemon_or_items):
        gc = GameChoice(name)
        return Rule(f"{name}->{pokemon_or_items}", [gc], [], pokemon_or_items, math.inf)

    @staticmethod
    def evolve(pre, post, item, other=None):
        consumed = [pre]
        if item:
            consumed.append(item)
        reqs = []
        if other is not None:
            reqs.append(other)
        return Rule(f"evolve:{pre}->{post}", consumed, reqs, post, math.inf, keep_props=True)

    @staticmethod
    def ingame_trade(give, get, item, evolution, choice=None, give_everstone=False, reqs=None):
        if evolution is None:
            output = [get]
            dex = None
        else:
            output = [evolution]
            dex = [get]
        if item:
            output.append(item)
        consumed = []
        if give:
            consumed.append(give)
        if choice is not None:
            consumed.append(GameChoice(choice))
        if give_everstone:
            consumed.append(Item("Everstone"))
        reqs = reqs or []
        return Rule(f"trade:{give}->{get}", consumed, reqs, output, 1, dex=dex)

    @staticmethod
    def fossil(fossil, pokemon, reqs=[]):
        return Rule(f"fossil:{fossil}->{pokemon}", [fossil], reqs, [pokemon], math.inf)

    @staticmethod
    def gift(pokemon_or_items, requirements):
        requirements = requirements or []
        return Rule(f"gift:{pokemon_or_items}", [], requirements, pokemon_or_items, 1)

    @staticmethod
    def pickup(item, pokemon):
        return Rule(f"pickup:{pokemon}->{item}", [], [pokemon], [item], math.inf)

    @staticmethod
    def wild(pokemon_or_item, consumed, required, repeats, item=None):
        output = [pokemon_or_item]
        if item:
            output.append(item)
        consumed = consumed or []
        required = required or []
        return Rule(f"wild:{pokemon_or_item}", consumed, required, output, repeats)


def parse_game(games, game_list):
    if games == "CORE":
        return [g for g in game_list if GAMES[g].core]
    games = games.split(',')
    bad = [game for game in games if game not in game_list]
    if bad:
        raise ValueError(f"Invalid game(s) {bad}")
    return games


def is_subset(tup, tup_set):
    '''
    tup is a tuple of Booleans, and tup_set is a set of tuples of Booleans.

    Return True if there exists a tuple in tup_set whose True values are a strict superset of the True values in tup.
    '''
    tup_count = sum(tup)
    for tup2 in tup_set:
        if sum(tup2) <= tup_count:
            continue
        if all([(not tup[idx]) or (tup[idx] and tup2[idx]) for idx in range(len(tup))]):
            return True
    return False

def main(args):
    pd.set_option('future.no_silent_downcasting', True)
    all_games = [[]]
    all_hardware = [{}]
    num_collections = 1
    hardware_with_software = {}
    for item in args.game_or_hardware:
        if item == '.':
            num_collections += 1
            all_games.append([])
            all_hardware.append({})
        if item in GAMES:
            if "SOFTWARE" in GAMES[item].props:
                raise ValueError(f"Game {item} is software only")
            all_games[-1].append((GAMES[item], None))
        else:
            h, _, gs = item.partition('[')
            all_hardware[-1][h] = all_hardware[-1].get(item, 0) + 1
            if gs:
                idx = hardware_with_software.get(h, 0) + 1
                hardware_with_software[h] = idx
                for game_name in gs[:-1].split(','):
                    if game_name not in GAMES:
                        raise ValueError(f"Unrecognized game {game_name}")
                    game = GAMES[game_name]
                    if game.console != h:
                        raise ValueError(f"Game {game_name} goes with {game.console}, not {h}")
                    if h not in CONSOLES_WITH_SOFTWARE:
                        raise ValueError(f"Console {h} doesn't support software games")
                    all_games[-1].append((game, f"{h}_{idx}"))

    if num_collections == 1:
        games = all_games[0]
        hardware = all_hardware[0]
        collection, result = calc_dexes(games, hardware)

        result.print_obtainable()
        print("\n---\n")
        result.print_missing()
        print("\n---\n")
        print(f"TOTAL: {result.count()}")
        cart = collection.main_cartridge
        import pdb; pdb.set_trace()
    elif num_collections == 2:
        raise ValueError("Games/hardware before the first '.' are universal, so there should be 0 or 2+ instances of '.'")
    else:
        collections = []
        results = []
        for idx in range(1, num_collections):
            games = all_games[idx] + all_games[0]
            hardware = all_hardware[0].copy()
            for k, v in all_hardware[idx].items():
                hardware[k] = hardware.get(k, 0) + v
            c, r = calc_dexes(games, hardware)
            collections.append(c)
            results.append(r)
        pokemon2idx = {}
        for collection in collections:
            for pokemon, row in collection.pokemon_list.iterrows():
                idx = row['index']
                pokemon2idx[pokemon] = idx
        result = MultiResult(results, pokemon2idx, args.version_exclusive)
        if args.all_present:
            result.print_all_present()
        elif args.compact:
            result.print_compact()
        else:
            result.print(obtainable=(not args.missing), skip_identical=(not args.full))


def calc_dexes(games, hardware):
    generations = {gen: Generation(gen) for gen in set([g.gen for g, _ in games])}
    cartridges = [Cartridge(g, generations[g.gen], console=c) for g, c in games]
    collection = Collection(cartridges, hardware)

    for cart in cartridges:
        cart.init_rules()

    return collection, collection.calc_dexes()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_or_hardware', nargs='+', default=[])
    parser.add_argument('--full', '-f', action='store_true')
    parser.add_argument('--all-present', '-a', action='store_true')
    parser.add_argument('--compact', '-c', action='store_true')
    parser.add_argument('--version-exclusive', '-v', action='store_true')
    parser.add_argument('--missing', '-m', action='store_true')
    args = parser.parse_args()
    main(args)
