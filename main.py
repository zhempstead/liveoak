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

from game import Game, Cartridge, Hardware, GAMES, GAMEIDS, GAMEINPUTS
from result import MultiResult, Result

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

        self.dex = {species: False for species in self.generation.pokemon_list.index}

        self.has_gender = self.game.gen >= 2 and "NO_BREED" not in self.game.props

        # Initialized later - indicates what other game states this one can communicate with
        self.transfers = {
            "POKEMON": [], # Transfer Pokemon w/o trading
            "TRADE": [], # Trade Pokemon (possible w/held items)
            "ITEMS": [], # Transfer items
            "MYSTERYGIFT": [], # Mystery gift
            "RECORDMIX": [], # Gen 3 record mixing
            "CONNECT": [], # Miscellaneous
        }

        self.env = set()
        if self.generation.environment is not None:
            for _, row in self.generation.environment.iterrows():
                if self.game.match(row.GAME):
                    self.env.add(row.ENVIRONMENT)

        self.pokemon = {}
        self.items = {}
        self.choices = {}

        self.path_child = None
        self.unique_pokemon = set()

    def copy(self):
        memo = {}
        gs = GameSave(self.cartridge, self.generation)
        for species, pokemon in self.pokemon.items():
            gs.pokemon[species] = pokemon.copy()
        gs.items = self.items.copy()
        gs.choices = self.choices.copy()
        gs.rules = deepcopy(self.rules, memo)
        gs.transfer_rules = {}
        for entry, rules in self.transfer_rules.items():
            gs.transfer_rules[entry] = rules.copy()
        gs.dex = self.dex.copy()
        gs.transfers = self.transfers
        return gs

    def transfer(self, other_cartridge, category):
        '''Register another cartridge as able to receive certain communications'''
        self.transfers[category].append(other_cartridge)

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
                if not self.game.match(row.GAME):
                    continue

                exchange, _ = self.parse_input(row.get("EXCHANGE"))
                required, valid = self.parse_input(row.get("REQUIRED"))
                if not valid:
                    continue
                for o in self.parse_output(row.POKEMON_OR_ITEM, self.has_gender):
                    if len(o) != 1:
                        raise ValueError(f"Invalid buy entry {o}")
                    rules.append(Rule.buy(o[0], exchange, required))

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
                if not self.game.match(row.GAME):
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
                if not self.game.match(row.GAME):
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
                if not self.game.match(row.GAME):
                    continue
                for pokemon in self.generation.pickup_pokemon["SPECIES"]:
                    req = self.parse_pokemon_input(pokemon, forbidden={"NOPICKUP"})
                    rules.append(Rule.pickup(Item(row.ITEM), req))

        if self.generation.trade is not None:
            for idx, row in self.generation.trade.iterrows():
                if not self.game.match(row.GAME):
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
                    items = [Item(i) for i in wild_items.get(pokemon_or_item, [])] or [None]
                    p_or_is = self.parse_output(pokemon_or_item, self.has_gender)
                    for idx, (consumed, reqs, count) in enumerate(self.parse_wild_entry(row[gamecol])):
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

        for cart in self.transfers["RECORDMIX"]:
            transfer_rules.append(ItemTransferRule(cart, None, f"RM_Eon Ticket", "Eon Ticket"))

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
                if not any([gs.cartridge != self.cartridge and gs.game.match(e) for gs in self.transfers["CONNECT"] + self.transfers["RECORDMIX"]]): 
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
                    gs_sets = self.unique_pokemon_transfers(entry.species)
                    if len(gs_sets) > 1:
                        raise ValueError("Not handled")
                    gs_set = gs_sets[0]
                    global all_gamestates
                    for cartridge in gameset:
                        if cartridge == self.cartridge:
                            continue
                        all_gamestates[cartridge].acquire(entry, 1, communicate=False)
                    return

            trules = self.transfer_rules.get(pokemon.species, [])
            for trule in trules:
                if not isinstance(trule, PokemonTransferRule):
                    continue
                if trule.item is not None and self.items.get(trule.item) == 0:
                    continue
                transfer_pokemon = replace(pokemon, species=trule.out_species)
                trule.game_save.acquire(transfer_pokemon, count=send_count)
            for trule in trules:
                if not isinstance(trule, TradePairRule):
                    continue
                if send_count != math.inf:
                    raise ValueError("Not handled")
                matches2 = self.get_matches(PokemonReq(trule.in2, forbidden=frozenset(["NOTRANSFER"])))
                if matches2:
                    transfer1 = replace(pokemon, species=trule.out1)
                    trule.game_save.acquire(transfer1, count=send_count)
                for in2, count in matches2:
                    if count != math.inf:
                        raise ValueError("Not handled")
                    transfer2 = replace(in2, species=trule.out2)
                    trule.game_save.acquire(transfer2, count=send_count)

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
                        trule.game_save.acquire(out_pokemon, count=math.inf)
                elif isinstance(trule, ItemTransferRule):
                    if trule.in_choice is not None and self.choices.get(trule.in_choice, 0) == 0:
                        continue
                    trule.game_save.acquire(Item(trule.out_item), count=math.inf)

        elif isinstance(entry, GameChoice):
            choice = entry.choice
            self.choices[choice] = self.choices.get(choice, 0) + count

            if not communicate:
                return

            for trule in self.transfer_rules.get(choice, []):
                if trule.in_item is not None and self.items.get(trule.in_item, 0) == 0:
                    continue
                trule.game_save.acquire(Item(trule.out_item), count=math.inf)
    
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

    def try_paths(self, subset=None, only_side_effects=False):
        component_dexsets = []

        for G in self.rule_graph_components(subset):
            local_dexsets = set()
            ruleset = set(G.nodes).intersection(self.rules.keys())
            for rule_name in ruleset:
                rule = self.rules[rule_name]
                if not self.check_rule_possible(rule):
                    continue
                save_copy = self.copy()
                self.path_child = save_copy
                save_copy.apply_rule_if_possible(rule_name)
                save_copy.all_safe_paths()
                if only_side_effects:
                    save_copy.try_paths(G.nodes, only_side_effects)
                else:
                    local_dexsets = local_dexsets.union(save_copy.try_paths(G.nodes, only_side_effects))
            if not only_side_effects:
                component_dexsets.append(local_dexsets)

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

    def rule_graph_components(self, subset=None):
        start_end = set(["START", "END"])
        G = self.rule_graph()
        if subset is not None:
            G.remove_nodes_from([n for n in G if n not in subset and n not in start_end])
        Gsub = G.subgraph([n for n in G.nodes if n not in start_end])
        for nodes in nx.connected_components(Gsub.to_undirected()):
            Gcomp = G.copy()
            Gcomp.remove_nodes_from([n for n in Gcomp if n not in nodes and n not in start_end])
            yield Gcomp

        
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
            raise ValueError(f"Invalid entry {entry}")

    def unique_pokemon_transfers(self, species, already_visited=None):
        if already_visited is None:
            already_visited = frozenset()
        visited = already_visited.union(set([self.cartridge]))
        possibilities = set()
        for gs in set(self.transfers["POKEMON"] + self.transfers["TRADE"]):
            if gs.cartridge in already_visited:
                continue
            if species not in gs.generation.tradeable_pokemon:
                continue
            for poss in gs.unique_pokemon_transfers(species, visited):
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

    def __str__(self):
        return str(self.cartridge)
                    

class Collection():
    def __init__(self, cartridges, hardware):
        if len(cartridges) == 0:
            raise ValueError("No games!")
        self.main_cartridge = cartridges[0]
        if "NO_DEX" in self.main_cartridge.game.props:
            raise ValueError(f"Can't use {self.main_cartridge.game.name} as main game")

        self.cartridges = cartridges
        self.hardware = hardware

        self.hw_ports = {}
        self.hw_plugs = {}
        self.hw_wireless = {}
        self.hw_carts = {}
        self.sw_carts = {}
        for hw in self.hardware:
            for idx, port in enumerate(hw.model.ports):
                for flow, port_types in port.items():
                    if flow not in self.hw_ports:
                        self.hw_ports[flow] = {}
                    for port_type in port_types:
                        if port_type not in self.hw_ports[flow]:
                            self.hw_ports[flow][port_type] = set()
                        self.hw_ports[flow][port_type].add((hw, idx))
            for idx, plug in enumerate(hw.model.plugs):
                for flow, plug_types in plug.items():
                    if flow not in self.hw_plugs:
                        self.hw_plugs[flow] = {}
                    for plug_type in plug_types:
                        if plug_type not in self.hw_plugs[flow]:
                            self.hw_plugs[flow][plug_type] = set()
                        self.hw_plugs[flow][plug_type].add((hw, idx))
            for flow in hw.model.wireless:
                if flow not in self.hw_wireless:
                    self.hw_wireless[flow] = set()
                self.hw_wireless[flow].add(hw)
        for cart in self.cartridges:
            if cart.console is None:
                if cart.game.console not in self.hw_carts:
                    self.hw_carts[cart.game.console] = set()
                self.hw_carts[cart.game.console].add(cart)
            else:
                if cart.console not in self.sw_carts:
                    self.sw_carts[cart.console] = set()
                self.sw_carts[cart.console].add(cart)

        self._connected_cache = {}
        for cart in self.cartridges:
            flow = cart.game.default_flow()
            can_play, connections = cart.connections(self, flow)
            if not can_play:
                raise ValueError(f"Game {cart} cannot be played with specified hardware.")
            self._connected_cache[cart] = connections

        generations = {gen: Generation(gen) for gen in set(c.game.gen for c in self.cartridges)} 
        self.game_states = {c: GameSave(c, generations[c.game.gen]) for c in self.cartridges}
        self._init_interactions()

        if any([gs.has_gender for gs in self.game_states.values()]):
            for gs in self.game_states.values():
                gs.has_gender = True
        self.pokemon_list = self.game_states[self.main_cartridge].generation.pokemon_list

        if len(self.cartridges) > 1:
            G = nx.DiGraph()
            for cart, gs in self.game_states.items():
                G.add_node(cart)
                for kind, gs2s in gs.transfers.items():
                    for gs2 in gs2s:
                        cart2 = gs2.cartridge
                        if kind == "CONNECT":
                            G.add_edge(cart2, cart)
                        else:
                            G.add_edge(cart, cart2)
            for cart in self.cartridges:
                if cart == self.main_cartridge:
                    continue
                if not nx.has_path(G, cart, self.main_cartridge):
                    raise ValueError(f"Game {cart.game} cannot interact with main game {self.main_cartridge.game} (possibly due to specified hardware)")

    def connected(self, cart1, cart2, flow=None):
        if flow is None:
            flow1 = cart1.game.default_flow()
            flow2 = cart2.game.default_flow()
            if flow1 == flow2:
                flow = flow1
            else:
                raise ValueError(f"Need to specify flow for {cart1} and {cart2}")
        if not cart1.game.core and cart2.game.core:
            cart1, cart2 = cart2, cart1
        if "NO_DEX" in cart1.game.props and "NO_DEX" not in cart2.game.props:
            cart1, cart2 = cart2, cart1
        if flow not in self._connected_cache:
            self._connected_cache[flow] = {}
        if cart1 not in self._connected_cache[flow] and cart2 not in self._connected_cache[flow]:
            _, self._connected_cache[flow][cart1] = cart1.connections(self, flow)

        if cart1 in self._connected_cache[flow]:
            return cart2 in self._connected_cache[flow][cart1]
        else:
            return cart1 in self._connected_cache[flow][cart2]


    def _init_interactions(self):
        for cart1, gs1 in self.game_states.items():
            game1 = cart1.game
            for cart2, gs2 in self.game_states.items():
                game2 = cart2.game
                if cart1 == cart2:
                    continue

                if game1.gen == 1 and game1.core:
                    if game2.gen == 1 and game2.core:
                        if (game1.language == "Japanese") == (game2.language == "Japanese") and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "TRADE")
                    elif game2.name in {"STADIUM_JPN", "STADIUM"}:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "POKEMON")
                    elif game2.gen == 2 and game2.core:
                        if (game1.language == "Japanese") == (game2.language == "Japanese") and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "TRADE")
                    elif game2.name == "STADIUM2":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "POKEMON")
                            # STADIUM2 is the only way to transfer items between gen 1 games
                            for cart3, gs3 in self.game_states.items():
                                game3 = cart3.game
                                if game3.gen == 1 and game3.core:
                                    if game1.language_match(game3) and self.connected(cart2, cart3):
                                        gs1.transfer(gs3, "ITEMS")
                elif game1.name in {"STADIUM_JPN", "STADIUM"}:
                    if game2.gen == 1 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "POKEMON")
                elif game1.gen == 2 and game1.core:
                    if game2.gen == 1 and game2.core:
                        if (game1.language == "Japanese") == (game2.language == "Japanese") and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "TRADE")
                            # Pokemon holding items can be traded to gen 1. Gen 1 can't access the
                            # items, but they will be preserved if the Pokemon is later traded back
                            # to gen 2, including the same cartridge.
                            for cart3, gs3 in self.game_states.items():
                                game3 = cart3.game
                                if game3.gen == 2 and game3.core:
                                    if (game1.language == "Japanese") == (game3.language == "Japanese") and self.connected(cart2, cart3):
                                        gs1.transfer(gs3, "ITEMS")
                    elif game2.gen == 2 and game2.core:
                        if (game1.language == "Japanese") == (game2.language == "Japanese") and self.connected(cart1, cart2):
                                gs1.transfer(gs2, "TRADE")
                        if game1.language_match(game2) and self.connected(cart1, cart2, "GBC_ir"):
                            gs1.transfer(gs2, "MYSTERYGIFT")
                    elif game2.name == "STADIUM2":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "POKEMON")
                            gs1.transfer(gs2, "ITEMS")
                            gs1.transfer(gs1, "MYSTERYGIFT") # self-transfer
                elif game1.name == "STADIUM2":
                    if game2.gen == 1 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "POKEMON")
                    elif game2.gen == 2 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "POKEMON")
                            gs1.transfer(gs2, "ITEMS")
                elif game1.gen == 3 and game1.core:
                    if game2.gen == 3 and game2.core:
                        if self.connected(cart1, cart2) or (
                                "GBA_WIRELESS" in game1.props and
                                "GBA_WIRELESS" in game2.props and
                                self.connected(cart1, cart2, "GBAw")
                        ):
                            gs1.transfer(gs2, "TRADE")
                            RSE = {"Ruby", "Sapphire", "Emerald"}
                            if game1.name in RSE and game2.name in RSE:
                                if game1.language == game2.language or (game1.name == "EMERALD" and game2.name == "EMERALD"):
                                    gs1.transfer(gs2, "RECORDMIX")
                    elif game2.name == "BOX":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "POKEMON")
                            gs1.transfer(gs2, "ITEMS")
                    elif game2.name in ["COLOSSEUM", "XD"]:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "TRADE")
                    elif game2.name == "BONUSDISC":
                        if game1.name in ["RUBY", "SAPPHIRE"] and game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "CONNECT")
                    elif game2.name == "BONUSDISC_JPN":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "CONNECT")
                    elif game2.name == "CHANNEL" and game2.region == "EUR":
                        if game1.name in ["RUBY", "SAPPHIRE"] and game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "CONNECT")
                    elif game2.name == "EONTICKET":
                        if game1.name in ["RUBY", "SAPPHIRE"] and game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "CONNECT")
                    elif game2.gen == 4 and game2.core:
                        if (game1.language_match(game2) or game2.language == "Korean") and self.connected(cart1, cart2, "DS"):
                            # Pal Park: language requirement
                            gs1.transfer(gs2, "POKEMON")
                            gs1.transfer(gs2, "ITEMS")
                elif game1.name == "BOX":
                    if game2.gen == 3 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "POKEMON")
                            gs1.transfer(gs2, "ITEMS")
                elif game1.name == "COLOSSEUM":
                    if game2.gen == 3 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "TRADE")
                    elif game2.name == "BONUSDISC_JPN":
                        # The Japanese bonus disc reads and edits the Colosseum save file, so they
                        # have to be compatible with the same GCN.
                        if game1.language_match(game2) and game1.region_match(game2):
                            gs1.transfer(gs2, "CONNECT")
                    elif game2.name == "DOUBLEBATTLE":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "CONNECT")
                elif game1.name == "XD":
                    if game2.gen == 3 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "TRADE")
                elif game1.gen == 4 and game1.core:
                    if game2.gen == 3 and game2.core:
                        if self.connected(cart1, cart2, "DS"):
                            # Dual-slot mode: no language requirement
                            gs1.transfer(gs2, "CONNECT")
                    if game2.gen == 4 and game2.core:
                        if (game1.language == "Korean") == (game2.language == "Korean") and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "TRADE")
                    elif game2.name == "POKEWALKER":
                        if game1.name in ["HEARTGOLD", "SOULSILVER"]:
                            gs1.transfer(gs2, "CONNECT")
                    elif game2.name == "RANCH":
                        if game1.language_match(game2) and self.connected(cart1, cart2) and (
                                game1.name in ["DIAMOND", "PEARL"] or \
                                (game2.language == "Japanese" and game1.name == "PLATINUM")
                        ):
                            gs1.transfer(gs2, "CONNECT")
                    elif game2.name in ["BATTLEREVOLUTION", "RANGER", "SHADOWOFALMIA", "GUARDIANSIGNS"]:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "CONNECT")
                    elif game2.gen == 5 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            gs1.transfer(gs2, "POKEMON")
                elif game1.gen == 5 and game1.core:
                    if game2.gen == 5 and game2.core:
                        if self.connected(cart1, cart2):
                            gs1.transfer(gs2, "TRADE")
                elif game1.name == "DREAMWORLD":
                    if game2.gen == 5 and game2.core:
                        if game1.region_match(game2):
                            gs1.transfer(gs2, "POKEMON")
                            gs1.transfer(gs2, "ITEMS")
                elif game1.name == "DREAMRADAR":
                    if game2.gen == 4 and game2.core:
                            gs1.transfer(gs2, "CONNECT")
                    elif game2.name in ["BLACK2", "WHITE2"]:
                        if game1.region_match(game2):
                            gs1.transfer(gs2, "POKEMON")
                            gs1.transfer(gs2, "ITEMS")


    def calc_dexes(self):
        for gs in self.game_states.values():
            gs.init_uniques()
        for gs in self.game_states.values():
            gs.simplify_rules()

        if len(self.cartridges) == 1:
            self.game_states[self.main_cartridge].all_safe_paths()
            if self.game_states[self.main_cartridge].handle_special(self):
                self.game_states[self.main_cartridge].all_safe_paths()
        else:
            self.try_cart_paths()
            if any(gs.handle_special(self) for gs in self.game_states.values()):
                self.try_cart_paths()

        paths = self.game_states[self.main_cartridge].try_paths()
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
        state_copies = {cart: None for cart in self.cartridges}
        while True:
            updates = False
            for cart, gs in self.game_states.items():
                save_state = (gs.pokemon, gs.items, gs.choices)
                if state_copies[cart] != save_state:
                    updates = True
                    state_copies[cart] = deepcopy(save_state)
                    gs.all_safe_paths()
            if not updates:
                break
        state_copies = {cart: None for cart in self.cartridges}
        while True:
            updates = False
            for cart, gs in self.game_states.items():
                save_state = (gs.pokemon, gs.items, gs.choices)
                if state_copies[cart] != save_state:
                    updates = True
                    state_copies[cart] = deepcopy(save_state)
                    gs.try_paths(only_side_effects=True)
            if not updates:
                break

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


@dataclass
class PokemonTransferRule():
    game_save: GameSave
    in_species: str
    out_species: str
    item: Optional[str] = None

@dataclass
class TradePairRule():
    game_save: GameSave
    in1: str
    out1: str
    in2: str
    out2: str

@dataclass
class ItemTransferRule():
    game_save: GameSave
    in_item: Optional[str]
    in_choice: Optional[str]
    out_item: str
 

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
    def buy(pokemon_or_item, exchange, required):
        consumed = exchange if exchange else []
        required = required or []
        return Rule(f"buy:{pokemon_or_item}", consumed, required, [pokemon_or_item], math.inf)

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
                    all_games[-1].append((game, item, hardware))

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
            hardware = all_hardware[idx] + all_hardware[0]
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
    cartridges = [Cartridge(g, cl, c) for g, cl, c in games]
    collection = Collection(cartridges, hardware)
    game_states = collection.game_states.values()

    for gs in game_states:
        gs.init_rules()

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
