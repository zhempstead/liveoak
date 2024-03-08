#!/usr/bin/python

import argparse
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations, product
import math
from pathlib import Path

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pandas as pd

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
]}

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

        self.trade_evos = {}
        self.trade_evos_by_item = {}
        self.gendered_trade_evos = {}
        self.gendered_trade_evos_by_item = {}
        if self.evolve is not None and self.trade is not None:
            for _, row in self.evolve.iterrows():
                if not row.TRADE:
                    continue
                if row.FROM not in self.trade_evos:
                    self.trade_evos[row.FROM] = {}
                self.trade_evos[row.FROM][row.ITEM] = row.TO
                if row.ITEM not in self.trade_evos_by_item:
                    self.trade_evos_by_item[row.ITEM] = {}
                self.trade_evos_by_item[row.ITEM][row.FROM] = row.TO
                suffixes = self.gender_suffixes(row.FROM)
                for suffix in suffixes:
                    frm = row.FROM + suffix
                    to = row.TO + suffix
                    if frm not in self.gendered_trade_evos:
                        self.gendered_trade_evos[frm] = {}
                    self.gendered_trade_evos[frm][row.ITEM] = to
                    if row.ITEM not in self.gendered_trade_evos_by_item:
                        self.gendered_trade_evos_by_item[row.ITEM] = {}
                    self.gendered_trade_evos_by_item[row.ITEM][frm] = to

    def gender_suffixes(self, pokemon):
        '''Return the list of possible gender suffixes.

        'pokemon' may actually be an item, or a Pokemon already with a gender suffix'''
        if pokemon.endswith("_MALE") or pokemon.endswith("_FEMALE"):
            return [""]
        # This is done to handle e.g. gen 1's Pikachu_STARTER, which still needs a gender suffix
        pokemon = pokemon.split('_')[0]
        if pokemon not in self.pokemon_list.index:
            return [""]
        gender = self.pokemon_list.loc[pokemon, "GENDER"]
        if gender == "BOTH":
            return ["_MALE", "_FEMALE"]
        elif gender == "MALE":
            return ["_MALE"]
        elif gender == "FEMALE":
            return ["_FEMALE"]
        elif gender == "UNKNOWN":
            return [""]
        else:
            raise ValueError(f"Unknown species gender label {gender}")

class Cartridge():
    counter = {}

    def __init__(self, game, generation, gameid=None):
        self.game = game
        self.generation = generation
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

        self.state = {}

    def copy(self):
        memo = {}
        c = Cartridge(self.game, self.generation, self.id)
        c.state = deepcopy(self.state, memo)
        c.rules = deepcopy(self.rules, memo)
        c.dex = deepcopy(self.dex, memo)
        c.transfers = {}
        for k, transfers in self.transfers.items():
            c.transfers[k] = transfers.copy()
            if self in transfers:
                c.transfers[k].append(c)
        return c

    def transfer(self, other_cartridge, category):
        '''Register another cartridge as able to receive certain communications'''
        self.transfers[category].append(other_cartridge)
        if category != "CONNECT":
            self.transfers["CONNECT"].append(other_cartridge)

    def self_transfer(self, category):
        self.transfer(self, self, category)

    def init_rules(self):
        if "NO_DEX" in self.game.props:
            self.rules = Rules([])
            return

        if self.has_gender:
            def gendered(p):
                '''Add gender to anything without'''
                return [p + s for s in self.generation.gender_suffixes(p)]
            trade_evos = self.generation.gendered_trade_evos
        else:
            def gendered(p):
                '''Strip gender from anything with gender'''
                if p.endswith("_MALE"):
                    return [p[:-5]]
                elif p.endswith("_FEMALE"):
                    return [p[:-7]]
                return [p]
            trade_evos = self.generation.trade_evos

        rules = []

        if self.generation.breed is not None and "NO_BREED" not in self.game.props:
            for _, row in self.generation.breed.iterrows():
                for p in gendered(row.PARENT):
                    for c in gendered(row.CHILD):
                        rules.append(Rule.breed(p, c, not p.endswith("_FEMALE"), row.get("ITEM")))

        if self.generation.buy is not None:
            for _, row in self.generation.buy.iterrows():
                if self.game.name in row.GAME:
                    for p_or_i in gendered(row.POKEMON_OR_ITEM):
                        rules.append(Rule.buy(p_or_i, row.get("EXCHANGE")))

        if self.generation.evolve is not None and "NO_EVOLVE" not in self.game.props:
            for _, row in self.generation.evolve.iterrows():
                env = row.get("ENVIRONMENT")
                if env and env not in self.env:
                    continue
                if row.get("TRADE"):
                    continue
                to = row.TO.split(',')
                suffixes = self.generation.gender_suffixes(row.FROM) if self.has_gender else ['']
                for suffix in suffixes:
                    to_w_suffix = [t + suffix for t in to]
                    rules.append(Rule.evolve(row.FROM + suffix, to_w_suffix, row.ITEM))
 
        if self.generation.fossil is not None:
            for _, row in self.generation.fossil.iterrows():
                if self.game.name not in row.GAME:
                    continue
                for p in gendered(row.POKEMON):
                    rules.append(Rule.fossil(row.ITEM, p))

        if self.generation.gift is not None:
            for idx, row in self.generation.gift.iterrows():
                if self.game.name not in row.GAME:
                    continue
                choices, reqs = parse_gift_entry(row.POKEMON_OR_ITEM, self.generation, gendered)
                missing_game_req = False
                reqs_with_suffixes = []
                for req in reqs:
                    if req in GAMES:
                        if req not in [c.game.name for c in self.transfers["CONNECT"]]:
                            missing_game_req = True
                            break
                    else:
                        reqg = gendered(req)
                        if len(reqg) == 1:
                            new_req = reqg[0]
                        else:
                            new_req = tuple(reqg)
                        reqs_with_suffixes.append(new_req)
                if missing_game_req:
                    continue

                if len(choices) == 1:
                    rules.append(Rule.gift(choices[0], reqs_with_suffixes))
                else:
                    rules.append(Rule.choice(idx, reqs_with_suffixes))
                    for pokemon_or_items in choices:
                        rules.append(Rule.choose(idx, pokemon_or_items))

        if self.generation.pickup_item is not None:
            for _, row in self.generation.pickup_item.iterrows():
                if self.game.name not in row.GAME:
                    continue
                for pokemon in self.generation.pickup_pokemon["SPECIES"]:
                    for p in gendered(pokemon):
                        rules.append(Rule.pickup(row.ITEM, p))

        if self.generation.trade is not None:
            for idx, row in self.generation.trade.iterrows():
                if self.game.name not in row.GAME:
                    continue
                gives = gendered(row.GIVE)
                gets = gendered(row.GET)
                choice_id = None
                if len(gives) == 1:
                    give = gives[0]
                else:
                    give = tuple(gives)
                if len(gets) > 1:
                    choice_id = f"CHOICE_TRADE_GENDER_{row.GET}_{idx}"
                    rules.append(Rule.choice(choice_id, [], repeats=1))
                for get in gets:
                    item = row.get("ITEM")
                    evolution = None
                    if get in trade_evos:
                        if item and item in trade_evos[get]:
                            evolution = trade_evos[get][item]
                            item = False
                        elif False in trade_evos[get] and (
                                row.get("ITEM") != "Everstone" or
                                # Held item loss glitch - Everstone is lost/ignored
                                self.generation.id == 3 or
                                # Everstone doesn't stop Kadabra from evolving since gen 4
                                (self.generation.id >= 4 and row.GET == "Kadabra")
                        ):
                            evolution = trade_evos[get][False]
                    rules.append(Rule.ingame_trade(give, get, item, evolution, choice_id))


        wild_items = {}
        if self.generation.wild_item is not None:
            for _, row in self.generation.wild_item.iterrows():
                if self.game.name not in row.GAME:
                    continue
                if row.SPECIES not in wild_items:
                    wild_items[row.SPECIES] = []
                wild_items[row.SPECIES].append(row.ITEM)

        if self.generation.wild is not None and self.game.name in self.generation.wild.columns:
            for _, row in self.generation.wild.iterrows():
                pokemon_or_item = row.SPECIES
                items = [None]
                if pokemon_or_item in wild_items:
                    items = wild_items[pokemon_or_item]
                p_or_is = gendered(pokemon_or_item)
                    
                for idx, (count, reqs) in enumerate(parse_wild_entry(row[self.game.name])):
                    if count == 0:
                        continue
                    missing_game_req = False
                    gendered_reqs = []
                    for req in reqs:
                        if req in GAMES:
                            if req not in [c.game.name for c in self.transfers["CONNECT"]]:
                                missing_game_req = True
                                break
                        else:
                            reqg = gendered(req)
                            if len(reqg) == 1:
                                new_req = reqg[0]
                            else:
                                new_req = tuple(reqg)
                            gendered_reqs.append(new_req)
                    if missing_game_req:
                        continue

                    if count != math.inf and len(items) * len(p_or_is) > 1:
                        choice_id = f"CHOICE_WILD_{pokemon_or_item}_{idx}"
                        rules.append(Rule.choice(choice_id, gendered_reqs, repeats=count))
                        for p_or_i in p_or_is:
                            for item in items:
                                out = [p_or_i]
                                if item:
                                    out.append(item)
                                rules.append(Rule.choose(choice_id, out))
                    else:
                        for p_or_i in p_or_is:
                            for item in items:
                                rules.append(Rule.wild(p_or_i, gendered_reqs, count, item))

        self.rules = Rules(rules)

    def acquire(self, pokemon_or_item, count):
        prev_count = self.state.get(pokemon_or_item, 0)
        if prev_count == math.inf:
            return
        self.state[pokemon_or_item] = self.state.get(pokemon_or_item, 0) + count

        if self.has_gender:
            trade_evos = self.generation.gendered_trade_evos
            trade_evos_by_item = self.generation.gendered_trade_evos_by_item
        else:
            trade_evos = self.generation.trade_evos
            trade_evos_by_item = self.generation.trade_evos_by_item

        species, _, _ = pokemon_or_item.partition('_')
        if species in self.dex:
            self.dex[species] = True

        transfer_pokemon_or_item = species
        if pokemon_or_item.endswith("_MALE"):
            transfer_pokemon_or_item += "_MALE"
        elif pokemon_or_item.endswith("_FEMALE"):
            transfer_pokemon_or_item += "_FEMALE"

        if species in self.generation.tradeable_pokemon:
            pokemon = transfer_pokemon_or_item
            for cart in self.transfers["POKEMON"]:
                if pokemon in cart.generation.tradeable_pokemon:
                    cart.acquire(pokemon, count=math.inf)
            for cart in self.transfers["TRADE"]:
                if species in cart.generation.tradeable_pokemon:
                    cart.dex[species] = True
                te = trade_evos.get(pokemon, {})
                simple_trade_evo = te.get(False)
                if simple_trade_evo is None or (
                        self.state.get("Everstone", 0) >= 1 and \
                        # Held item loss glitch - Everstone is lost/ignored
                        self.generation.id != 3 and
                        # Everstone doesn't stop Kadabra from evolving since gen 4
                        (self.generation.id < 4 or species != "Kadabra")
                ):
                    if species in cart.generation.tradeable_pokemon:
                        cart.acquire(pokemon, count=math.inf)
                for item, trade_evo in te.items():
                    if (not item or self.state.get(item, 0) >= 1):
                        trade_evo_species, _, _ = trade_evo.partition('_')
                        if trade_evo_species in cart.generation.tradeable_pokemon:
                            cart.acquire(trade_evo, count=math.inf)

        elif pokemon_or_item in self.generation.tradeable_items:
            item = pokemon_or_item
            for cart in self.transfers["ITEMS"]:
                if item in cart.generation.tradeable_items:
                    cart.acquire(item, count=math.inf)
            for cart in self.transfers["TRADE"]:
                for pokemon, trade_evo in trade_evos_by_item.get(item, {}).items():
                    if self.state.get(pokemon, 0) >= 1 and \
                            pokemon in self.generation.tradeable_pokemon and \
                            pokemon in cart.generation.tradeable_pokemon and \
                            trade_evo in self.generation.tradeable_pokemon and \
                            trade_evo in cart.generation.tradeable_pokemon:
                        cart.acquire(trade_evo, count=math.inf)
                if "NO_TRADE_ITEMS" not in self.game.props and \
                        "NO_TRADE_ITEMS" not in cart.game.props and \
                        item in cart.generation.tradeable_items:
                    cart.acquire(item, count=math.inf)
                # Held item loss glitch - Everstone is lost/ignored in gen 3
                if item == "Everstone" and self.generation.id != 3:
                    for pokemon, trade_evo in trade_evos_by_item.get(False, {}).items():
                        # Everstone doesn't stop Kadabra from evolving since gen 4
                        if pokemon == "Kadabra" and self.generation.id >= 4:
                            continue
                        if self.state.get(pokemon, 0) >= 1 and \
                                pokemon in self.generation.tradeable_pokemon and \
                                pokemon in cart.generation.tradeable_pokemon:
                            cart.acquire(pokemon, count=math.inf)   

        elif pokemon_or_item.startswith("MG_"):
            _, _, item = pokemon_or_item.partition('_')
            for cart in self.transfers["MYSTERYGIFT"]:
                if item in cart.generation.tradeable_items:
                    cart.acquire(item, count=math.inf) 

    def check_rule_possible(self, rule_name, consumed_choices=None):
        rule = self.rules[rule_name]
        for idx in range(len(rule.consumed)):
            c = rule.consumed[idx]
            choice = None
            if consumed_choices is not None:
                choice = consumed_choices[idx]
            if isinstance(c, tuple):
                if consumed_choices is None:
                    if not any([self.state.get(sub_c, 0) >= 1 for sub_c in c]):
                        return False
                elif choice is None:
                    raise ValueError(f"No choice for consumed '{c}'")
                else:
                    if not self.state.get(c[consumed_choices[idx]], 0) >= 1:
                        return False
            else:
                if choice is not None:
                    raise ValueError(f"Choice for non-tuple consumed '{c}'")
                if not self.state.get(c, 0) >= 1:
                    return False

        def required_possible(required):
            if isinstance(required, tuple):
                return any([self.state.get(r, 0) >= 1 for r in required])
            return self.state.get(required, 0) >= 1

        if not all([required_possible(r) for r in rule.required]):
            return False

        if rule.repeats == 0:
            return False
        return True

    def apply_rule_if_possible(self, rule_name, consumed_choices=None):
        if consumed_choices is None:
            consumed_choices = [None] * len(rule.consumed)
        if not self.check_rule_possible(rule_name, consumed_choices):
            return False
        rule = self.rules[rule_name]
        for idx, c in enumerate(rule.consumed):
            if isinstance(c, tuple):
                c = c[consumed_choices[idx]]
            self.state[c] -= 1

        for o in rule.output:
            self.acquire(o, count=1)
        rule.repeats -= 1
        if rule.repeats == 0:
            del self.rules[rule_name]
        return True
      

    def apply_rule_if_safe(self, rule_name):
        rule = self.rules[rule_name]
        if all([self.state.get(o, 0) == math.inf for o in rule.output]) and all([self.dex[d] for d in rule.dex]):
            del self.rules[rule_name]
            return False

        def consumed_safe(consumed):
            if isinstance(consumed, tuple):
                return any([self.state.get(c, 0) == math.inf for c in consumed])
            return self.state.get(consumed, 0) == math.inf

        def required_possible(required):
            if isinstance(required, tuple):
                return any([self.state.get(r, 0) >= 1 for r in required])
            return self.state.get(required, 0) >= 1

        if not all([consumed_safe(c) for c in rule.consumed]):
            return False
        if not all([required_possible(r) for r in rule.required]):
            return False

        for o in rule.output:
            self.acquire(o, count=rule.repeats)
        for d in rule.dex:
            self.dex[d] = True
        del self.rules[rule_name]
        return True

    def apply_all_safe_rules(self):
        while True:
            changed = False
            for name in list(self.rules.keys()):
                changed |= self.apply_rule_if_safe(name)

            if not changed:
                break

    def rule_graph(self, draw=False):

        required = set()
        G = nx.DiGraph()
        G.add_node("START")
        for rule_name, rule in self.rules.items():
            for c in rule.consumed:
                if not isinstance(c, tuple):
                    c = [c]
                for subc in c:
                    count = self.state.get(subc, 0)
                    if count == math.inf:
                        continue
                    G.add_edge(subc, rule_name, kind="consumed")
                    if count > 0:
                        G.add_edge("START", subc)
            for r in rule.required:
                if not isinstance(r, tuple):
                    r = [r]
                for subr in r:
                    count = self.state.get(subr, 0)
                    if count == math.inf:
                        continue
                    G.add_edge(subr, rule_name, kind="required")
                    required.add(subr)
                    if count > 0:
                        G.add_edge("START", subr)
            for o in rule.output:
                count = self.state.get(o, 0)
                if count == math.inf:
                    continue
                G.add_edge(rule_name, o)
            for d in rule.dex:
                species = d.split('_')[0]
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
        for pokemon_or_item in self.get_useful(G.nodes):
                G.add_edge(pokemon_or_item, "END")

        if "END" not in G.nodes:
            return G

        if draw:
            A = nx.nx_agraph.to_agraph(G)
            A.layout('dot')
            A.draw('graph.png')
        return G
 
    def get_useful(self, pokemon_or_items):
        '''
        Return the subset of pokemon_or_items that is directly useful for this cartridge, or might
        be useful for another cartridge
        '''
        useful = set()
        item_carts = set(self.transfers["ITEMS"])
        if "NO_TRADE_ITEMS" not in self.game.props:
            for trade_cart in self.transfers["TRADE"]:
                if "NO_TRADE_ITEMS" not in trade_cart.game.props:
                    item_carts.add(trade_cart)
        for pokemon_or_item in pokemon_or_items:
            species = pokemon_or_item.split('_')[0]
            if species in self.dex and not self.dex[species]:
                useful.add(pokemon_or_item)

            if pokemon_or_item in self.generation.tradeable_items:
                item = pokemon_or_item
                for cart in item_carts:
                    if cart.state.get(item, 0) != math.inf:
                        useful.add(item)
                        break
            
            if pokemon_or_item.startswith("MG_"):
                _, _, item = pokemon_or_item.partition("_")
                for  cart in self.transfers["MYSTERYGIFT"]:
                    if cart.state.get(item, 0) != math.inf:
                        useful.add(pokemon_or_item)
                        break
        return useful


    def all_safe_paths(self):
        self.apply_all_safe_rules()
        G = self.rule_graph()

        next_steps = {}
        total_paths = {}

        for path in nx.all_simple_paths(G, "START", "END"):
            for idx in range(1, len(path) - 2, 2):
                pokemon_or_item = path[idx]
                rule_name = path[idx + 1]
                if pokemon_or_item not in next_steps:
                    next_steps[pokemon_or_item] = {}
                    total_paths[pokemon_or_item] = 0
                if rule_name not in next_steps[pokemon_or_item]:
                    next_steps[pokemon_or_item][rule_name] = 0
                total_paths[pokemon_or_item] += 1
                next_steps[pokemon_or_item][rule_name] += 1

        while True:
            any_changed = False
            for rule_name, rule in list(self.rules.items()):
                if rule_name not in G.nodes:
                    continue
                if not self.check_rule_possible(rule_name):
                    continue
                is_safe = True

                def consumed_safe(c):
                    pathcount = total_paths.get(c, 0)
                    if pathcount == 0 or pathcount > self.state[c]:
                        return False
                    if next_steps.get(c, {}).get(rule_name, 0) == 0:
                        return False
                    return True

                consumed_choices = []
                for c in rule.consumed:
                    if isinstance(c, tuple):
                        safe_idxs = []
                        for idx, sub_c in enumerate(c):
                            if consumed_safe(sub_c):
                                safe_idxs.append(idx)
                        if len(safe_idxs) == 0:
                            is_safe = False
                            break
                        consumed_choices.append(safe_idxs)
                    else:
                        if not consumed_safe(c):
                            is_safe = False
                            break
                        consumed_choices.append([None])
                            
                if not is_safe:
                    continue
                for choices in product(*consumed_choices):
                    changed = self.apply_rule_if_possible(rule_name, choices)
                    if changed:
                        any_changed = True
                        for idx, c in enumerate(rule.consumed):
                            choice = choices[idx]
                            if choice is not None:
                                c = c[choice]
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
        
    def explore_component(self, G):
        final_dexes = set()
        ruleset = set(G.nodes).intersection(self.rules.keys())

        for rule_name in ruleset:
            rule = self.rules[rule_name]
            consumed_choices = []
            for c in rule.consumed:
                if isinstance(c, tuple):
                    consumed_choices.append(range(len(c)))
                else:
                    consumed_choices.append([None])
            for choices in product(*consumed_choices):
                if not self.check_rule_possible(rule_name, choices):
                    continue
                game_copy = self.copy()
                game_copy.apply_rule_if_possible(rule_name, choices)
                game_copy.all_safe_paths()
                Gcopy = G.copy()
                dexes = game_copy.explore_component(Gcopy)
                final_dexes = final_dexes.union(dexes)
        
        if not final_dexes:
            final_dexes.add(tuple(self.dex.items()))
        return final_dexes


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
                for cart2 in cart.transfers["CONNECT"]:
                    G.add_edge(cart.id, cart2.id)
            for cart in self.cartridges:
                if cart == self.main_cartridge:
                    continue
                if not nx.has_path(G, cart.id, self.main_cartridge.id):
                    raise ValueError(f"Game {cart.game.name} cannot interact with main game {self.main_cartridge.game.name} (possibly due to specified hardware)")

    def can_play(self, game):
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
            if self.hardware.get("GCN", 0) + self.hardware.get("Wii", 0):
                return True

        return False

    def _init_interactions(self):
        gb1s = self.hardware.get("GB", 0)
        gb2s = self.hardware.get("GBP", 0) + self.hardware.get("GBC", 0) + self.hardware.get("GBA", 0)
        gbcs = self.hardware.get("GBC", 0) + self.hardware.get("GBA", 0)
        gba1s = self.hardware.get("GBA", 0)
        gba2s = self.hardware.get("GBM", 0)


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
            (self.hardware.get("GCN", 0) >= 1 or self.hardware.get("Wii", 0) >= 1) and \
            self.hardware.get("DOL-011", 0) >= 1

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
                elif game1.name == "BOX":
                    if game2.gen == 3 and game2.core:
                        if gba_gcn:
                            cart1.transfer(cart2, "POKEMON")
                            cart1.transfer(cart2, "ITEMS")
                elif game1.name in ["COLOSSEUM", "XD"]:
                    if game2.gen == 3 and game2.core:
                        if gba_gcn:
                            cart1.transfer(cart2, "TRADE")
                elif game1.name == "BONUSDISC":
                    if game2.name in ["RUBY", "SAPPHIRE"]:
                        cart1.transfer(cart2, "CONNECT")

    def calc_dexes(self):
        if len(self.cartridges) == 1:
            self.main_cartridge.all_safe_paths()
        else:
            state_copies = {cart.id: None for cart in self.cartridges}
            while True:
                updates = False
                for cart in self.cartridges:
                    if state_copies[cart.id] != cart.state:
                        updates = True
                        state_copies[cart.id] = deepcopy(cart.state)
                        cart.all_safe_paths()
                if not updates:
                    break
            for cart in self.cartridges:
                cart.try_paths(only_side_effects=True)
            while True:
                updates = False
                for cart in self.cartridges:
                    if state_copies[cart.id] != cart.state:
                        updates = True
                        state_copies[cart.id] = deepcopy(cart.state)
                        cart.try_paths(only_side_effects=True)
                if not updates:
                    break

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

        self.dexes = dexes
        self.groups = groups
        self.group_combos = group_combos
        self.pokemon_by_group_idx = pokemon_by_group_idx

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


class Rule():
    def __init__(self, name, consumed, required, output, repeats, dex=None):
        self.name = name
        self.consumed = consumed
        self.required = required
        self.output = output
        self.repeats = repeats
        self.dex = dex or []

    def __repr__(self):
        return '/'.join(self.consumed) + ' -> ' + '/'.join(self.output)

    def __copy__(self):
        return Rule(self.name, self.consumed, self.required, self.output, self.repeats)

    def __deepcopy__(self, memo):
        return Rule(
            deepcopy(self.name, memo),
            deepcopy(self.consumed, memo),
            deepcopy(self.required, memo),
            deepcopy(self.output, memo),
            deepcopy(self.repeats, memo),
        )

    @staticmethod
    def breed(parent, child, ditto=False, item=None):
        required = [parent]
        if ditto:
            required.append("Ditto")
        if item:
            required.append(item)
        return Rule(f"breed:{parent}->{child}", [], required, [child], math.inf)

    @staticmethod
    def buy(pokemon_or_item, exchange):
        consumed = []
        if exchange:
            consumed.append(exchange)
        return Rule(f"buy:{pokemon_or_item}", consumed, [], [pokemon_or_item], math.inf)

    @staticmethod
    def choice(name, requirements, repeats=1):
        requirements = requirements or []
        name = f"choice:{name}"
        return Rule(name, [], requirements, [name], repeats)

    @staticmethod
    def choose(name, pokemon_or_items):
        name = f"choice:{name}"
        return Rule(f"{name}->{pokemon_or_items}", [name], [], pokemon_or_items, math.inf)

    @staticmethod
    def evolve(pre, post, item):
        consumed = [pre]
        if item:
            consumed.append(item)
        if not isinstance(post, list):
            post = [post]
        return Rule(f"evolve:{pre}->{post}", consumed, [], post, math.inf)

    @staticmethod
    def ingame_trade(give, get, item, evolution, choice=None):
        if evolution is None:
            output = [get]
            dex = None
        else:
            output = [evolution]
            dex = [get]
        if item:
            output.append(item)
        consumed = [give]
        if choice is not None:
            consumed.append(f"choice:{choice}")
        return Rule(f"trade:{give}->{get}", consumed, [], output, 1, dex=dex)

    @staticmethod
    def fossil(fossil, pokemon):
        return Rule(f"fossil:{fossil}->{pokemon}", [fossil], [], [pokemon], math.inf)
    
    @staticmethod
    def gift(pokemon_or_items, requirements):
        requirements = requirements or []
        return Rule(f"gift:{pokemon_or_items}", [], requirements, pokemon_or_items, 1)

    @staticmethod
    def pickup(item, pokemon):
        return Rule(f"pickup:{pokemon}->{item}", [], [pokemon], [item], math.inf)

    @staticmethod
    def wild(pokemon_or_item, required, repeats, item=None):
        output = [pokemon_or_item]
        if item:
            output.append(item)
        required = required or []
        return Rule(f"wild:{pokemon_or_item}", [], required, output, repeats)



def parse_game(games, game_list):
    if games == "CORE":
        return [g for g in game_list if GAMES[g].core]
    games = games.split(',')
    bad = [game for game in games if game not in game_list]
    if bad:
        raise ValueError(f"Invalid game(s) {bad}")
    return games

def parse_gift_entry(entry, gen, gender_func):
    entry, _, reqs = entry.partition('[')
    reqs = reqs or []
    if reqs:
        reqs = reqs[:-1].split(',')
    for req in reqs.copy():
        if req.startswith("DEX_"):
            reqs.remove(req)
            reqs += [p for p in gen.pokemon_list.index if gen.pokemon_list.loc[p, req]]
    choices = entry.split(';')
    choices = [c.split(',') for c in choices]

    choices_w_gender = []
    for choice_group in choices:
        gendered = [gender_func(choice) for choice in choice_group]
        for group in product(*gendered):
            choices_w_gender.append(group)
    return choices_w_gender, reqs

def parse_wild_entry(entry):
    out = []
    if isinstance(entry, float) or isinstance(entry, int):
        return [(entry, [])]
    entries = entry.split(';')
    for entry in entries:
        entry, _, reqs = entry.partition('[')
        reqs = reqs or []
        if reqs:
            reqs = reqs[:-1].split(',')
        entry = float(entry)
        out.append((entry, reqs))
    return out

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
    games = []
    hardware = {}
    for item in args.game_or_hardware:
        if item in GAMES:
            games.append(GAMES[item])
        else:
            hardware[item] = hardware.get(item, 0) + 1
    generations = {gen: Generation(gen) for gen in set([g.gen for g in games])}
    cartridges = [Cartridge(g, generations[g.gen]) for g in games]
    collection = Collection(cartridges, hardware)
    for cart in cartridges:
        cart.init_rules()

    collection.calc_dexes()
    obtainable = collection.obtainable()
    print("\n".join(obtainable))
    print("\n---\n")
    print("\n".join(collection.missing()))
    print("\n---\n")
    print(f"TOTAL: {len(obtainable)}")
    #cart = collection.main_cartridge
    #import pdb; pdb.set_trace()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_or_hardware', nargs='+', default=[])
    args = parser.parse_args()
    main(args)
