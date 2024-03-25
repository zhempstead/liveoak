#!/usr/bin/python

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from frozendict import frozendict
import networkx as nx
import pandas as pd

@dataclass(frozen=True)
class Game():
    '''
    Represents a particular game, complete with language and region
    (i.e. French and English BLACK2 are distinct Games)
    '''
    name: str
    gen: int
    console: str
    core: bool
    region: Optional[str]
    language: Optional[str]
    props: frozenset = frozenset([])

    @classmethod
    def parse(cls, gamestr, console=None):
        split = gamestr.split('.')
        input_name = split[0]
        if input_name not in GAMEINPUTS:
            raise ValueError(f"Unrecognized game '{split[0]}'")
        region = None
        if console is not None:
            region = console.region
        language = None
        for subitem in split[1:]:
            if subitem in REGIONS:
                if region is not None:
                    raise ValueError(f"'{gamestr}' has multiple regions specified.")
                region = subitem
            elif subitem in LANGUAGES:
                if language is not None:
                    raise ValueError(f"'{gamestr}' has multiple languages specified.")
                language = subitem
            else:
                raise ValueError(f"Unrecognized region/language '{subitem}'.")
        game = None
        if region is None and language is None:
            for r in DEFAULT_REGIONS:
                for l in DEFAULT_LANGUAGES[r]:
                    game = cls._get_game(input_name, r, l)
                    if game is not None:
                        break
                if game is not None:
                    break
            if game is None:
                raise ValueError("Not handled")
        elif region is None and language is not None:
            for r in DEFAULT_REGIONS:
                game = cls._get_game(input_name, r, language)
                if game is not None:
                    break
            if game is None:
                raise ValueError(f"Game {input_name} doesn't come in language {language}.")
        elif region is not None and language is None:
            for l in DEFAULT_LANGUAGES[region]:
                game = cls._get_game(input_name, region, l)
                if game is not None:
                    break
            if game is None:
                raise ValueError(f"Game {input_name} wasn't released in region {region}.")
        else:
            game = cls._get_game(input_name, region, language)
            if game is None:
                raise ValueError(f"Game {input_name} wasn't released in the {region} region in {language}.")
        return game

    @classmethod
    def _get_game(cls, input_name, region, language):
        languages = [language]
        if language in EUROPEAN_LANGUAGES:
            languages.append("European")
        candidates = GAMES[(GAMES.GAME == input_name) & (GAMES.REGION == region) & (GAMES.LANGUAGE.isin(languages))]
        if len(candidates) > 1:
            raise ValueError("Shouldn't happen")
        if len(candidates) == 1:
            c = candidates.iloc[0]
            props = frozenset()
            if c.TAGS:
                props = frozenset(c.TAGS.split(','))
            return cls(c.GAMEID, c.GENERATION, c.CONSOLE or None, c.CORE, c.REGION or None, c.LANGUAGE or None, props)
        return None

    def default_flow(self):
        if self.console in {"GB", "GBC", "N64"}:
            return "GB"
        elif self.console in {"GBA", "GCN"}:
            return "GBA"
        elif self.console in {"DS", "3DS", "Wii", None}:
            return "DS"

    def language_match(self, other):
        l1 = self.language
        l2 = other.language
        if l1 == l2:
            return True
        if l1 == "European" and l2 in EUROPEAN_LANGUAGES:
            return True
        if l2 == "European" and l1 in EUROPEAN_LANGUAGES:
            return True
        return False

    def region_match(self, other):
        if self.region is None:
            return True
        if other.region is None:
            return True
        if self.region == "INTL" and other.region in INTL_REGIONS:
            return True
        if other.region == "INTL" and self.region in INTL_REGIONS:
            return True
        return self.region == other.region

    def match(self, gamestr):
        if gamestr == "CORE":
            return self.core
        for g in gamestr.split(','):
            g, _, r_or_l = g.partition('.')
            if g != self.name:
                continue
            if not r_or_l:
                return True
            if r_or_l in {self.region, self.language}:
                return True
        return False

    def __repr__(self):
        out = self.name
        if self.region not in {None, 'USA'}:
            out += f".{self.region}"
        if self.language is not None:
            if (
                        (self.region in {None, 'USA', 'EUR', 'AUS'} and self.language not in {'English', 'European'}) or \
                        (self.region == "JPN" and self.language != "Japanese") or \
                        (self.region == "KOR" and self.language != "Korean") or \
                        (self.region == "CHN" and self.language != "Chinese")
            ):
                out += f".{self.language}"
        return out

_cartridge_idx = 0

@dataclass(frozen=True)
class Cartridge():
    '''
    Represents a single copy (cartridge, disc, or software) of a game, with a unique ID
    '''
    game: Game
    cl_name: str
    console: Optional[str] = None # For software games only
    id: int = field(init=False)

    def __post_init__(self):
        global _cartridge_idx
        object.__setattr__(self, 'id', _cartridge_idx)
        _cartridge_idx += 1

        if self.console is None:
            if "SOFTWARE" in self.game.props:
                raise ValueError(f"{self} is software-only.")
        else:
            if self.game.console not in self.console.model.software:
                raise ValueError(f"Console {self.console} does not support {self.game.console} software games")
            if not self.console.region_match(self.game):
                raise ValueError(f"Console {self.console} has incompatible region with {self}.")

    def __str__(self):
        return self.cl_name

    def __repr__(self):
        return f"{self.game.__repr__()}_{self.id}"

    def connections(self, collection, flow):
        can_play = self.game.console is None
        connects_to = set()
        G_start = nx.DiGraph()
        G_start.add_node(self)
        for G in self._connections(G_start, collection, flow):
            G = self._simplify_connection_graph(G)
            G = self._filter_connection_graph(G)

            for node in G:
                if isinstance(node, Hardware):
                    if node.model.is_console:
                        can_play = True
                elif isinstance(node, Cartridge):
                    if node != self:
                        connects_to.add(node)
        return can_play, connects_to


    def _connections(self, G, collection, flow):
        '''
        Yield directed graphs representing ways the cartridges/hardware can be connected
        '''
        found = False
        for node in G:
            if G.edges(node):
                continue
            if isinstance(node, Cartridge):
                if G.in_edges(node):
                    continue
                cart = node
                if cart.console is None:
                    for hw, pidx in collection.hw_ports.get(flow, {}).get(cart.game.console, ()):
                        if not hw.region_match(cart):
                            continue
                        pname = f"port_{pidx}"
                        if (hw, pname) in G.nodes and G.edges((hw, pname)):
                            continue
                        found = True
                        Gcopy = self._add_hardware(G, hw, pname, cart, collection.sw_carts, flow)
                        for Gfull in self._connections(Gcopy, collection, flow):
                            yield Gfull
                else:
                    Gcopy = self._add_hardware(G, cart.console, None, cart, collection.sw_carts, flow)
                    for Gfull in self._connections(Gcopy, collection, flow):
                        yield Gfull
                continue

            if not isinstance(node, tuple):
                continue
            hardware, conn_name = node
            if conn_name.startswith("port"):
                idx = int(conn_name.split('_')[1])
                for plug_type in hardware.model.ports[idx].get(flow, ()):
                    for hw, pidx in collection.hw_plugs.get(flow, {}).get(plug_type, ()):
                        if not hardware.region_match(hw):
                            continue
                        pname = f"plug_{pidx}"
                        if (hw, pname) in G.nodes and G.edges((hw, pname)):
                            continue
                        found = True
                        Gcopy = self._add_hardware(G, hw, pname, node, collection.sw_carts, flow)
                        for Gfull in self._connections(Gcopy, collection, flow):
                            yield Gfull
                    for cart in collection.hw_carts.get(plug_type, ()):
                        if not hardware.region_match(cart):
                            continue
                        if cart in G.nodes:
                            continue
                        found = True
                        Gcopy = G.copy()
                        Gcopy.add_edge(node, cart)
                        for Gfull in self._connections(Gcopy, collection, flow):
                            yield Gfull

            elif conn_name.startswith("plug"):
                idx = int(conn_name.split('_')[1])
                for port_type in hardware.model.plugs[idx].get(flow, ()):
                    for hw, pidx in collection.hw_ports.get(flow, {}).get(port_type, ()):
                        if not hardware.region_match(hw):
                            continue
                        pname = f"port_{pidx}"
                        if (hw, pname) in G.nodes and G.edges((hw, pname)):
                            continue
                        found = True
                        Gcopy = self._add_hardware(G, hw, pname, node, collection.sw_carts, flow)
                        for Gfull in self._connections(Gcopy, collection, flow):
                            yield Gfull

            elif conn_name == "wireless":
                for hw in collection.hw_wireless.get(flow, ()):
                    if (hw, "wireless") in G.nodes and G.edges((hw, "wireless")):
                        continue
                    found = True
                    Gcopy = self._add_hardware(G, hw, "wireless", node, collection.sw_carts, flow)
                    for Gfull in self._connections(Gcopy, collection, flow):
                        yield Gfull
            else:
                raise ValueError("Unexpected")
        if not found:
            yield G

    def _add_hardware(self, G, hw, in_connection, from_node, software, flow):
        '''
        Return copy of G with from_node linked to relevant port/plug of G
        '''
        G = G.copy()
        if in_connection is None:
            if hw in G.nodes:
                raise ValueError("Unexpected")
            G.add_edge(from_node, hw)
        else:
            G.add_edge(from_node, (hw, in_connection))
            if hw in G.nodes:
                G.remove_edge(hw, (hw, in_connection))
            G.add_edge((hw, in_connection), hw)
        for idx, port in enumerate(hw.model.ports):
            name = f"port_{idx}"
            if in_connection == name or flow not in port:
                continue
            if (hw, name) not in G.nodes:
                G.add_edge(hw, (hw, name))
        for idx, plug in enumerate(hw.model.plugs):
            name = f"plug_{idx}"
            if in_connection == name or flow not in plug:
                continue
            if (hw, name) not in G.nodes:
                G.add_edge(hw, (hw, name))
        if flow in hw.model.wireless:
            name = "wireless"
            if in_connection != name and (hw, name) not in G.nodes:
                G.add_edge(hw, (hw, name))
        for cart in software.get(hw, ()):
            if cart not in G.nodes:
                G.add_edge(hw, cart)
        return G

    def _simplify_connection_graph(self, G_in):
        '''
        Convert from directed graph of ports/plugs to undirected graph of games/hardware
        '''
        G = nx.Graph()
        for node1, node2 in G_in.edges:
                if isinstance(node1, tuple):
                    node1 = node1[0]
                if isinstance(node2, tuple):
                    node2 = node2[0]
                if node1 != node2:
                    G.add_edge(node1, node2)
        return G

    def _filter_connection_graph(self, G_in):
        G = G_in.copy()
        to_delete = set()
        for node in G.nodes:
            # GB/GBC games can only be played on N64 with an appropriate version of STADIUM
            if isinstance(node, Hardware) and node.model.name == "N64":
                n64 = node
                n64_game = None
                tpak_games = set()
                for node2 in G.adj[n64]:
                    if isinstance(node2, Cartridge):
                        n64_game = node2.game
                    elif isinstance(node2, Hardware) and node2.model.name == "TRANSFERPAK":
                        tpak = node2
                        for node3 in G.adj[tpak]:
                            if isinstance(node3, Cartridge):
                                tpak_games.add((tpak, node3.game))
                                break
                if n64_game is None:
                    for tpak, _ in tpak_games:
                        to_delete.add(tpak)
                elif n64_game.name in {"STADIUM_JP", "STADIUM"}:
                    for tpak, game in tpak_games:
                        if game.gen != 1 or game.language != n64_game.language:
                            to_delete.add(tpak)
                elif n64_game.name == "STADIUM2":
                    for tpak, game in tpak_games:
                        if game.language != n64_game.language:
                            to_delete.add(tpak)
                else:
                    raise ValueError(f"Unexpected N64 game {n64_game}")
            # - GBP requires the GBP disc
            # - The GCN-GBA connector cannot be used to connect a game on GBA with a game on GBP
            elif isinstance(node, Hardware) and node.model.name == "GCN":
                gcn = node
                gbp = None
                game = None
                connector = None
                for node2 in G.adj[gcn]:
                    if isinstance(node2, Hardware) and node2.model.name == "GBP":
                        gbp = node2
                    elif isinstance(node2, Cartridge):
                        game = node2.game.name
                    elif isinstance(node2, Hardware) and node2.model.name == "DOL-011":
                        connector = node2
                    if gbp is not None:
                        if disc != "GBPDISC":
                            to_delete.add(gbp)
                        elif connector is not None:
                            to_delete.add(connector)

        if not to_delete:
            return G
        for node in to_delete:
            G.remove_node(node)
        component = nx.node_connected_component(G, self)
        return nx.subgraph_view(G, filter_node=(lambda n: n in component))


@dataclass(frozen=True)
class HardwareModel():
    name: str
    regions: Optional[frozenset] = None
    is_console: bool = False
    software: frozenset[str] = field(default_factory=frozenset)
    ports: tuple[frozendict[str, tuple[str, ...]], ...] = ()
    plugs: tuple[frozendict[str, tuple[str, ...]], ...] = ()
    wireless: frozenset[str] = field(default_factory=frozenset)

    def __repr__(self):
        return name

_hardware_idx = 0

@dataclass(frozen=True)
class Hardware():
    model: HardwareModel
    region: Optional[str] = None
    id: int = field(init=False)

    def __post_init__(self):
        global _hardware_idx
        object.__setattr__(self, 'id', _hardware_idx)
        _hardware_idx += 1

        if self.region is not None and self.model.regions is None:
            raise ValueError(f"{self.model} shouldn't have a region")
        if self.model.regions is not None and self.region not in self.model.regions:
            raise ValueError(f"{self.model} wasn't released in region {self.region}")

    @classmethod
    def parse(cls, hardware_str):
        model_name, _, region = hardware_str.partition('.')
        if model_name not in HARDWARE_MODELS:
            raise ValueError(f"Unrecognized hardware '{model_name}'")
        model = HARDWARE_MODELS[model_name]
        if region == '':
            if model.regions is None:
                region = None
            else:
                for r in DEFAULT_REGIONS:
                    if r in model.regions:
                        region = r
                        break
                if not region:
                    raise ValueError("Shouldn't happen")
        return cls(model=model, region=region)

    def region_match(self, other):
        if isinstance(other, Cartridge):
            other = other.game
        if not isinstance(other, Hardware) and not isinstance(other, Game):
            raise ValueError(f"{other} is not a Hardware or Game instance.")
        if self.region is None:
            return True
        if other.region is None:
            return True
        return self.region == other.region

    def __repr__(self):
        if self.region is None or self.region == "USA":
            return f"{self.model.name}_{self.id}"
        return f"{self.model.name}.{self.region}_{self.id}"


DEFAULT_REGIONS = [False, "INTL", "USA", "JPN", "EUR", "AUS", "KOR", "CHN"]
DEFAULT_LANGUAGES = {False: [False, "European", "English", "Japanese"], "INTL": [False, "English"], "USA": [False, "English"], "JPN": [False, "Japanese"], "EUR": [False, "European", "English"], "AUS": [False, "English"], "KOR": [False, "Korean"], "CHN": [False, "Chinese"]}

GAMES = pd.read_csv(Path('data') / 'games.csv').fillna(False)
GAMEIDS = set(GAMES.GAMEID)
GAMEINPUTS = set(GAMES.GAME)
REGIONS = set(GAMES.REGION)
LANGUAGES = set(GAMES.LANGUAGE)
EUROPEAN_LANGUAGES = {"English", "French", "German", "Italian", "Spanish"}
INTL_REGIONS = {"USA", "EUR", "AUS"}

HARDWARE_MODELS = {h.name: h for h in [
    # Handhelds
    # Original Game Boy
    HardwareModel("GB", is_console=True, ports=(
        frozendict({"GB": ("GB",)}),
        frozendict({"GB": ("GLC1",)}),
    )),
    # Game Boy Pocket/Light; has smaller cable ports
    HardwareModel("GB2", is_console=True, ports=(
        frozendict({"GB": ("GB",)}),
        frozendict({"GB": ("GLC2",)}),
    )),
    # Game Boy Color; has same cable ports as Pocket/Light
    HardwareModel("GBC", is_console=True, ports=(
        frozendict({"GB": ("GB", "GBC"), "GBCir": ("GBC",)}),
        frozendict({"GB": ("GLC2",)}),
    ), wireless=frozenset({"GBCir"})),
    # Game Boy Advance; includes SP
    HardwareModel("GBA", is_console=True, ports=(
        frozendict({"GB": ("GB", "GBC"), "GBA": ("GBA",), "GBAw": ("GBA",)}),
        frozendict({"GB": ("GLC2", "GLC3g"), "GBA": ("GLC3p", "GLC3g"), "GBAw": ("GLC3p",)}),
    )),
    # Game Boy Micro; can only play GBA games, has different cable port
    HardwareModel("GBM", is_console=True, ports=(
        frozendict({"GBA": ("GBA",), "GBAw": ("GBA",)}),
        frozendict({"GBA": ("GLC4",), "GBAw": ("GLC4",)}),
    )),
    # DS and DS Lite - both have GBA slot and no region locking
    HardwareModel("DS", is_console=True, ports=(
        frozendict({"GBA": ("GBA",), "DS": ("GBA",)}),
        frozendict({"DS": ("DS",)}),
    ), wireless=frozenset({"DS"})),
    # DSi and DSi XL - no GBA slot and region locking
    HardwareModel("DSi", frozenset({"JPN", "USA", "EUR", "AUS", "KOR", "CHN"}), True, ports=(
        frozendict({"DS": ("DS",)}),
    ), wireless=frozenset({"DS"})),
    # 3DS, 3DS XL, 2DS, 2DS XL
    HardwareModel("3DS", frozenset({"JPN", "USA", "EUR", "KOR", "TWN", "CHN"}), True, frozenset({"3DS"}), (
        frozendict({"DS": ("DS", "3DS")}),
    ), wireless=frozenset({"DS"})),

    # Home consoles
    # Super Nintendo. No relevant games, but works with the Super Game Boy
    HardwareModel("SNES", frozenset({"JPN", "USA", "EUR"}), True, ports=(
        frozendict({"GB": ("SNES",)}),
    )),
    # Nintendo 64
    HardwareModel("N64", frozenset({"JPN", "USA", "EUR", "CHN"}), True, ports=(
        frozendict({"GB": ("N64",), "STADIUM2": ("N64",)}),
        frozendict({"GB": ("N64c",), "STADIUM2": ("N64c",)}),
        frozendict({"GB": ("N64c",), "STADIUM2": ("N64c",)}),
    )),
    # Game Cube
    HardwareModel("GCN", frozenset({"JPN", "USA", "EUR"}), True, ports=(
        frozendict({"GB": ("GCN",), "GBA": ("GCN",), "GBAw": ("GCN",)}),
        frozendict({"GB": ("GCNc",), "GBA": ("GCNc",)}),
        frozendict({"GB": ("GCNp",), "GBA": ("GCNp",), "GBAw": ("GCNp",)}),
    )),
    # Original Wii; supports GCN games and have ports for the controllers
    HardwareModel("Wii", frozenset({"JPN", "USA", "EUR", "KOR"}), True, ports=(
        frozendict({"GB": ("GCN",), "GBA": ("GCN",), "DS": ("Wii",)}),
        frozendict({"GB": ("GCNc",), "GBA": ("GCNc",)}),
    ), wireless=frozenset({"DS"})),
    # Later Wii models and Wii U. Later Wiis lack GCN support, and there are no relevant Wii U
    # games, so for our purposes the two are equivalent.
    HardwareModel("WiiU", frozenset({"JPN", "USA", "EUR", "KOR"}), True, ports=(
        frozendict({"DS": ("Wii",)}),
    ), wireless=frozenset({"DS"})),

    # Peripherals
    # Super Game Boy
    HardwareModel("SGB", frozenset({"JPN", "USA", "EUR"}), ports=(
        frozendict({"GB": ("GB",)}),
    ), plugs=(
        frozendict({"GB": ("SNES",)}),
    )),
    # Super Game Boy 2; only released in Japan, has game link cable port
    HardwareModel("SGB", frozenset({"JPN"}), ports=(
        frozendict({"GB": ("GB",)}),
        frozendict({"GB": ("GLC2",)}),
    ), plugs=(
        frozendict({"GB": ("SNES",)}),
    )),
    # Transfer Pak; connects a GB/GBC game to the N64
    HardwareModel("TRANSFERPAK", ports=(
        frozendict({"GB": ("GB", "GBC"), "STADIUM2": ("GB", "GBC")}),
    ), plugs=(
        frozendict({"GB": ("N64c",), "STADIUM2": ("N64c",)}),
    )),
    # Game Boy Player
    HardwareModel("GBP", ports=(
        frozendict({"GB": ("GB", "GBC"), "GBA": ("GBA",), "GBAw": ("GBA",)}),
        frozendict({"GB": ("GLC2", "GLC3g"), "GBA": ("GLC3p", "GLC3g"), "GBAw": ("GLC3p"),}),
    ), plugs=(
        frozendict({"GB": ("GCNp",), "GBA": ("GCNp",), "GBAw": ("GCNp",)}),
    )),
    # e-Reader
    HardwareModel("eREADER", frozenset({"JPN", "USA"}), ports=(
        frozendict({"GBA": ("eREADER",), "GBAw": ("eReader",)}),
    ), plugs=(
        frozendict({"GBA": ("GBA",), "GBAw": ("GBA",)}),
    )),

    # Cables and adapters
    # Original Game Link Cable for original GB
    HardwareModel("DMG-04", plugs=(
        frozendict({"GB": ("GLC1",)}),
        frozendict({"GB": ("GLC1",)}),
    )),
    # Universal Game Link Cable - connects two newer GBs or one new and one old
    HardwareModel("MGB-010", plugs=(
        frozendict({"GB": ("GLC1", "GLC2",)}),
        frozendict({"GB": ("GLC2",)}),
    )),
    # CGB-003 and MGB-008 are equivalent. Both connect two newer GBs.
    HardwareModel("CGB-003", plugs=(
        frozendict({"GB": ("GLC2",)}),
        frozendict({"GB": ("GLC2",)}),
    )),
    # Adapts original cable to newer GB
    HardwareModel("MGB-004", ports=(
        frozendict({"GB": ("GLC1",)}),
    ), plugs=(
        frozendict({"GB": ("GLC2",)}),
    )),
    # Adapts newer cable to original GB
    HardwareModel("DMG-14", ports=(
        frozendict({"GB": ("GLC2",)}),
    ), plugs=(
        frozendict({"GB": ("GLC1",)}),
    )),
    # GBA link cable. Only works with GBA. Required for GBA games.
    # Two can be used to connect GB/GBC games.
    HardwareModel("AGB-005", ports=(
        frozendict({"GB": ("GLC3p",)}), 
    ), plugs=(
        frozendict({"GB": ("GLC3p",), "GBA": ("GLC3p",)}),
        frozendict({"GB": ("GLC3g",), "GBA": ("GLC3g",)}),
    )),
    # Game Boy Micro link cable
    HardwareModel("OXY-008", plugs=(
        frozendict({"GBA": ("GLC4",)}),
        frozendict({"GBA": ("GLC4",)}),
    )),
    # Adapts GBM cable to original GBA
    HardwareModel("OXY-009", ports=(
        frozendict({"GBA": ("GLC4",)}),
    ), plugs=(
        frozendict({"GBA": ("GLC3g", "GLC3p")}),
    )),
    # GBA wireless adapter
    HardwareModel("AGB-015", plugs=(
        frozendict({"GBAw": ("GLC3p",)}),
    ), wireless=frozenset({"GBAw"})),
    # GBM wireless adapter
    HardwareModel("OXY-004", plugs=(
        frozendict({"GBAw": ("GLC4",)}),
    ), wireless=frozenset({"GBAw"})),
    # Game Cube - Game Boy Advance Link Cable
    HardwareModel("DOL-011", plugs=(
        frozendict({"GBA": ("GLC3p"),}),
        frozendict({"GBA": ("GCNc"),}),
    )),
]}
