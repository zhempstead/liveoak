#!/usr/bin/python

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Literal, Optional

from networkx import DiGraph, Graph

from frozendict import frozendict # type: ignore
import networkx as nx
import pandas as pd
import numpy as np

type Region = str | Literal[False]
type Language = str | Literal[False]

GB_LANG_GROUPS = {
    "Japanese": 1,
    "English": 2,
    "French": 2,
    "German": 2,
    "Italian": 2,
    "Spanish": 2,
    # Can cause glitches but generally possible to trade between Korean and Western games
    "Korean": 2,
}

VC_LANG_GROUPS = {
    "Japanese": 1,
    "English": 2,
    "French": 2,
    "German": 2,
    "Italian": 2,
    "Spanish": 2,
    # Trading between Western and Korean games disallowed on VC
    "Korean": 3,
}
    

@dataclass(frozen=True)
class Game():
    '''
    Represents a particular game, complete with language and region
    (i.e. French and English BLACK2 are distinct Games)
    '''
    name: str
    gen: int
    console: Optional[str]
    core: bool
    region: Optional[str]
    language: Optional[str]
    props: frozenset[str] = frozenset([])

    @staticmethod
    def parse(gamestr: str, console=None) -> "Game":
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
                    game = Game._get_game(input_name, r, l)
                    if game is not None:
                        break
                if game is not None:
                    break
            if game is None:
                raise ValueError("Not handled")
        elif region is None and language is not None:
            for r in DEFAULT_REGIONS:
                game = Game._get_game(input_name, r, language)
                if game is not None:
                    break
            if game is None:
                raise ValueError(f"Game {input_name} doesn't come in language {language}.")
        elif region is not None and language is None:
            for l in DEFAULT_LANGUAGES[region]:
                game = Game._get_game(input_name, region, l)
                if game is not None:
                    break
            if game is None:
                raise ValueError(f"Game {input_name} wasn't released in region {region}.")
        else:
            game = Game._get_game(input_name, region, language)
            if game is None:
                raise ValueError(f"Game {input_name} wasn't released in the {region} region in {language}.")
        return game

    @staticmethod
    def _get_game(input_name: str, region: Region | None, language: Language | None) -> "Game | None":
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
            return Game(c.GAMEID, c.GENERATION, c.CONSOLE or None, c.CORE, c.REGION or None, c.LANGUAGE or None, props)
        return None

    def default_flow(self) -> str:
        if self.console in {"GB", "GBC", "N64"}:
            return "GB"
        elif self.console in {"GBA", "GCN", "eREADER"}:
            return "GBA"
        elif self.console in {"DS", "3DS", "Wii", "SWITCH", None}:
            return "DS"
        assert False, "Shouldn't reach here"

    def language_match(self, other: "Game") -> bool:
        '''
        Returns True iff the languages are identical
        (or overlapping if a cart supports multiple languages)
        '''
        l1 = self.language
        l2 = other.language
        if l1 == l2:
            return True
        if l1 == "European" and l2 in EUROPEAN_LANGUAGES:
            return True
        if l2 == "European" and l1 in EUROPEAN_LANGUAGES:
            return True
        return False

    def language_group(self, other: "Game") -> bool:
        '''
        Returns True iff the languages are compatible for trading
        Only relevant for core gen 1, 2, and 4 games
        '''
        if self.gen in {1, 2}:
            if self.console in {"GB", "GBC"}:
                # Glitchy but possible to trade between Western and Korean games.
                # Japanese games are fully incompatible with others
                return (self.language == "Japanese") == (other.language == "Japanese")
            else:
                # VC disallows Western-Korean trading
                return (self.language == "Japanese") == (other.language == "Japanese") and \
                        (self.language == "Korean") == (other.language == "Korean")
        if self.gen == 4:
            # Korean games were released later, so non-Korean games don't support trading w/Korean
            return (self.language == "Korean") == (other.language == "Korean")
        raise ValueError("Irrelevant")

    def region_match(self, other: "Game") -> bool:
        if self.region is None:
            return True
        if other.region is None:
            return True
        if self.region == "INTL" and other.region in INTL_REGIONS:
            return True
        if other.region == "INTL" and self.region in INTL_REGIONS:
            return True
        return self.region == other.region

    def match(self, gamestr: str) -> bool:
        if gamestr == "ALL":
            return True
        if gamestr == "NONE":
            return False
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
    console: Optional["Hardware"] = None # For software games only
    id: str = field(init=False)

    def __post_init__(self):
        global _cartridge_idx
        object.__setattr__(self, 'id', str(_cartridge_idx))
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

    def connections(self, collection: "Collection", flow: str) -> tuple[set["Hardware"], set["Cartridge"]]:
        consoles = set()
        connects_to = set()
        G_start = nx.DiGraph()
        G_start.add_node(self)
        for G in self._connections(G_start, collection, flow):
            G = self._simplify_connection_graph(G)
            G = self._filter_connection_graph(G)
            if self not in G.nodes:
                continue

            for neighbor in G.adj[self]:
                if neighbor.model.is_console:
                    consoles.add(neighbor)
                else:
                    for neighbor2 in G.adj[neighbor]:
                        if isinstance(neighbor2, Hardware) and neighbor2.model.is_console:
                            consoles.add(neighbor2)

            for node in G:
                if isinstance(node, Cartridge):
                    if node != self:
                        connects_to.add(node)

        return consoles, connects_to


    def _connections(self, G: nx.DiGraph, collection, flow) -> Generator[nx.DiGraph, None, None]:
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
                for hw, pidx in collection.hw_ports.get(flow, {}).get(cart.game.console, ()):
                    if not hw.region_match(cart):
                        continue
                    if cart.console is not None and hw != cart.console:
                        continue
                    pname = f"port_{pidx}"
                    if (hw, pname) in G.nodes and G.edges((hw, pname)):
                        continue
                    found = True
                    Gcopy = self._add_hardware(G, hw, pname, cart, flow)
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
                        if hardware == hw:
                            continue
                        if not hardware.region_match(hw):
                            continue
                        pname = f"plug_{pidx}"
                        if (hw, pname) in G.nodes and G.edges((hw, pname)):
                            continue
                        found = True
                        Gcopy = self._add_hardware(G, hw, pname, node, flow)
                        for Gfull in self._connections(Gcopy, collection, flow):
                            yield Gfull
                    for cart in collection.carts_by_type.get(plug_type, ()):
                        if not hardware.region_match(cart):
                            continue
                        if cart in G.nodes:
                            continue
                        if cart.console is not None and cart.console != hardware:
                            continue
                        found = True
                        Gcopy = G.copy()
                        assert(isinstance(Gcopy, nx.DiGraph))
                        Gcopy.add_edge(node, cart)
                        for Gfull in self._connections(Gcopy, collection, flow):
                            yield Gfull

            elif conn_name.startswith("plug"):
                idx = int(conn_name.split('_')[1])
                for port_type in hardware.model.plugs[idx].get(flow, ()):
                    for hw, pidx in collection.hw_ports.get(flow, {}).get(port_type, ()):
                        if hardware == hw:
                            continue
                        if not hardware.region_match(hw):
                            continue
                        pname = f"port_{pidx}"
                        if (hw, pname) in G.nodes and G.edges((hw, pname)):
                            continue
                        found = True
                        Gcopy = self._add_hardware(G, hw, pname, node, flow)
                        for Gfull in self._connections(Gcopy, collection, flow):
                            yield Gfull

            elif conn_name == "wireless":
                for hw in collection.hw_wireless.get(flow, ()):
                    if hardware == hw:
                        continue
                    if (hw, "wireless") in G.nodes and G.edges((hw, "wireless")):
                        continue
                    found = True
                    Gcopy = self._add_hardware(G, hw, "wireless", node, flow)
                    for Gfull in self._connections(Gcopy, collection, flow):
                        yield Gfull
            else:
                raise ValueError("Unexpected")
        if not found:
            yield G

    def _add_hardware(self, G: nx.DiGraph, hw: "Hardware", in_connection: str, from_node: "Cartridge | tuple[Hardware, str]", flow: str) -> nx.DiGraph:
        '''
        Return copy of G with from_node linked to relevant port/plug of G
        '''
        G = G.copy() # type: ignore
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
        return G

    def _simplify_connection_graph(self, G_in: nx.DiGraph) -> nx.Graph:
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

    def _filter_connection_graph(self, G_in: nx.Graph) -> nx.Graph:
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
                        if game != "GBPDISC":
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
        return self.name

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

    @staticmethod
    def parse(hardware_str) -> "Hardware":
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
        return Hardware(model=model, region=region)

    def region_match(self, other: "Cartridge | Hardware | Game") -> bool:
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

DEFAULT_REGIONS: list[Region] = [False, "INTL", "USA", "JPN", "EUR", "AUS", "KOR", "CHN"]
DEFAULT_LANGUAGES: dict[Region, list[Language]] = {False: [False, "European", "English", "Japanese"], "INTL": [False, "English"], "USA": [False, "English"], "JPN": [False, "Japanese"], "EUR": [False, "European", "English"], "AUS": [False, "English"], "KOR": [False, "Korean"], "CHN": [False, "Chinese"]}

GAMES: pd.DataFrame = pd.read_csv(Path('data') / 'games.csv').fillna(False)
GAMEIDS: set[str] = set(GAMES.GAMEID)
GAMEINPUTS: set[str] = set(GAMES.GAME)
REGIONS: set[Region] = set(GAMES.REGION)
LANGUAGES: set[Language] = set(GAMES.LANGUAGE)
EUROPEAN_LANGUAGES: set[Language] = {"English", "French", "German", "Italian", "Spanish"}
INTL_REGIONS: set[Region] = {"USA", "EUR", "AUS"}

HARDWARE_MODELS: dict[str, HardwareModel] = {h.name: h for h in [
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
        frozendict({"GB": ("GB", "GBC"), "GBCir": ("GB", "GBC")}),
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
    ), wireless=frozenset({"DS", "3DSir"})),
    # A 3DS that one is willing to factory reset in order to get all Friend Safari Pokemon
    HardwareModel("3DSr", frozenset({"JPN", "USA", "EUR", "KOR", "TWN", "CHN"}), True, frozenset({"3DS"}), (
        frozendict({"DS": ("DS", "3DS")}),
    ), wireless=frozenset({"DS", "3DSir"})),


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
    # Switch
    HardwareModel("SWITCH", is_console=True, software=frozenset({"SWITCH"}), ports=(
        frozendict({"DS": ("SWITCH",)}),
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
        frozendict({"GBA": ("GLC3p",)}),
        frozendict({"GBA": ("GCNc",)}),
    )),
]}

class Collection():
    def __init__(self, cartridges: list[Cartridge], hardware: set[Hardware], skip_validation=False):
        if len(cartridges) == 0:
            raise ValueError("No games!")
        main_cartridge: Optional[Cartridge] = None
        for cart in cartridges:
            if "NO_DEX" in cart.game.props:
                continue
            main_cartridge = cart
            break
        if main_cartridge is None:
            raise ValueError("No valid main game.")
        else:
            self.main_cartridge: Cartridge = main_cartridge

        self.cartridges = cartridges
        self.hardware = hardware

        self.hw_ports = {}
        self.hw_plugs = {}
        self.hw_wireless = {}
        self.carts_by_type = {}
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
            if cart.game.console not in self.carts_by_type:
                self.carts_by_type[cart.game.console] = set()
            self.carts_by_type[cart.game.console].add(cart)

        self._connected_cache = {}
        self.cart2consoles = {}
        for cart in self.cartridges:
            flow = cart.game.default_flow()
            consoles, connections = cart.connections(self, flow)
            if not consoles and cart.game.console is not None:
                raise ValueError(f"Game {cart} cannot be played with specified hardware.")
            self.cart2consoles[cart] = consoles
            if flow not in self._connected_cache:
                self._connected_cache[flow] = {}
            self._connected_cache[flow][cart] = connections

        self._init_interactions()

        if len(self.cartridges) > 1 and not skip_validation:
            unreachable = self.unreachable_games()
            if unreachable:
                raise ValueError(f"The following game(s) can't interact with the main game {self.main_cartridge}, at least with the current collection: {unreachable}")
     
    def connected(self, cart1: Cartridge, cart2: Cartridge, flow=None) -> bool:
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
        interactions = {
            "TRADE": set(),
            "POKEMON": set(),
            "ITEMS": set(),
            "MYSTERYGIFT": set(),
            "RECORDMIX": set(),
            "CONNECT": set(),
        }

        for cart1 in self.cartridges:
            game1 = cart1.game
            for cart2 in self.cartridges:
                if cart1 == cart2:
                    continue
                game2 = cart2.game

                # The cart1.console checks distinguish gen I/II physical games from VC games
                if game1.gen == 1 and game1.core and cart1.console is None:
                    if game2.gen == 1 and game2.core and cart2.console is None:
                        if game1.language_group(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name in {"STADIUM_JPN", "STADIUM"}:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["POKEMON"].add((cart1, cart2))
                    elif game2.gen == 2 and game2.core and cart2.console is None:
                        if game1.language_group(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "STADIUM2":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["POKEMON"].add((cart1, cart2))
                            # STADIUM2 is the only way to transfer items between gen 1 games
                            for cart3 in self.cartridges:
                                game3 = cart3.game
                                if game3.gen == 1 and game3.core and game3.console is None:
                                    if game1.language_match(game3) and self.connected(cart2, cart3):
                                        interactions["ITEMS"].add((cart1, cart3))
                elif game1.name in {"STADIUM_JPN", "STADIUM"}:
                    if game2.gen == 1 and game2.core and cart2.console is None:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["POKEMON"].add((cart1, cart2))
                elif game1.gen == 2 and game1.core and cart1.console is None:
                    if game2.gen == 1 and game2.core and cart2.console is None:
                        if game1.language_group(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                            # Pokemon holding items can be traded to gen 1. Gen 1 can't access the
                            # items, but they will be preserved if the Pokemon is later traded back
                            # to gen 2, including the same cartridge.
                            for cart3 in self.cartridges:
                                game3 = cart3.game
                                if game3.gen == 2 and game3.core and cart3.console is None:
                                    if game1.language_group(game3) and self.connected(cart2, cart3):
                                        interactions["ITEMS"].add((cart1, cart3))
                    elif game2.gen == 2 and game2.core and cart2.console is None:
                        if game1.language_group(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                        if game1.language_match(game2) and self.connected(cart1, cart2, "GBCir"):
                            interactions["MYSTERYGIFT"].add((cart1, cart2))
                    elif game2.name == "STADIUM2":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["POKEMON"].add((cart1, cart2))
                            interactions["ITEMS"].add((cart1, cart2))
                            interactions["MYSTERYGIFT"].add((cart1, cart1)) # self-transfer
                elif game1.name == "STADIUM2":
                    if game2.gen == 1 and game2.core and cart2.console is None:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["POKEMON"].add((cart1, cart2))
                    elif game2.gen == 2 and game2.core and cart2.console is None:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["POKEMON"].add((cart1, cart2))
                            interactions["ITEMS"].add((cart1, cart2))
                elif game1.gen == 3 and game1.core:
                    if game2.gen == 3 and game2.core:
                        if self.connected(cart1, cart2) or (
                                "GBA_WIRELESS" in game1.props and
                                "GBA_WIRELESS" in game2.props and
                                self.connected(cart1, cart2, "GBAw")
                        ):
                            interactions["TRADE"].add((cart1, cart2))
                            RSE = {"Ruby", "Sapphire", "Emerald"}
                            if game1.name in RSE and game2.name in RSE:
                                if game1.language == game2.language or (game1.name == "EMERALD" and game2.name == "EMERALD"):
                                    interactions["RECORDMIX"].add((cart1, cart2))
                    elif game2.name == "BOX":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["POKEMON"].add((cart1, cart2))
                            interactions["ITEMS"].add((cart1, cart2))
                    elif game2.name in ["COLOSSEUM", "XD"]:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "BONUSDISC":
                        if game1.name in ["RUBY", "SAPPHIRE"] and game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.name == "BONUSDISC_JPN":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.name == "CHANNEL" and game2.region == "EUR":
                        if game1.name in ["RUBY", "SAPPHIRE"] and game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.name == "EONTICKET":
                        if game1.name in ["RUBY", "SAPPHIRE"] and game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.gen == 4 and game2.core:
                        if (game1.language_match(game2) or game2.language == "Korean") and self.connected(cart1, cart2, "DS"):
                            # Pal Park: language requirement
                            interactions["POKEMON"].add((cart1, cart2))
                            interactions["ITEMS"].add((cart1, cart2))
                elif game1.name == "BOX":
                    if game2.gen == 3 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["POKEMON"].add((cart1, cart2))
                            interactions["ITEMS"].add((cart1, cart2))
                elif game1.name == "COLOSSEUM":
                    if game2.gen == 3 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "BONUSDISC_JPN":
                        # The Japanese bonus disc reads and edits the Colosseum save file, so they
                        # have to be compatible with the same GCN.
                        if game1.language_match(game2) and game1.region_match(game2):
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.name == "DOUBLEBATTLE":
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["CONNECT"].add((cart1, cart2))
                elif game1.name == "XD":
                    if game2.gen == 3 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                elif game1.gen == 4 and game1.core:
                    if game2.gen == 3 and game2.core:
                        if self.connected(cart1, cart2, "DS"):
                            # Dual-slot mode: no language requirement
                            interactions["CONNECT"].add((cart1, cart2))
                    if game2.gen == 4 and game2.core:
                        if game1.language_group(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                            interactions["RECORDMIX"].add((cart1, cart2))
                    elif game2.name == "POKEWALKER":
                        if game1.name in ["HEARTGOLD", "SOULSILVER"]:
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.name == "RANCH":
                        if game1.language_match(game2) and self.connected(cart1, cart2) and (
                                game1.name in ["DIAMOND", "PEARL"] or \
                                (game2.language == "Japanese" and game1.name == "PLATINUM")
                        ):
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.name in ["BATTLEREVOLUTION", "RANGER", "SHADOWOFALMIA", "GUARDIANSIGNS"]:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.gen == 5 and game2.core:
                        if game1.language_match(game2) and self.connected(cart1, cart2):
                            interactions["POKEMON"].add((cart1, cart2))
                elif game1.gen == 5 and game1.core:
                    if game2.gen == 5 and game2.core:
                        if self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "TRANSPORTER":
                        # Works with any region
                        interactions["POKEMON"].add((cart1, cart2))
                elif game1.name == "DREAMWORLD":
                    if game2.gen == 5 and game2.core:
                        if game1.region_match(game2):
                            interactions["CONNECT"].add((cart1, cart2))
                            interactions["POKEMON"].add((cart1, cart2))
                            interactions["ITEMS"].add((cart1, cart2))
                elif game1.name == "DREAMRADAR":
                    if game2.gen == 4 and game2.core:
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.name in ["BLACK2", "WHITE2"]:
                        if game1.region_match(game2):
                            interactions["POKEMON"].add((cart1, cart2))
                            interactions["ITEMS"].add((cart1, cart2))
                elif game1.gen == 6 and game1.core:
                    if game2.gen == 6 and game2.core:
                        if self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "ORAS_DEMO":
                        if game1.name in ["OMEGARUBY", "ALPHASAPPHIRE"] and (cart1.console is None and game1.region_match(game2)) or cart1.console == cart2.console: 
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.name == "BANK":
                        # Any region
                        if cart1.console is None or cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                            #interactions["CONNECT"].add((cart1, cart2))
                elif game1.name == "BANK":
                    if game2.gen == 6 and game2.core:
                        # Any region
                        if cart2.console is None or cart1.console == cart2.console:
                            # You can't transport VC or gen 7 Pokemon to gen 6 games, so treat BANK
                            # as enabling gen 6 games to transfer Pokemon between one another (and
                            # potentially receive Pokemon from gen 5 games)
                            has_transporter = any([c.game.name == "TRANSPORTER" and c.console == cart1.console for c in self.cartridges])
                            for cart3 in self.cartridges:
                                game3 = cart3.game
                                if (has_transporter and game3.gen == 5 and game3.core) or \
                                        (game3.gen == 6 and game3.core and (cart3.console is None or cart3.console == cart1.console)):
                                    interactions["POKEMON"].add((cart3, cart2))
                    elif game2.name in ["SUN", "MOON", "ULTRASUN", "ULTRAMOON"]:
                        if cart2.console is None or cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                    elif game2.name == "HOME":
                        interactions["POKEMON"].add((cart1, cart2))
                elif game1.name == "TRANSPORTER":
                    if game2.name == "BANK":
                        if cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                # VC games
                elif game1.gen == 1 and cart1.console is not None:
                    if game2.gen == 1 and cart2.console is not None:
                        if game1.language_group(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.gen == 2 and game2.core and cart2.console is not None:
                        if game1.language_group(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "TRANSPORTER":
                        if cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                elif game1.gen == 2 and cart1.console is not None:
                    if game2.gen == 1 and game2.core and cart2.console is not None:
                        if game1.language_group(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                            # As before, you can effectively use a gen 1 game to trade items between gen 2s
                            for cart3 in self.cartridges:
                                game3 = cart3.game
                                if game3.gen == 2 and game3.core and cart3.console is not None:
                                    if game1.language_group(game3) and self.connected(cart2, cart3):
                                        interactions["ITEMS"].add((cart1, cart3))
                    elif game2.gen == 2 and game2.core and cart2.console is not None:
                        if game1.language_group(game2) and self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                        # Adding this for completeness; in practice on the VC, you can never MG if
                        # you can't trade, and MG adds no practical benefit over trading
                        if game1.language_match(game2) and self.connected(cart1, cart2, "3DSir"):
                            interactions["MYSTERYGIFT"].add((cart1, cart2))
                    elif game2.name == "TRANSPORTER":
                        if cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                elif game1.name in ["SUN", "MOON", "ULTRASUN", "ULTRAMOON"]:
                    if game2.name in ["SUN", "MOON", "ULTRASUN", "ULTRAMOON"]:
                        if self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "SM_DEMO":
                        if game1.name in ["SUN", "MOON"] and (cart1.console is None and game1.region_match(game2)) or cart1.console == cart2.console:
                            interactions["CONNECT"].add((cart1, cart2))
                    elif game2.name == "BANK":
                        # Any region
                        if cart1.console is None or cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                            interactions["CONNECT"].add((cart1, cart2))
                elif game1.name in ["LETSGOPIKACHU", "LETSGOEEVEE"]:
                    if game2.name in ["LETSGOPIKACHU", "LETSGOEEVEE"]:
                        if self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "HOME":
                        if cart1.console is None or cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                        # You can transfer to other LETSGO games via HOME
                        for cart3 in self.cartridges:
                            game3 = cart3.game
                            if game3.name in ["LETSGOPIKACHU", "LETSGOEEVEE"]:
                                if cart3.console == cart2.console:
                                    interactions["POKEMON"].add((cart1, cart3))
                elif game1.name in ["SWORD", "SHIELD"]:
                    if game2.name in ["SWORD", "SHIELD"]:
                        if self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "HOME":
                        if cart1.console is None or cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                    elif game2.name in ["LETSGOPIKACHU", "LETSGOEEVEE", "SWORD_DLC", "SHIELD_DLC"]:
                        if cart1.console is None or cart2.console is None or cart1.console == cart2.console:
                            interactions["CONNECT"].add((cart1, cart2))
                elif game1.name in ["BRILLIANTDIAMOND", "SHININGPEARL"]:
                    if game2.name in ["BRILLIANTDIAMOND", "SHININGPEARL"]:
                        if self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "HOME":
                        if cart1.console is None or cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                    elif game2.name in ["LETSGOPIKACHU", "LETSGOEEVEE", "SWORD", "SHIELD", "LEGENDSARCEUS"]:
                        if cart1.console is None or cart2.console is None or cart1.console == cart2.console:
                            interactions["CONNECT"].add((cart1, cart2))
                elif game1.name == "LEGENDSARCEUS":
                    if game2.name == "LEGENDSARCEUS":
                        if self.connected(cart1, cart2):
                            interactions["TRADE"].add((cart1, cart2))
                    elif game2.name == "HOME":
                        if cart1.console is None or cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                    elif game2.name in ["SWORD", "SHIELD", "BRILLIANTDIAMOND", "SHININGPEARL"]:
                        if cart1.console is None or cart2.console is None or cart1.console == cart2.console:
                            interactions["CONNECT"].add((cart1, cart2))
                elif game1.name == "HOME":
                    if game2.gen == 8 and game2.core:
                        if cart2.console is None or cart1.console == cart2.console:
                            interactions["POKEMON"].add((cart1, cart2))
                            interactions["CONNECT"].add((cart1, cart2))
                elif game1.name == "POKEBALLPLUS":
                    if game2.name in ["LETSGOPIKACHU", "LETSGOEEVEE", "SWORD", "SHIELD"]:
                        interactions["POKEMON"].add((cart1, cart2))
                        
       
        self.interactions = interactions

    def interactions_graph(self) -> DiGraph:
        G = nx.DiGraph()
        for cart in self.cartridges:
            G.add_node(cart)
        for kind, cart_pairs in self.interactions.items():
            for cart1, cart2 in cart_pairs:
                if kind == "CONNECT":
                    G.add_edge(cart2, cart1)
                else:
                    G.add_edge(cart1, cart2)
        return G

    def unreachable_games(self) -> set[Cartridge]:
        unreachable = set()
        if len(self.cartridges) == 1:
            return unreachable
        G = self.interactions_graph()
        for cart in self.cartridges:
            if cart == self.main_cartridge:
                continue
            if not nx.has_path(G, cart, self.main_cartridge):
                unreachable.add(cart)
            elif cart.game.name == "BONUSDISC_JPN":
                colosseum = [c for c in G.neighbors(cart) if c.game.name == "COLOSSEUM"]
                if not colosseum:
                    unreachable.add(cart)
            elif cart.game.name == "TRANSPORTER":
                transported = G.predecessors(cart)
                if self.main_cartridge.game.gen == 6 and self.main_cartridge.game.core:
                    # Can't deposit Pokemon from VC games to gen 6 games
                    transported = [t for t in transported if t.game.gen == 5]
                if not transported:
                    unreachable.add(cart)
        return unreachable


    def friend_safari_consoles(self) -> dict[Hardware, dict[Cartridge, float]]:
        xy_carts = {cart for cart in self.cartridges if cart.game.name in {"X", "Y"}}
        gen6_carts = {cart for cart in self.cartridges if cart.game.gen == 6 and cart.game.core}
        consoles = {hw for hw in self.hardware if hw.model.name in {"3DS", "3DSr"}}
        safaris = {hw: {} for hw in consoles}
        if not xy_carts:
            return safaris
        console2carts = {hw: set() for hw in consoles}
        for cart in gen6_carts:
            for console in self.cart2consoles[cart]:
                console2carts[console].add(cart)

        for console in consoles:
            for cart in xy_carts:
                # It must be possible to play cart on a different console
                if not self.cart2consoles[cart].difference({console}):
                    continue
                # If there is a cart other than the current one that can be played on the console,
                # we can unlock the third safari slot.
                other_carts = console2carts[console].difference({cart})

                if other_carts:
                    if console.model.name != "3DSr" or any(c.console is None for c in other_carts):
                        safaris[console][cart] = 3
                    else:
                        # Third safari slot is only possible until reset
                        safaris[console][cart] = 2.5
                else:
                    safaris[console][cart] = 2

        return safaris


    def reset_consoles(self) -> dict[Hardware, set[Cartridge]]:
        out = {}
        consoles = {hw for hw in self.hardware if hw.model.name == "3DSr"}
        for console in consoles:
            software_carts = set(cart for cart in self.cartridges if cart.console == console)
            if software_carts:
                out[console] = software_carts
        return out


        
