from dataclasses import dataclass, replace
from enum import Enum
from typing import Self

class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class Pokemon():
    '''Represents a unique species/form combination'''
    species: str
    form: str | None
    dex_form: str | None # For visually-indistinct forms
    idx: int
    form_idx: int
    dex_form_idx: int

    @staticmethod
    def new(species, form, idx, form_idx) -> "Pokemon":
        if not form:
            return Pokemon(species, None, None, idx, form_idx, form_idx)
        split = form.split(';')
        if len(split) == 1:
            return Pokemon(species, form, form, idx, form_idx, form_idx)
        return Pokemon(species, split[0], split[2] or None, idx, form_idx, int(split[1]))

    def __post_init__(self):
        if self.form is not None and not self.form:
            object.__setattr__(self, 'form', None)
    
    def __lt__(self, other: "Pokemon") -> bool:
        return (self.idx, self.form_idx) < (other.idx, other.form_idx)

    def __str__(self):
        out = f"{self.idx:04}. {self.species}"
        if self.form is not None:
            out += f" ({self.form})"
        return out

@dataclass(frozen=True)
class PokemonReq():
    '''Represents a unique species/form combination with possible gender/props requirements'''
    species: str
    form: str | None = None
    gender: Gender | None = None
    required: frozenset = frozenset()
    forbidden: frozenset = frozenset([])

    def matches(self, pokemon_entry):
        if self.species != pokemon_entry.species:
            return False
        if self.form != pokemon_entry.form:
            return False
        if self.gender is not None and pokemon_entry.gender != self.gender:
            return False
        if not self.required.issubset(pokemon_entry.props):
            return False
        if self.forbidden.intersection(pokemon_entry.props):
            return False
        return True
 
    def __str__(self):
        out = self.species
        if self.form:
            out += f' ({self.form})'
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
class Entry():
    '''Base class for an acquirable Pokemon/item/game state'''
    cart_id: str

@dataclass(frozen=True)
class PokemonEntry(Entry):
    '''A particular acquirable Pokemon with set gender/props'''
    species: str
    form: str | None
    gender: Gender | None
    props: frozenset[str] = frozenset([])

    @staticmethod
    def new(cart_id: str, pokemon: Pokemon, gender=None, props: set[str] | frozenset[str]=set()):
        return PokemonEntry(cart_id, pokemon.species, pokemon.form, gender, frozenset(props))

    def __str__(self):
        out = self.species
        if self.form is not None:
            out += f" ({self.form})"
        if self.gender == Gender.MALE:
            out += '(♂)'
        elif self.gender == Gender.FEMALE:
            out += '(♀)'

        if self.props:
            out += ' {'
            out += ', '.join(self.props)
            out += '}'
        out += f'[{self.cart_id}]'
        return out

    __repr__ = __str__

@dataclass(frozen=True)
class ItemEntry(Entry):
    item: str

    def __repr__(self):
        return f"ItemEntry({self.cart_id}, {self.item})"

@dataclass(frozen=True)
class ChoiceEntry(Entry):
    choice: str

    def __repr__(self):
        return f"ChoiceEntry({self.cart_id}, {self.choice})"

@dataclass(frozen=True)
class DexEntry(Entry):
    idx: int
    species: str
    
    def to_pokemon(self) -> Pokemon:
        return Pokemon(self.species, None, None, self.idx, 0, 0)

@dataclass(frozen=True)
class SpeciesDexEntry(DexEntry):
    @staticmethod
    def new(cart_id: str, pokemon: Pokemon):
        return SpeciesDexEntry(cart_id, pokemon.idx, pokemon.species)

    def __repr__(self):
        return f"SpeciesDexEntry({self.cart_id}, {self.species})"

@dataclass(frozen=True)
class FormDexEntry(DexEntry):
    form: str | None
    form_idx: int

    @staticmethod
    def new(cart_id: str, pokemon: Pokemon):
        return FormDexEntry(cart_id, pokemon.idx, pokemon.species, pokemon.dex_form, form_idx=pokemon.dex_form_idx)

    def to_pokemon(self) -> Pokemon:
        # It's okay that this doesn't roundtrip regarding form and dex_form
        return Pokemon(self.species, self.form, self.form, self.idx, self.form_idx, self.form_idx)

    def __repr__(self):
        pokemon = self.species
        if self.form:
            pokemon += f' ({self.form})'
        return f"FormDexEntry({self.cart_id}, {pokemon})"

type Req = PokemonReq | Entry

@dataclass(frozen=True)
class Rule():
    consumed: frozenset[Entry]
    required: frozenset[Entry]
    output: frozenset[Entry]
    is_transfer: bool = False
    can_explore: bool = True

    def __repr__(self) -> str:
        consumed = set(self.consumed) or '{}'
        required = set(self.required) or '{}'
        output = set(self.output)
        return f"{'T' if self.is_transfer else ''}{'E' if self.can_explore else ''}{consumed}{required}->{output}"

    def in_cart_ids(self) -> set[str]:
        return {c.cart_id for c in self.consumed} | {r.cart_id for r in self.required}

    def out_cart_ids(self) -> set[str]:
        return {o.cart_id for o in self.output}

    def replace_in_cart_ids(self, old_cart_id: str, new_cart_id: str) -> Self:
        new_consumed = frozenset({
            replace(c, cart_id=new_cart_id) if c.cart_id == old_cart_id else c
            for c in self.consumed})
        new_required = frozenset({
            replace(r, cart_id=new_cart_id) if r.cart_id == old_cart_id else r
            for r in self.required})
        return replace(self, consumed=new_consumed, required=new_required)

    def replace_out_cart_ids(self, old_cart_id: str, new_cart_id: str) -> Self:
        new_output = frozenset({
            replace(o, cart_id=new_cart_id) if o.cart_id == old_cart_id else o
            for o in self.output})
        return replace(self, output=new_output)