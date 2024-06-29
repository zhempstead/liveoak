#!/usr/bin/python

from dataclasses import replace
from itertools import zip_longest
from pathlib import Path
from tkinter import W
from typing import Iterable

import pandas as pd

from structs import Pokemon

class MultiResult():
    def __init__(self, results: list["Result"], use_forms: bool, version_exclusive_pairs: list[tuple[Pokemon, Pokemon]] = []):
        self.results = results
        self.all_pokemon: list[Pokemon] = sorted(set.union(*(set(r.all_pokemon) for r in self.results)))
        self.all_present: set[Pokemon] = set()
        self.all_missing: set[Pokemon] = set()
        for pokemon in self.all_pokemon:
            line = self.results[0].line(pokemon)
            if all(result.line(pokemon) == line for result in self.results[1:]):
                if line is None:
                    self.all_missing.add(pokemon)
                else:
                    self.all_present.add(pokemon)
        
        self.version_exclusives: dict[Pokemon, set[Pokemon]] = {}
        for pokemon1, pokemon2 in version_exclusive_pairs:
            if not use_forms:
                pokemon1 = replace(pokemon1, form=None)
                pokemon2 = replace(pokemon2, form=None)
                if pokemon1 == pokemon2:
                    continue
            if pokemon1 not in self.all_pokemon or pokemon2 not in self.all_pokemon:
                continue
            present_patterns = {(pokemon1 in r.present, pokemon2 in r.present) for r in self.results}
            if (True, False) in present_patterns and (False, True) in present_patterns and (True, True) not in present_patterns:
                if pokemon1 not in self.version_exclusives:
                    self.version_exclusives[pokemon1] = set()
                if pokemon2 not in self.version_exclusives:
                    self.version_exclusives[pokemon2] = set()
                self.version_exclusives[pokemon1].add(pokemon2)
                self.version_exclusives[pokemon2].add(pokemon1)

    def full_group(self, pokemon: Pokemon) -> set[Pokemon]:
        processed: set[Pokemon] = set()
        to_process = {pokemon}
        while to_process:
            pokemon = to_process.pop()
            processed.add(pokemon)
            ve_opposites = self.version_exclusives.get(pokemon)
            if ve_opposites is not None:
                for opp_pokemon in ve_opposites:
                    if opp_pokemon not in processed:
                        to_process.add(opp_pokemon)
                
            for result in self.results:
                group = result.pokemon2group.get(pokemon)
                if group is None:
                    continue
                for sub_result in result.group2pokes[group]:
                    if sub_result not in processed:
                        to_process.add(sub_result)
        return processed


    def _group_lines(self, pokes: list[Pokemon], obtainable=True) -> list[tuple[str, ...]]:
        lines: list[list[str]] = []
        pokemon_set = set(pokes)
        for r in self.results:
            pokes_ordered = [o for o in r.order() if o in pokemon_set]
            lines.append(r._lines(r.present if obtainable else r.missing, pokes_ordered))
        return list(zip_longest(*lines, fillvalue=""))


    def print(self, obtainable=True, skip_identical=True):
        vline = ["-" for _ in self.results]
        handled_pokes: set[Pokemon] = set()
        lines = []
        for pokemon in self.all_pokemon:
            if pokemon in handled_pokes:
                continue
            if pokemon in self.all_missing and skip_identical:
                continue
            if pokemon in self.all_present and skip_identical:
                continue
            
            sub_pokes = self.full_group(pokemon)
            if skip_identical:
                sub_pokes = sub_pokes.difference(self.all_missing).difference(self.all_present)
            sub_pokes = sorted(sub_pokes)
            group_lines = self._group_lines(sub_pokes, obtainable)
            if len(group_lines) > 1:
                if lines and lines[-1] != vline:
                    lines.append(vline)
                lines += group_lines
                lines.append(vline)
            else:
                lines += group_lines
            handled_pokes = handled_pokes.union(sub_pokes)
        if lines and lines[-1] != vline:
            lines.append(vline)
        lines.append([str(r.count()) for r in self.results])
        self._print_lines(lines)

    def print_compact(self, obtainable=True, skip_identical=True):
        skip = self.all_present | self.all_missing
        if obtainable:
            lines = [r._lines(r.present, [p for p in r.order() if p not in skip or not skip_identical]) for r in self.results]
        else:
            lines = [r._lines(r.missing, [p for p in r.order() if p not in skip or not skip_identical]) for r in self.results]
        lines_filled: list[tuple[str, ...]] = list(zip_longest(*lines, fillvalue=''))
        lines_filled.append(tuple(str(r.count()) for r in self.results))
        self._print_lines(lines_filled)

    def print_all_present(self):
        r = self.results[0]
        print("\n".join(self.results[0]._lines(self.results[0].present, sorted(self.all_present))))

    def _print_lines(self, lines):
        max_lens = [max([len(l[j]) for l in lines]) for j in range(len(lines[0]))]
        lines = [[l[j] + ("-" if l[j] == "-" else " ")*(max_lens[j] - len(l[j])) for j in range(len(l))] for l in lines]
        lines = [('-|-' if l[0].startswith('--') else ' | ').join(l) for l in lines]
        print("\n".join(lines))
        
type Row = list[Pokemon | None]
class Result():
    def __init__(self, all_pokemon: list[Pokemon], present: dict[Pokemon, Row], missing: dict[Pokemon, Row], pokemon2group: dict[Pokemon, int]):
        self.all_pokemon: list[Pokemon] = all_pokemon
        self.present = present
        self.missing = missing
        self.pokemon2group = pokemon2group
        self.max_idx = max(max(de.idx for de in present.keys()), max(de.idx for de in missing.keys()))
        self.group2pokes: dict[int, set[Pokemon]] = {}
        for pokemon, group_pokemon in self.pokemon2group.items():
            if group_pokemon not in self.group2pokes:
                self.group2pokes[group_pokemon] = set()
            self.group2pokes[group_pokemon].add(pokemon)

    @staticmethod
    def new(all_pokes: Iterable[Pokemon], present_set: set[Pokemon], branches: list[set[frozenset[Pokemon]]]) -> "Result":
        present: dict[Pokemon, Row] = {p: [p] for p in present_set}
        missing: dict[Pokemon, Row] = {}
        pokemon2group: dict[Pokemon, int] = {}
        not_missing = set(present.keys())
        next_gidx = 0
        for branch in branches:
            # Replace species with idxs
            branch = [sorted(choice) for choice in branch]
            if len(branch) > 12:
                branch = Result._curtail_branch_output(branch)
            branch = sorted(branch, key=lambda choice: (-len(choice), choice))
            all_branch_pokes = {idx for choice in branch for idx in choice}
            not_missing = not_missing.union(all_branch_pokes)
            to_update: list[tuple[list[list[Pokemon]], dict[Pokemon, Row], bool]] = [(branch, present, True), (Result._present2missing(branch), missing, False)]
            for b, p_or_m, add_gidx in to_update:
                if add_gidx:
                    gidx = next_gidx
                    for dex_pokemon in all_branch_pokes:
                        pokemon2group[dex_pokemon] = gidx
                    next_gidx += 1
                max_choice_len = len(b[0])
                padded: list[Row] = [choice + [None]*(max_choice_len - len(choice)) for choice in b]
                for pos in range(max_choice_len):
                    row: Row = [choice[pos] for choice in padded]
                    assert(row[0] is not None)
                    p_or_m[row[0]] = row
        for pokemon in set(all_pokes) - not_missing:
            missing[pokemon] = [pokemon]

        return Result(sorted(all_pokes), present, missing, pokemon2group)

    @staticmethod
    def _curtail_branch_output(orig_branch: list[list[Pokemon]]) -> list[list[Pokemon]]:
        '''
        Reduce a large number of possibilities to a smaller set where
        - every pokemon is in at least one branch
        - every pokemon is *missing* from at least one branch
        - the branch with the most pokes is guaranteed to be present (if a tie, at least one will be)
        '''
        orig_branch = [choice.copy() for choice in orig_branch]
        all_pokes: frozenset[Pokemon] = frozenset.union(*[frozenset(choice) for choice in orig_branch])
        not_present: frozenset[Pokemon] = all_pokes.copy()
        not_missing: frozenset[Pokemon] = all_pokes.copy()
        final_branch: list[list[Pokemon]] = []
        branch = [(choice, set(choice), all_pokes - set(choice)) for choice in orig_branch]
        while not_present or not_missing:
            # Pick the one with the most new species
            # If a tie, favor the one that avoids the most species that have been in every choice so far
            # If a tie, favor the one with the most species
            # If a tie, favor those with lower idxs
            best, _, missing = min(branch, key=lambda c: (-len(c[1]), -len(c[2]), -len(c[0]), c[0]))
            best_set = set(best)
            final_branch.append(best)
            not_present = not_present - best_set
            not_missing = not_missing & best_set
            branch = [(c[0], c[1] - best_set, c[2] - missing) for c in branch]
            branch = [c for c in branch if c[1] or c[2]]
        return final_branch

    @staticmethod
    def _present2missing(branch: list[list[Pokemon]]) -> list[list[Pokemon]]:
        all_idxs = {idx for choice in branch for idx in choice}
        inverse_branch = sorted(
            [sorted(list(all_idxs.difference(choice))) for choice in branch],
            key=lambda choice: (-len(choice), choice)
        )
        return inverse_branch

    def out_str(self, pokemon: Pokemon | None) -> str:
        if pokemon is None:
            return "-"
        return str(pokemon)

    def line(self, pokemon: Pokemon) -> str | None:
        pokes = self.present.get(pokemon)
        if not pokes:
            return None
        return " / ".join([self.out_str(e) for e in pokes])

    def _lines(self, source: dict[Pokemon, Row], pokes: list[Pokemon]) -> list[str]:
        group_widths: dict[int, list[int]] = {}
        for pokemon in pokes:
            group = self.pokemon2group.get(pokemon)
            if group is None:
                continue
            row = source.get(pokemon)
            if not row:
                continue
            lengths = [len(str(e)) if e else 0 for e in row]

            if group in group_widths:
                for j in range(len(lengths)):
                    group_widths[group][j] = max(group_widths[group][j], lengths[j])
            else:
                group_widths[group] = lengths

        out: list[str] = []
        for pokemon in pokes:
            group = self.pokemon2group.get(pokemon)
            if group is None:
                if pokemon in source:
                    out.append(str(pokemon))
                continue
            row = source.get(pokemon)
            if not row:
                continue
            line = zip([self.out_str(e) for e in row], group_widths[group])
            line = [text + " "*(w - len(text)) for text, w in line]
            out.append(" / ".join(line))

        return out

    def order(self) -> list[Pokemon]:
        handled_groups: set[int] = set()
        order: list[Pokemon] = []
        for pokemon in self.all_pokemon:
            group = self.pokemon2group.get(pokemon)
            if group is None:
                order.append(pokemon)
                continue
            if group in handled_groups:
                continue
            order += sorted(self.group2pokes[group])
            handled_groups.add(group)
        return order

    def print_obtainable(self):
        print("\n".join(self._lines(self.present, self.order())))

    def print_missing(self):
        print("\n".join(self._lines(self.missing, self.order())))

    def count(self) -> int:
        return len(self._lines(self.present, self.order()))
