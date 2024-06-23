#!/usr/bin/python

from itertools import zip_longest
from pathlib import Path

import pandas as pd

class MultiResult():
    def __init__(self, results: list["Result"], pokemon2idx: dict[str, int], match_version_exclusives: bool):
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
                

    def full_group(self, idx: int) -> set[int]:
        processed: set[int] = set()
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


    def _group_lines(self, idxs: list[int], obtainable=True) -> list[tuple[str, ...]]:
        lines: list[list[str]] = []
        idx_set = set(idxs)
        for r in self.results:
            idxs_ordered = [o for o in r.order() if o in idx_set]
            lines.append(r._lines(r.present if obtainable else r.missing, idxs_ordered))
        #lines = [r._lines(r.present if obtainable else r.missing, idxs) for r in self.results]
        #import pdb; pdb.set_trace()
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

    # TODO: fix - this prints all rows
    def print_compact(self, obtainable=True):
        if obtainable:
            lines = [r._lines(r.present, r.order()) for r in self.results]
        else:
            lines = [r._lines(r.missing, r.order()) for r in self.results]
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
        

class Result():
    def __init__(self, present: dict[int, list[int]], missing: dict[int, list[int]], idx2gidx: dict[int, int], idx2pokemon: dict[int, str]):
        self.present = present
        self.missing = missing
        self.idx2gidx = idx2gidx
        self.idx2pokemon = idx2pokemon
        self.max_idx = max(idx2pokemon.keys())
        self.gidx2idxs: dict[int, set[int]] = {}
        for idx, gidx in self.idx2gidx.items():
            if gidx not in self.gidx2idxs:
                self.gidx2idxs[gidx] = set()
            self.gidx2idxs[gidx].add(idx)

    @staticmethod
    def new(idx2pokemon: dict[int, str], present_set: set[str], branches: list[set[frozenset[str]]]):
        pokemon2idx = {v: k for k, v in idx2pokemon.items()}
        missing: dict[int, list[int]] = {}
        idx2gidx: dict[int, int] = {}
        present = {pokemon2idx[p]: [pokemon2idx[p]] for p in present_set}
        not_missing = set(present.keys())
        next_gidx = 0
        for branch in branches:
            # Replace species with idxs
            branch = [sorted([pokemon2idx[p] for p in choice]) for choice in branch]
            if len(branch) > 12:
                branch = Result._curtail_branch_output(branch)
            branch = sorted(branch, key=lambda choice: (-len(choice), choice))
            all_idxs = {idx for choice in branch for idx in choice}
            not_missing = not_missing.union(all_idxs)
            for b, p_or_m, add_gidx in [(branch, present, True), (Result._present2missing(branch), missing, False)]:
                if add_gidx:
                    gidx = next_gidx
                    for idx in all_idxs:
                        idx2gidx[idx] = gidx
                    next_gidx += 1
                max_choice_len = len(b[0])
                padded = [choice + [None]*(max_choice_len - len(choice)) for choice in b]
                for pos in range(max_choice_len):
                    pos_idxs = [choice[pos] for choice in padded]
                    p_or_m[pos_idxs[0]] = pos_idxs
        for idx in set(idx2pokemon.keys()).difference(not_missing):
            missing[idx] = [idx]

        return Result(present, missing, idx2gidx, idx2pokemon)

    @staticmethod
    def _curtail_branch_output(orig_branch: list[list[int]]) -> list[list[int]]:
        '''
        Reduce a large number of possibilities to a smaller set where
        - every entry is in at least one branch
        - every entry is *missing* from at least one branch
        - the branch with the most entries is guaranteed to be present (if a tie, at least one will be)
        '''
        orig_branch = [choice.copy() for choice in orig_branch]
        all_entries: frozenset[int] = frozenset.union(*[frozenset(choice) for choice in orig_branch])
        not_present: frozenset[int] = all_entries.copy()
        not_missing: frozenset[int] = all_entries.copy()
        final_branch: list[list[int]] = []
        branch = [(choice, set(choice), all_entries - set(choice)) for choice in orig_branch]
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
    def _present2missing(branch: list[list[int]]) -> list[list[int]]:
        all_idxs = {idx for choice in branch for idx in choice}
        inverse_branch = sorted(
            [sorted(list(all_idxs.difference(choice))) for choice in branch],
            key=lambda choice: (-len(choice), choice)
        )
        return inverse_branch

    def obtainable_line(self, idx: int) -> str | None:
        if idx not in self.present:
            return None
        return " / ".join([self.entry(i) for i in self.present[idx]])

    def entry(self, idx: int) -> str:
        if idx is None:
            return "-"
        return f"{idx:04}. {self.idx2pokemon[idx]}"

    def line(self, idx: int) -> str | None:
        sub_idxs = self.present.get(idx)
        if not sub_idxs:
            return None
        return " / ".join([self.entry(sub_idx) for sub_idx in sub_idxs])

    def _lines(self, source: dict[int, list[int]], idxs: list[int]) -> list[str]:
        group_widths: dict[int, list[int]] = {}
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

        out: list[str] = []
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

    def order(self) -> list[int]:
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

    def count(self) -> int:
        return len(self._lines(self.present, self.order()))
