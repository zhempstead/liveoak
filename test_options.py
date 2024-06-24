import os
import tempfile

import pytest

def check_and_replace(fname, command, expect_fail=False):
    print(fname, command)
    with tempfile.NamedTemporaryFile() as tfile:
        code = os.system(f"{command} > {tfile.name}")
        if expect_fail:
            assert code != 0, command
            return
        assert os.system(f"{command} > {tfile.name}") == 0, command
        if os.system(f"diff {fname} {tfile.name}") != 0:
            os.system(f"cp {tfile.name} {fname}")
            assert False, command

@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('all_present', [False, True])
@pytest.mark.parametrize('flatten', [False, True])
@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('version_exclusive', [False, True])
@pytest.mark.parametrize('missing', [False, True])
def test_options(full, all_present, flatten, compact, version_exclusive, missing):
    command = ['GBA . RED . YELLOW']
    fname = ["args"]
    expect_fail = False
    if all_present and (full or compact or version_exclusive or missing):
        expect_fail = True
    if compact and version_exclusive:
        expect_fail = True

    if full:
        command.append('--full')
        fname.append('full')
    if all_present:
        command.append('--all-present')
        fname.append('allpresent')
    if flatten:
        command.append('--flatten')
        fname.append('flatten')
    if compact:
        command.append('--compact')
        fname.append('compact')
    if version_exclusive:
        command.append('--version-exclusive')
        fname.append('versionexclusive')
    if missing:
        command.append('--missing')
        fname.append('missing')
    check_and_replace(f"artifacts/{'_'.join(fname)}.txt", f"python main.py {' '.join(command)}", expect_fail)
