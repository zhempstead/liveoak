import os
import tempfile

cases = [
    ('gen1_red', 'GBC RED'),
    ('gen1_redblue', 'GBC GBC CGB-003 RED BLUE'),
    ('gen1_yellow', 'GBC YELLOW'),
    ('gen1_yellow_2', 'GBC GBC CGB-003 YELLOW YELLOW'),
    ('gen2_silver_red', 'GBC GBC CGB-003 . RED SILVER . SILVER RED'),
    ('gen2_korean', 'GBC . SILVER . SILVER.Korean'),
    ('gen2_ir', 'GBC . GOLD . GOLD GBC SILVER'),
    ('gen3_noditto', 'GBA DOL-011 RUBY GCN COLOSSEUM'),
    ('gen3_xd', 'GCN XD'),
    ('gen3_johto', 'GBA GBA AGB-005 . EMERALD RUBY . EMERALD SAPPHIRE . EMERALD RUBY SAPPHIRE'),
    ('gen4_dualslot', 'DS . PEARL . PEARL SAPPHIRE.Japanese . PEARL SAPPHIRE'),
    ('gen5_wf', 'DS DS WHITE . . WHITE'),
]

def check_and_replace(fname, command):
    print(fname, command)
    with tempfile.NamedTemporaryFile() as tfile:
        assert os.system(f"{command} > {tfile.name}") == 0, command
        if os.system(f"diff {fname} {tfile.name}") != 0:
            os.system(f"cp {tfile.name} {fname}")
            assert False, command

# Need this for scope reasons
def lambda_factory(name, args):
    return lambda: check_and_replace(f"artifacts/{name}.txt", f"python main.py {args}")

for name, args in cases:
    globals()[f"test_{name}"] = lambda_factory(name, args)
