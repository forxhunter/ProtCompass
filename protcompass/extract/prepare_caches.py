"""Parse the stage-1 structures of each task once, before the extraction queues start."""
import sys
import pandas as pd
from common import HERE
from extract import coords_cache
for task in sys.argv[1:]:
    man = pd.read_csv(HERE / "manifest" / f"{task}.csv"); man = man[man.stage1 == 1].reset_index(drop=True)
    xyz, off = coords_cache(task, man)
    print(task, len(man), "proteins,", len(xyz), "residues", flush=True)
