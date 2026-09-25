"""Appendix Figure 2 radars with the full floor family of the text: six sequence floors and all ten floors.

make_figures_paper.fig2_radars draws the released sweep against make_panels_original_style's floor lists, which hold
four sequence floors. This wrapper sets the lists to the six sequence floors and ten floors of Section 4.1 and redraws
only the radar figure.

  python make_figure2_radars_sixfloors.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import make_figures_paper as F                                           # noqa: E402

SIX = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global"]
F.P.SEQ_FLOORS = SIX
F.P.ALL_FLOORS = SIX + ["ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]

if __name__ == "__main__":
    fam, groups, seq_floor, all_floor = F._probe_matrix()
    print("floors used:", len(F.P.SEQ_FLOORS), "sequence,", len(F.P.ALL_FLOORS), "in all")
    F.fig2_radars()
