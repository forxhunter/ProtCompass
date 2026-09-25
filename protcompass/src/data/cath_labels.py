"""
Utility to parse CATH domain list file.
"""
from pathlib import Path
from typing import Dict, Tuple

def load_cath_labels(path: str) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Load CATH labels from domain list file.
    
    Args:
        path: Path to cath-domain-list-v4_3.txt
        
    Returns:
        Dictionary mapping cath_id -> (Class, Architecture, Topology, Homology)
    """
    labels = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
                
            cath_id = parts[0]
            # C, A, T, H are indices 1, 2, 3, 4
            c = int(parts[1])
            a = int(parts[2])
            t = int(parts[3])
            h = int(parts[4])
            
            labels[cath_id] = (c, a, t, h)
            
    return labels

def get_unique_classes(labels: Dict[str, Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Get number of unique classes for C, A, T, H."""
    c_set, a_set, t_set, h_set = set(), set(), set(), set()
    
    for c, a, t, h in labels.values():
        c_set.add(c)
        a_set.add((c, a))  # Architecture is hierarchical
        t_set.add((c, a, t)) # Topology is hierarchical
        h_set.add((c, a, t, h))
        
    return len(c_set), len(a_set), len(t_set), len(h_set)
