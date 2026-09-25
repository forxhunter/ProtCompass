"""
IPA Encoder worker for tf2_ipa conda env.

Uses the standard invariant-attention library (InvariantPointAttention)
to implement AlphaFold2-style Invariant Point Attention.

This runs inside the `tf2_ipa` conda environment via subprocess.
"""
import json
import sys
import numpy as np
import tensorflow as tf

# ── IPA encoder ───────────────────────────────────────────────────────────────


def build_local_frames(n_pos, ca_pos, c_pos):
    """
    Build orthonormal local reference frames from backbone atoms.
    Convention (matches AlphaFold2):
        x-axis: CA → C (normalized)
        y-axis: perpendicular component of CA → N
        z-axis: cross(x, y) → right-handed frame

    Args:
        n_pos: (b, n, 3) nitrogen positions
        ca_pos: (b, n, 3) CA positions
        c_pos: (b, n, 3) carbon positions

    Returns:
        rotations: (b, n, 3, 3) right-handed rotation matrices
        translations: (b, n, 3) frame origins (CA positions)
    """
    v1 = c_pos - ca_pos
    v2 = n_pos - ca_pos

    e1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    v2_proj = v2 - (np.sum(v2 * e1, axis=-1, keepdims=True)) * e1
    e2 = v2_proj / (np.linalg.norm(v2_proj, axis=-1, keepdims=True) + 1e-8)
    e3 = np.cross(e1, e2, axis=-1)

    rotations = np.stack([e1, e2, e3], axis=-1).astype(np.float32)
    translations = ca_pos.astype(np.float32)
    return rotations, translations


def build_ipa_encoder(
    num_res,
    hidden_dim=128,
    num_layers=4,
    num_heads=8,
    scalar_key_dim=16,
    scalar_value_dim=16,
    point_key_dim=4,
    point_value_dim=4,
    pairwise_repr_dim=64,
    dropout=0.1,
):
    """
    Build an IPA encoder model using the standard invariant-attention library.

    Takes precomputed rotations/translations as inputs (computed in numpy
    outside the Keras graph to avoid symbolic shape issues).

    Args:
        num_res: maximum sequence length
        hidden_dim: per-residue embedding dimension
        num_layers: number of IPA + transition blocks

    Returns:
        Keras model with inputs:
            - aa_input: (batch, num_res) amino acid indices
            - pairwise_input: (batch, num_res, num_res) sequence distance offsets
            - rotations: (batch, num_res, 3, 3) precomputed rotation matrices
            - translations: (batch, num_res, 3) precomputed translations
            - mask_input: (batch, num_res) sequence mask
        Returns: (batch, num_res, hidden_dim) per-residue embeddings
    """
    from invariant_attention import InvariantPointAttention

    max_dist = 32

    # Inputs
    aa_input = tf.keras.Input(shape=(num_res,), dtype=tf.int32, name="aa_input")
    pairwise_input = tf.keras.Input(
        shape=(num_res, num_res), dtype=tf.int32, name="pairwise_input"
    )
    rotations = tf.keras.Input(shape=(num_res, 3, 3), dtype=tf.float32, name="rotations")
    translations = tf.keras.Input(shape=(num_res, 3), dtype=tf.float32, name="translations")
    mask_input = tf.keras.Input(shape=(num_res,), dtype=tf.float32, name="mask_input")

    # Amino acid embedding
    aa_embed = tf.keras.layers.Embedding(21, hidden_dim, name="aa_embed")(aa_input)

    # Sequence distance embedding for pairwise repr
    dist_embed = tf.keras.layers.Embedding(
        2 * max_dist + 1, pairwise_repr_dim, name="dist_embed"
    )(pairwise_input)

    h = aa_embed
    for layer_idx in range(num_layers):
        # IPA layer
        ipa = InvariantPointAttention(
            dim=hidden_dim,
            heads=num_heads,
            scalar_key_dim=scalar_key_dim,
            scalar_value_dim=scalar_value_dim,
            point_key_dim=point_key_dim,
            point_value_dim=point_value_dim,
            pairwise_repr_dim=pairwise_repr_dim,
            require_pairwise_repr=True,
            name=f"ipa_layer_{layer_idx}",
        )
        ipa_out = ipa(
            h,
            dist_embed,
            rotations=rotations,
            translations=translations,
            mask=mask_input,
        )
        h = tf.keras.layers.LayerNormalization(name=f"ipa_ln_{layer_idx}")(h + ipa_out)

        # Transition layer: FFN
        ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    hidden_dim * 2, activation="gelu", name=f"ffn_dense1_{layer_idx}"
                ),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim, name=f"ffn_dense2_{layer_idx}"),
                tf.keras.layers.Dropout(dropout),
            ],
            name=f"ffn_{layer_idx}",
        )
        h = tf.keras.layers.LayerNormalization(name=f"trans_ln_{layer_idx}")(
            h + ffn(h)
        )

    # Output projection
    output = tf.keras.layers.Dense(hidden_dim, name="output_proj")(h)

    model = tf.keras.Model(
        inputs={
            "aa_input": aa_input,
            "pairwise_input": pairwise_input,
            "rotations": rotations,
            "translations": translations,
            "mask_input": mask_input,
        },
        outputs=output,
        name="ipa_encoder",
    )
    return model


# ── PDB parsing ───────────────────────────────────────────────────────────────


AA_TO_IDX = {
    "A": 1, "R": 2, "N": 3, "D": 4, "C": 5,
    "Q": 6, "E": 7, "G": 8, "H": 9, "I": 10,
    "L": 11, "K": 12, "M": 13, "F": 14, "P": 15,
    "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20,
    "X": 0,
}


def parse_pdb_coords(pdb_lines):
    """Parse CA, N, C coordinates from PDB lines."""
    n_atoms, ca_atoms, c_atoms = [], [], []
    prev_res = None

    for line in pdb_lines:
        if line[:4] == "ATOM":
            atomname = line[12:16].strip()
            resnum = line[22:26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            if resnum != prev_res:
                prev_res = resnum
                n_atoms.append([0.0, 0.0, 0.0])
                ca_atoms.append([0.0, 0.0, 0.0])
                c_atoms.append([0.0, 0.0, 0.0])

            if atomname == "N":
                n_atoms[-1] = [x, y, z]
            elif atomname == "CA":
                ca_atoms[-1] = [x, y, z]
            elif atomname == "C":
                c_atoms[-1] = [x, y, z]

    min_len = min(len(n_atoms), len(ca_atoms), len(c_atoms))
    return (
        np.array(n_atoms[:min_len]),
        np.array(ca_atoms[:min_len]),
        np.array(c_atoms[:min_len]),
    )


def parse_sequence_from_pdb(pdb_lines):
    """Extract amino acid sequence from PDB residues."""
    seq = []
    prev_res = None
    for line in pdb_lines:
        if line[:4] == "ATOM":
            resname = line[17:20].strip()
            resnum = line[22:26].strip()
            if resnum != prev_res:
                aa_idx = AA_TO_IDX.get(resname, 0)
                seq.append(aa_idx)
                prev_res = resnum
    return np.array(seq, dtype=np.int32)


# ── Main worker ───────────────────────────────────────────────────────────────


def main():
    input_data = json.loads(sys.argv[1])
    input_path = input_data["input_path"]
    output_path = input_data["output_path"]
    model_weights = input_data.get("model_weights", None)
    hidden_dim = input_data.get("hidden_dim", 128)
    num_layers = input_data.get("num_layers", 4)

    # Parse PDB
    with open(input_path) as f:
        pdb_lines = f.readlines()

    n_pos, ca_pos, c_pos = parse_pdb_coords(pdb_lines)
    sequence = parse_sequence_from_pdb(pdb_lines)
    L = len(sequence)

    # Compute local frames (numpy, outside graph)
    rotations, translations = build_local_frames(
        n_pos[None, :, :], ca_pos[None, :, :], c_pos[None, :, :]
    )

    # Build model
    model = build_ipa_encoder(
        num_res=L,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    # Load pretrained weights if provided
    if model_weights and model_weights != "none":
        try:
            model.load_weights(model_weights, skip_mismatch=True)
        except Exception:
            pass  # Random init if weights don't match

    # Prepare pairwise (sequence distance offsets)
    idx_offsets = np.arange(L)[None, :]  # (1, L)
    pairwise_offsets = idx_offsets - idx_offsets.T  # (1, L, L)
    max_dist = 32
    pairwise_offsets = np.clip(pairwise_offsets + max_dist, 0, 2 * max_dist).astype(
        np.int32
    )

    mask = np.ones((1, L), dtype=np.float32)

    # Predict
    embeddings = model.predict(
        {
            "aa_input": sequence[None, :].astype(np.int32),
            "pairwise_input": pairwise_offsets,
            "rotations": rotations,
            "translations": translations,
            "mask_input": mask,
        },
        verbose=0,
    )  # (1, L, hidden_dim)

    # Save
    np.save(output_path, embeddings[0], allow_pickle=False)
    print(f"IPA embeddings saved: {output_path}, shape {embeddings[0].shape}")


if __name__ == "__main__":
    main()
