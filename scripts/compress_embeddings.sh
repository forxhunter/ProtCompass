#!/bin/bash
# Compress embeddings for HuggingFace upload
# Hybrid strategy: per-encoder for large tasks, per-task for medium/small

set -e

EMBEDDINGS_DIR="downstream_tasks/embeddings"
cd "$EMBEDDINGS_DIR"

echo "Starting embeddings compression..."
echo "Original size: $(du -sh . | cut -f1)"

# Large tasks (>30GB) - compress per encoder for flexibility
echo ""
echo "=== Compressing large tasks per-encoder ==="
for task in contact_prediction secondary_structure ppi_site metal_binding; do
    if [ -d "$task" ]; then
        echo "Processing $task..."
        cd "$task"
        for encoder in */; do
            if [ -d "$encoder" ]; then
                encoder_name="${encoder%/}"
                echo "  Compressing $encoder_name..."
                tar -czf "${encoder_name}.tar.gz" "$encoder" && rm -rf "$encoder"
            fi
        done
        cd ..
    fi
done

# Medium tasks (5-30GB) - compress entire task
echo ""
echo "=== Compressing medium tasks ==="
for task in mutation_effect go_bp stability solubility go_mf fluorescence; do
    if [ -d "$task" ]; then
        echo "Compressing $task..."
        tar -czf "${task}.tar.gz" "$task" && rm -rf "$task"
    fi
done

# Small tasks (<5GB) - compress entire task
echo ""
echo "=== Compressing small tasks ==="
for task in ec_classification subcellular_localization membrane_soluble remote_homology ppi_affinity; do
    if [ -d "$task" ]; then
        echo "Compressing $task..."
        tar -czf "${task}.tar.gz" "$task" && rm -rf "$task"
    fi
done

echo ""
echo "Compression complete!"
echo "Compressed size: $(du -sh . | cut -f1)"
echo ""
echo "Next steps:"
echo "1. Copy compressed embeddings to ProtCompass_Embeddings:"
echo "   cp -r downstream_tasks/embeddings/* ProtCompass_Embeddings/embeddings/"
echo "2. Upload to HuggingFace:"
echo "   cd ProtCompass_Embeddings && git add embeddings/ && git commit -m 'Add compressed embeddings' && git push"
