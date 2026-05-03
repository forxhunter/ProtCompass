#!/bin/bash
# Organize data for HuggingFace repositories

set -e

echo "=========================================="
echo "Data Organization for HuggingFace Upload"
echo "=========================================="
echo ""

# Step 1: Add structure_encoder_data to ProtEnv
echo "Step 1: Adding structure_encoder_data to ProtEnv..."
if [ ! -d "ProtEnv/structure_encoder_data" ]; then
    echo "  Copying structure_encoder_data (117GB) to ProtEnv..."
    cp -r structure_encoder_data/ ProtEnv/
    echo "  ✓ Done"
else
    echo "  ✓ structure_encoder_data already in ProtEnv"
fi

# Step 2: Compress embeddings
echo ""
echo "Step 2: Compressing embeddings (601GB -> ~150GB)..."
echo "  This will take several hours..."
read -p "  Proceed with compression? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash scripts/compress_embeddings.sh
    echo "  ✓ Compression complete"
else
    echo "  ⊘ Skipped compression"
fi

# Step 3: Update ProtCompass_Embeddings with compressed embeddings
echo ""
echo "Step 3: Updating ProtCompass_Embeddings..."
if [ -f "downstream_tasks/embeddings/secondary_structure.tar.gz" ]; then
    echo "  Removing old partial embeddings..."
    rm -rf ProtCompass_Embeddings/embeddings/*

    echo "  Copying compressed embeddings..."
    # Copy compressed task archives
    find downstream_tasks/embeddings/ -name "*.tar.gz" -exec cp {} ProtCompass_Embeddings/embeddings/ \;

    # Copy per-encoder compressed files from large tasks
    for task in contact_prediction secondary_structure ppi_site metal_binding; do
        if [ -d "downstream_tasks/embeddings/$task" ]; then
            mkdir -p "ProtCompass_Embeddings/embeddings/$task"
            cp downstream_tasks/embeddings/$task/*.tar.gz "ProtCompass_Embeddings/embeddings/$task/" 2>/dev/null || true
        fi
    done

    echo "  ✓ Done"
else
    echo "  ⊘ Compressed embeddings not found. Run compression first."
fi

# Step 4: Summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "ProtEnv (Datasets):"
du -sh ProtEnv/ 2>/dev/null || echo "  Error checking size"
echo ""
echo "ProtCompass_Embeddings (Embeddings + Results):"
du -sh ProtCompass_Embeddings/ 2>/dev/null || echo "  Error checking size"
echo ""
echo "Next steps:"
echo "1. cd ProtEnv && git add structure_encoder_data/ && git commit -m 'Add structure_encoder_data' && git push"
echo "2. cd ProtCompass_Embeddings && git add embeddings/ && git commit -m 'Add compressed embeddings' && git push"
echo "3. Update README files with download/decompression instructions"
