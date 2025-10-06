"""
Quick test script to verify data preparation pipeline.
"""
from homework_scientalab.data import DataConfig, prepare_dataloaders

if __name__ == "__main__":
    # Configure data preparation
    cfg = DataConfig(
        data_path="data/pancreas.h5ad",
        backup_url="https://www.dropbox.com/s/qj1jlm9w10wmt0u/pancreas.h5ad?dl=1",
        min_genes=200,
        min_cells=10,
        n_top_genes=2000,  # Use 2000 HVGs
        batch_key="tech",
        celltype_key="celltype",
        val_size=0.2,
        random_state=42,
    )
    
    # Prepare dataloaders
    train_loader, val_loader, dims = prepare_dataloaders(
        cfg,
        batch_size=256,
        num_workers=0,  # Use 0 for debugging, increase for training
        pin_memory=True,
    )
    
    # Test one batch
    print("\n" + "=" * 80)
    print("TESTING ONE BATCH")
    print("=" * 80)
    
    batch = next(iter(train_loader))
    print(f"Batch contents:")
    for key, val in batch.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        if key == "x":
            print(f"    Raw counts - min={val.min():.0f}, max={val.max():.0f}, mean={val.mean():.1f}")
        elif key == "size_factor":
            print(f"    Size factors - min={val.min():.3f}, max={val.max():.3f}, mean={val.mean():.3f}")
    
    print("\n✓ Data preparation successful!")
    print(f"\nReady for model initialization with:")
    print(f"  x_dim={dims['x_dim']}, b_dim={dims['b_dim']}, t_dim={dims['t_dim']}")

