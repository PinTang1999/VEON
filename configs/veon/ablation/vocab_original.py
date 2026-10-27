"""Original vocabulary control with an explicit binary-only label boundary."""
_base_ = ['../veon-temporal-base-512x1408-dav2-nodepthcache.py']
model = dict(loss_occ=dict(semantic_supervision='occupancy_only', priority_mode='original'))
# train_vocab.py snapshots the inherited pipeline and collapses Occ3D IDs at load.
