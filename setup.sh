mkdir datasets
python upload_folder.py s3://bucket-6425/tangpin/data/ datasets/data/

cd datasets/data/nuscenes/

# 解压 trainval 数据（共 10 个包）
for i in $(seq -w 1 10); do
    tar -xf v1.0-trainval${i}_blobs.tar
done

# 解压 trainval 元数据
tar -xzf v1.0-trainval_meta.tgz

# 解压 test 数据及元数据
tar -xf v1.0-test_blobs.tar
tar -xf v1.0-test_meta.tar

# 解压语义分割与全景分割标注
tar -xf nuScenes-lidarseg-all-v1.0.tar
tar -xf nuScenes-panoptic-v1.0-all.tar

cd ../nuscenes_occ/trainval/
tar -xzf gts.tar.gz
cd ../../../..
rm -rf data/nuscenes

mv datasets/data/nuscenes/ data/nuscenes/
mv datasets/data/nuscenes_occ/trainval/gts/ data/nuscenes/gts/

conda activate veon
python setup.py build_ext --inplace --force
python -m pip install --no-index --no-build-isolation --no-deps -v -e .