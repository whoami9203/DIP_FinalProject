# DIP_FinalProject

Create virtual environment via `conda`.

```bash
# for Mac(Intel) / Linux
conda create -n dip_gpu -c rapidsai -c nvidia -c conda-forge cuml=23.12 python=3.9 cudatoolkit=11.8
conda activate dip_project
pip install -r requirements.txt
```

The project consists of these parts:
- Background Modeling
    - K-Means Background Clustering
	- Gaussian Modeling
- Object Mask Generation
    - Foreground Segmentation
	- Noise Reduction
	- Cast Shadow Cancellation
	- Watershed Transform
	- Region Coverage Testing

Each part is implemented in a separate file/function. The main function is in `main.py`.

```bash
python main.py --input Video/test1.mp4 --output Results/output1.mp4 --alpha 9.0 --preprocess \
    --start-time 1 --end-time 18 --fps 15 --max-k 4 \
    --force-recompute

python main.py --input Video/test6.mp4 --output Results/output6.mp4 --alpha 9.0 --preprocess \
    --start-time 18 --end-time 40 --fps 15 --max-k 4 \
    --force-recompute
```