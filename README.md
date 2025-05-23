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
python main.py --input Video/test1.mp4 --output Results/output1.mp4 --alpha 30.0 --preprocess \
    --start-time 1 --end-time 18 --fps 15 --max-k 4 \
    --force-recompute

python main.py --input Video/test2.mp4 --output Results/output2.mp4 --alpha 30.0 --preprocess \
    --start-time 11 --end-time 20 --fps 15 --max-k 4 \
    --force-recompute

python main.py --input Video/test3.mp4 --output Results/output3.mp4 --alpha 30.0 --preprocess \
    --start-time 1 --end-time 18 --fps 15 --max-k 4 \
    --force-recompute

python main.py --input Video/test4.mp4 --output Results/output4.mp4 --alpha 30.0 --preprocess \
    --fps 15 --max-k 4 \
    --force-recompute

python main.py --input Video/test6.mp4 --output Results/output6.mp4 --alpha 30.0 --preprocess \
    --start-time 18 --end-time 40 --fps 15 --max-k 4 \
    --force-recompute

python main.py --input Video/test7.mp4 --output Results/output7.mp4 --alpha 30.0 --preprocess \
    --start-time 10 --end-time 30 --fps 15 --max-k 4 --resolution 1280x720 \
    --force-recompute
```