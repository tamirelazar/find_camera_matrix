# find_camera_matrix
A tool that finds a calibration matrix for a given camera from an input image.

### Usage
just run :
```bash
python3 zhang_it.py IMG_DIR 
```

### Methods
Uses the zhang algorithm, following the following steps:
1. Calculate homography between squares in picture and real-world coords.
2. Apply homography to points in infinity (6 PII in total).
3. Fit a conic to those 6 points.
4. Apply cholesky decomposition to conic representative matrix.
5. Return result.
