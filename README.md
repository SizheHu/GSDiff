# GSDiff
Official implementation of the AAAI 2025 paper: "GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation"

## Data
1. Create folder `datasets/rplandata/Data`.
2. Download the 80,788 RPLAN dataset (http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html). It contains a `floorplan_dataset` folder. Place this `floorplan_dataset` folder under `datasets/rplandata/Data`.
3. Run the following scripts to obtain structural graph data:
   ```bash
   python rplan-extract.py
   python rplan-process1.py
   python rplan-process2.py
   python rplan-process3.py
   python rplan-process4.py
   ```
   - After completion, a directory `rplang-v3-withsemantics` (65,763 train + 3,000 val + 3,000 test = 71,763 `.npy` files) will be created under `datasets/rplandata/Data`.
   - Running `rplan-extract.py` also generates folders (`1-channel-semantics-256`, `3-channel-semantics-256`, `bin_imgs`, `e_imgs`, etc.). You can remove them.

4. Run the following scripts to obtain structural graph data with boundaries:
   ```bash
   python rplan-process5.py
   python rplan-process6.py
   python rplan-process7.py
   ```
   - After completion, 2 directories `rplang-v3-withsemantics-withboundary` and `rplang-v3-withsemantics-withboundary-v2`, will appear under `datasets/rplandata/Data`, each containing 71,763 files.

5. Run the following scripts to obtain topology graphs:
   ```bash
   python rplan-process8.py
   python rplan-process9.py
   python rplan-process10.py
   ```
   - After completion, a `rplang-v3-bubble-diagram` folder will be created under `datasets/rplandata/Data`, containing the same number of files.

Note, when we conducted our experiments, the semantics of bubble diagram GT involved randomness. Due to the terms of the RPLAN dataset, we are not permitted to release any part of it. Therefore, the bubble diagram GT semantics extracted using the provided scripts may differ slightly from our experimental results. However, given the large data scale, the bias should be minor (for rooms with ambiguous categories, both our GT and the GT you extract randomly select one category, which is no statistically significant difference).
Alternatively, you can use `get_cycle_basis_and_semantic_3_semansimplified` instead of `get_cycle_basis_and_semantic_2_semansimplified` in `rplan-process8/9/10.py` to extract room semantics and train your own topology models. This method is not random, may yield improvements over the metrics reported in the paper.


# Usage
The test scripts for no constraints, topology constraints, and boundary constraints are all placed under `scripts` (test_xxx.py). 
Download the corresponding weights and run them via:
   ```bash
   python test_xxx.py
   ```

No constraints: We use the original 3000 results and run them 5 times to get the average.

Topology constraints: We took the intersection of the original 3000 results with the test set numbers of HouseDiffusion and House-GAN++, and got 757. 
We ran them 5 times and averaged them to get the FID, KID, GED, and statistical analysis of each room type. 
The sample numbers of 757 are in line 183 of `evalmetric-topoconstrain-ged-roomnumber.py`.

Boundary constraints: We took the intersection of the original 3000 results with the test set numbers of HouseDiffusion and House-GAN++, and got 378. 
We got the FID, KID, GED, and statistical analysis of each room type. 
The sample number of 378 is on line 9 of `evalmetric-boun-constrain-fid-kid.py`.


# params (place in the 'outputs' folder)
unconstrained params: https://drive.google.com/file/d/15gM0GtW2GwHmlpz0r-rpvo-k-BlNy_gu/view?usp=sharing

topology-constrained params: https://drive.google.com/file/d/1pk7SmvLZ8ON3OUL3SNxPRu73ndVKru0z/view?usp=sharing

boundary-constrained params: https://drive.google.com/file/d/1XMU5zxXoxiXg3LB9rlluWOX9BT2YIGdK/view?usp=sharing

boundary-autoencoder CNN params: https://drive.google.com/file/d/1l6QRpfX5Jtucg3R995HajlwRG8SewUJW/view?usp=sharing

topology-autoencoder Transformer params: https://drive.google.com/file/d/1tExX8LdrFpJfBQH5y2emC6BltBwf9tHx/view?usp=sharing

