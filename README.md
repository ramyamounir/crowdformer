# crowdformer

This is an unofficial implementation of the CrowdFormer architecture: [https://arxiv.org/abs/2203.03768](https://arxiv.org/abs/2203.03768).

---

### USAGE

We train and evaluate on two datasets `ShaghaiTech` (SH) and the yet to be realeased `CraneCounting` (CC) dataset from the recently accepted IJCAI'23 paper: [**Long-term Monitoring of Bird Flocks in the Wild**](https://www.ijcai.org/proceedings/2023/0704.pdf).

To preprocess the SH dataset, use the `prep_SH` folder, which also contains the dataloader in `prep_SH/dataset.py`.

To run on the SH dataset, use the `main.py` file.

> Similar commands are used for CC dataset. (e.g., `main_CC.py`, `Prep_CC/`)
