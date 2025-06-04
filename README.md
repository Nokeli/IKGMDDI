# IKGMDDI  

**Integrated Knowledge Graph and Drug Molecular Graph Fusion via Adversarial Networks for Drug–Drug Interaction Prediction**  
[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.4c01647-blue)](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01647)

Official implementation of the IKGMDDI model for drug-drug interaction prediction.

## Installation Guide  

### Prerequisites  
- Python 3.8+  
- [RDKit](https://www.rdkit.org/docs/Install.html) (Recommended to install via `conda`)  
- Other dependencies: `numpy`, `pandas`, `tqdm`, `joblib`  

### Quick Installation  
```bash  
https://github.com/Nokeli/IKGMDDI.git 
cd IKGMDDI  
pip install -r requirements.txt

## run code
1. For datasets, You can get its from the link
https://drive.google.com/file/d/1oGkck0F7hDmQBsNOEbnM2uJRPwnLwiuI/view?usp=sharing
2. For Binary—classification task, you can run:
```
python train_on_fold_our.py
```

3.For multi-classification task in drugbank datasets, you can run:
```
python train_on_fold_our.py
```
