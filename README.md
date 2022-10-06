## End_to_end_decoupled_training_for_skin_lesion_classification
_________________

This is the official implementation code of the paper [End-to-End Decoupled Training: a Robust Deep Learning Method for Long-tailed Classification of Dermoscopic Images for
Skin Lesion Classification](https:) in Pytorch.

### Abstract figure

![Alt text](ressources/images/abstract_figure.PNG?)
### Dependency
The code is build with following main libraries
- [Pytorch](https://www.tensorflow.org) 1.11.0
- [Numpy](https://numpy.org/) 
- [Pandas](https://pandas.pydata.org/)
- [Sklearn](https://scikit-learn.org/stable/)
- [Matlab](https://ch.mathworks.com/fr/products/matlab.html)

You can install all dependencies with requirements.txt following the command:
```bash
pip install -r requirements.txt 
```


### Dataset

- ISIC2018 [ISIC2018](https://challenge2019.isic-archive.com/). The original data will be preprocessed by `/isic2019_implementation/preprocessing/preprocessImageConstancy.m`and split by `/isic2019_implementation/preprocessing/train_valid_split_task.py`.


###Training

- To train the HMLoss baseline on 2-class version of isic2019 for melanoma versus nevi classification

```bash
python train_isic.py --loss_type 'HML1' --delta 10000   
```

### Reference

If you find our paper and repo useful, please cite as

```
@Article{foahom2022end,
  title={End-to-End Decoupled Training: a Robust Deep Learning Method for Long-tailed Classification of Dermoscopic Images for Skin Lesion Classification},
  author={Foahom Gouabou, Arthur Cartel and Iguernaissi, Rabah and Damoiseaux, Jean Luc and Moudafi, Abdellatif and Merad, Djamal},
  JOURNAL = {Electronics},
  DOI = {},
  pages={},
  year={2022}
}
