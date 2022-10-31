## End_to_end_decoupled_training_for_skin_lesion_classification
_________________

This is the official implementation code of the paper [End-to-End Decoupled Training: a Robust Deep Learning Method for Long-tailed Classification of Dermoscopic Images for
Skin Lesion Classification](https://doi.org/10.3390/electronics11203275) in Pytorch.

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

- The original ISIC2018 dataset can be found at the following link [ISIC2018](https://challenge2018.isic-archive.com/task3/training/). 
- The breakdown into training, validation and test data sets of the original database as a '.csv' file is located in the  `/base/` directory



###Training

- To start training the model

```bash
python train_isic.py --loss_type 'DHML1' --weighting_type CS   
```

### Reference

If you find our paper and repo useful, please cite as

```
@article{foahom2022end,
  title={End-to-End Decoupled Training: A Robust Deep Learning Method for Long-Tailed Classification of Dermoscopic Images for Skin Lesion Classification},
  author={Foahom Gouabou, Arthur Cartel and Iguernaissi, Rabah and Damoiseaux, Jean-Luc and Moudafi, Abdellatif and Merad, Djamal},
  journal={Electronics},
  volume={11},
  number={20},
  pages={3275},
  year={2022},
  publisher={MDPI}
}

