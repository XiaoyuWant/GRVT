GRVT:Toward effective grocery recognition via vision transformer

##  Download dataset
1. GSD. https://github.com/marcusklasson/GroceryStoreDataset
2. FGD. http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/
3. Modify file path in `utils.dataset`

## Requirements
1. Download weight `ViT-B_16.npz` to folder `pretrain/`

## Experiment
GSD
```
sh train_grocery.sh
```

FGD
```
sh train_freiburg.sh
```