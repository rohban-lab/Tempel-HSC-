### Clone:
``` bash
git clone https://github.com/garjania/Tempel-HSC-.git
cd Tempel-HSC-
```
### Datsets:
Experiments both on [Tempel](https://www.researchgate.net/publication/338951815_Tempel_Time-series_Mutation_Prediction_of_Influenza_A_Viruses_via_Attention-based_Recurrent_Neural_Networks) and our model is done on the H1N1, H3N2 and H5N1 influenza subtypes. The datasets are provided in this [link](https://drive.google.com/drive/folders/1-pJGBsVfIqCEizetTQe43OQJvkmhocdW?usp=sharing). After downloading this folder (named raw) run the following commands to create a directory for datasets:
``` bash
mkdir -p data/processed
```
Place the downloaded folder in the data directory just created.
### Train and Test the Model:
Run the command below to train and test the model. `--dataset` argument can be one of H1N1, H3N1 or H5N1. `--start_year` indicates the first year which datasets are processed from and `--end_year` shows the target year. To do preprocessing set `--create_dataset` True and for training set `--train` True.
``` bash
python Tempel\ HSC.py --dataset=H1N1 --start_year=2001 --end_year=2016 --create_dataset=True --train=True
```

Similarly, for running [Tempel](https://www.researchgate.net/publication/338951815_Tempel_Time-series_Mutation_Prediction_of_Influenza_A_Viruses_via_Attention-based_Recurrent_Neural_Networks) 's code run the following command:
``` bash
python Tempel.py --dataset=H1N1 --start_year=2001 --end_year=2016 --create_dataset=True --train=True
```