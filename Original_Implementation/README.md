# PatternChrome Readme
## Description
PatternChrome aims to predict the expression of genes from profiles of 5 histone modifications (HM) around their transcirption start site (TSS). The algorithm utilizes particle swarm optimization for extracting predictive HM patterns. PatternChrome was trained and tested on 56 samples from the [REMC](https://www.nature.com/articles/nbt1010-1045?error=cookies_not_supported&code=32ca7b94-c8f0-4bda-bd7b-e85c45e7d1bc) database separately.

In order to compare PatternChromes performance it aims to solve the same task as multiple other algorithms. These are listed below:


| Algorithm | Published | Architecture |
| -------- | -------- | -------- |
| [DeepChrome](https://doi.org/10.1093/bioinformatics/btw427) | 2016-08-29 | CNN |
| [DeepChrome 2.0](https://doi.org/10.48550/arXiv.2209.11923) | 2022-09-24 | CNN |
| [ShallowChrome](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04687-x) | 2023-04-26 | Linear Regression |

## Installation
PatternChromes repository is not written in the format of an installable package but rather constitutes the collection of scripts, their preprocessd data and their outputs. Therefore the repository can be downladed and, given that the [prerequisites](#requirements) are met, the scripts can be run individually. Before running any scripts from this repository, the **Path_config.yaml file needs to be adjusted** to your architecture since all scripts use this file to find the required data and settings. 


### Requirements
PatternChrome was run on a CPU server with GNU/Linux 11 as operating system and kernel version 5.10.0-22-amd64 but should run on any system if an **R** installation (version 4.3.1 or newer) and the required dependencies are available. PatternChrome utilizes particle swarm optimization for feature extraction and hyperparameter tuning. This is a very **time intensive** process (training the models takes roughly 30-45 minutes each of the 56 samples on the aforementioned machine). Although one can adjust the number of used cpus (n_workers setting in the Path_config.yaml file) the PatternChrome.R script will **use all availabe cpus** on the machine due to an internal parallelization of a library function.

### Dependencies
PatternChrome requires the listed R libraries:

| library | Version | Scripts |
| -------- | -------- | -------- |
| parallel | 4.3.1 | |
| xgboost |1.7.5.1 | |
| caTools | 1.18.2 | |
| pROC | 1.18.5 | |
| pso | 1.0.4 | |
| dplyr | 1.1.4 | |
| yaml | 2.3.7 | |
| MLmetrics | 1.1.1| |
| ggplot2 |3.4.4 | |
| ggpubr |0.6.0 | |
| data.table |1.14.8 | |
| cluster |2.1.4 | |
| factoextra |1.0.7 | |
| forcats |1.0.0 | |

## Usage
### Quick Start
If you want to reenact how PatternChrome results were produced follow these commands after checking the [requisites](#requirements) and installing required [libraries](#dependencies).
1. adjust Path_config.yaml
2. clear stats.csv
3. clear figures
4. run ```Rscript --vanilla PatternChrome.R```
5. run ```Rscript --vanilla Calculate_pattern_importance.R```
6. run ```Rscript --vanilla Cell_type_annotation.R```

Visualization:

7. run ```Rscript --vanilla Plot_auc_scores.R```
8. run ```Rscript --vanilla Analysis_generalizability.R```
9. run ```Rscript --vanilla General_pattern_trends.R ```
10. run ```Rscript --vanilla XGB_explainer.R```
11. run ```Rscript --vanilla Cell_line_performance.R```

```
./Analysis/Interpretation_cell_line.R

./Visualization/Pattern_location_visualization.R

./Visualization/Trained_pattern_number.R

./Visualization/Visualize_alternative_cutoff.R


Rscript --vanilla PatternChrome_regression.R 

```




### Structure of the project

#### The Path_config.yaml file
The scripts in PatternChrome use the Path_config.yaml file to locate required files and parameters. This allows the user to easily adjust the rough folder structure of PatternChrome and customize specific parameters. The Paths_config.yaml files requires the adjustment of the following directories:

**PatternChrome_dir**: Directory of the local clone of the online repository.
**dataset_path**: Directory that stores the results from PatternChrome, for each of the samples (*E003* etc.).
**main_path**: Directory of the binned input histone modification data for PatternChrome. Contains a file for each sample and for each sample 5 RData files that contain actual data. 
**RNAseq_path**: Directory of the RNA-seq data for all samples and genes savd as .RData file. This also is a required input for PatternChrome.
**analysis_data**: Directory where results from the analysis of patterns are stored.
**Custom_functions**: Path to the .R (*Custom_functions.R*) file that contains function definitions that are needed by multiple scripts.
**Figures**: Directory where the scripts will save simple .png pictures (picture's name includes the name of the respective script) if the parameter ```Plot``` is set to ```TRUE```.


The Path_config.yaml allows the adjustment of the following parameters:
**Plot**: Should all scripts save a simple png file of the produced figures in the ```Figures``` directory? The picture's names contain the names of the rwspctive script. This parameter accepts only logical values (```TRUE```/```FALSE```).
**font_size**: Font size for the created figures
**n_workers**: Number of cpus that should be used for parallelization. **Caution!** Due to an internal parallelization of some library functions most scripts that utilize parallelization will use all available cores temporarily anyway!

## Support
Niels Benjamin Paul:
niels.paul(at)bioinf.med.uni-goettingen.de

Jonas Wolber
jwolber(at)ukaachende

Martin Haubrock:
martin.haubrock(at)bioinf.med.uni-goettingen.de


## Acknowledgment
Please note the included LICENSE file.
