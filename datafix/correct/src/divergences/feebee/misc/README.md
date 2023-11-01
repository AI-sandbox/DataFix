# FeeBee
FeeBee is a ***F***ram***e***work for ***e***valuating ***B***ayes ***e***rror ***e***stimators on real-world data with unknown underlying distribution.

## How-To: Run the framework

Running the evaluation on all datasets, feature transformations, methods and their hyper-parameters involves four major steps.

### (1) Export all dataset representations
The app used for exporting dataset representations into numpy arrays is can be found in the file `export.py`.
In order to export all representations inspect and run the following scripts:

- `bash scripts/export/mnist.sh`
- `bash scripts/export/cifar10.sh`
- `bash scripts/export/cifar100.sh`
- `bash scripts/export/imdb.sh`
- `bash scripts/export/sst2.sh`
- `bash scripts/export/yelp.sh`

### (2) Estimate the lower and upper bounds

Running all defined methods for a fixed feature transformation, dataset, and set of hyper-parameters can be done using the app given in the file `estimate.py`.
Every method should have a corresponding script in the folder `scripts/estimate/`. The current scripts are written such that they can be executed using [slurm](https://slurm.schedmd.com/documentation.html). Finally, the script `scripts/estimate/run_all.sh` runs all the methods on all datasets.

### (3) Collect the results

Every method executed for a fixed feature transformation, dataset, and set of hyper-parameters creates a single csv file with all estimates accross all independent runs. In order to collect these results into a exported pandas dataframe (`results.csv`), the corresponding script `collect_results.py` needs to be executed.

Running the script `run_analysis.py` allows to collect the failure state of single executions (timeout or memory error) into an exported dataframe `analysis.csv`.

### (4) Calculate the areas for FeeBee

Finally, using the collected results `results.csv`, one can calcuate the areas (i.e., FeeBee scores) for each successfull dataset, method, variant and tranformation combination. The script `calculate_areas.py` will perform this task and export the areas from a pandas dataframe into the file `areas.csv`.

In order to calculate the areas with varying SOTA factors, set the corresponding variable in the same script and re-run it.

## How-To: Perform the analysis

Examples on how to use the resulting Pandas dataframe along with all the code used to create Figures and Tables in the original publication of FeeBee, can be found in the provided jupyter notebook `analysis_notebook.ipynb`.
The three used pandas dataframes are shared over a public GDrive and downloaded automatically inside the above notebook if the flag `download_pre_computed_files` is set to True.

## How-To: Contribute

In order to implement and test your BER estimator using FeeBee, please submit a pull-request with your own BER estimator as a new file in the folder `methods`.
Your method needs to implement one of the following signatures:

- `def eval_from_single_matrix(features, labels)`
- `def eval_from_two_matrices(train_features, train_labels, test_features, test_labels)`

The features are represented by a 2d numpy array (first: number of samples, second: feature dimension), wereas the labels are 1D numpy arrays (number of samples).

Independent of the choice of signature, the method should return a dictionary. Every item in it should have a key representing a single variant (i.e., set of hyper-parameters, or use 'default' if not present), whereas the value of every item should be a list of two element. The first beeing the upper bound estimate and the second the lower bound estimate for the set of hyper-parameters. If a method estimates the BER directly, and not any lower or upper bound, the value in the dictionary should contain a list with twice the same element in it.

## Citation

Renggli, C., Rimanic, L., Hollenstein, N., & Zhang, C. (2021). Evaluating Bayes Error Estimators on Read-World Datasets with FeeBee. arXiv preprint arXiv:2108.13034. [[link]](https://arxiv.org/abs/2108.13034)
