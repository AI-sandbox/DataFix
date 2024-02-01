################################################################################
# DataFix Correction.
################################################################################

import copy
import math
import numpy as np

from catboost import CatBoostClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

from ..divergences.discriminator_divergence_estimator import div_tv_clf
from .supervised_imputer import SupervisedImputer


class DFCorrect:
    """
    DataFix corrupted feature correction.
    
    Parameters
    ----------
    base_classifier : object, default=CatBoostClassifier(verbose=False, random_state=0)
        The base classifier used as discriminator.
    batch_size : int, default=5000
        The size of the batches used for processing the query dataset.
    max_dims : int, default=None
        If there are more than `max_dims` features, up to `max_dims` features are 
        randomly selected from the correct features to speed up processing. None means 
        that all correct features are used for the correction of corrupted features.
    num_epochs : int, default=1
        Number of epochs for iterative adversarial imputation.
    verbose : bool, default=False
        Verbosity.

    Returns
    -------
    query : array-like
        Query with corrected features.
    """
    def __init__(
        self,
        base_classifier=CatBoostClassifier(verbose=False, random_state=0),
        batch_size=5000,
        max_dims=None,
        num_epochs=1,
        random_seed=0,
        verbose=False
    ):
        self.base_classifier = base_classifier
        self.batch_size = batch_size
        self.max_dims = max_dims
        self.num_epochs = num_epochs
        self.random_seed = random_seed
        self.verbose = verbose
        
        np.random.seed(random_seed)
        

    def copy(self):
        """
        Returns a deep copy of the current object.
        """
        return copy.deepcopy(self)

    def fit_transform(self, reference, query, mask):
        """
        Parameters
        ----------
        reference : array-like
            The reference dataset is assumed to contain high-quality data.
        query : array-like
            The query dataset might contain some partially or completely corrupted features.
            It must contain the same features as the reference appearing in the same order.
        mask : array of shape (n_features_in_,)
            The mask of corrupted features, where 1 indicates a variable is
            corrupted and 0 otherwise.
        """
        self.num_corrupted_feats = sum(mask)
        self.query_original_sorted = query.copy()
        
        query = np.concatenate([query[:,mask==1], query[:,mask==0]], axis=1)
        reference = np.concatenate([reference[:,mask==1], reference[:,mask==0]], axis=1)
        
        self.mask = mask
        self.query_original = query.copy()

        if reference.shape[0] > 15000:
            self.batch_size = 3000

        # Select up to `max_dims` features at random from the correct features to speed up processing
        if self.max_dims is not None and query[:, self.num_corrupted_feats:].shape[1] > self.max_dims:
            std = np.std(query[:, self.num_corrupted_feats:], axis=0)
            indexes = np.concatenate(
                [
                    np.arange(self.num_corrupted_feats),
                    np.argsort(std)[-self.max_dims:] + self.num_corrupted_feats,
                ]
            )
            reference = reference[:, indexes]
            query = query[:, indexes]
            
            if self.verbose:
                print(f"{len(indexes)} features in reference and query.")

        self.reference = reference
        self.query = query
        self.num_samples = reference.shape[0]
        self.num_batches = math.floor(self.num_samples / self.batch_size)

        if self.reference.shape[0] <= self.batch_size:
            if self.verbose:
                print("No batch processing")
            # Define ``AdversarialIterativePerSampleImputer``
            imputer = AdversarialIterativePerSampleImputer(
                self.base_classifier,
                self.reference.copy(),
                self.query.copy(),
                self.num_corrupted_feats,
                self.num_epochs,
                self.verbose
            )
            # Correct corrupted features in query
            query_imputed = imputer.fit_transform()

            if self.query.shape[1] < self.query_original.shape[1]:
                # Update original query with corrected corrupted features
                # This condition is triggered when ``max_dims`` is not None
                query = self.query_original
                query[:, :self.num_corrupted_feats] = query_imputed[
                    :, :self.num_corrupted_feats
                ]
                query_sorted = self.query_original_sorted.copy()
                query_sorted[:, mask==1] = query[:, :self.num_corrupted_feats]
                return query_sorted
            else:
                query_sorted = self.query_original_sorted.copy()
                query_sorted[:, mask==1] = query_imputed[:, :self.num_corrupted_feats]
                return query_sorted
        else:
            # Permute reference samples
            perm_indices = np.random.permutation(self.reference.shape[0])
            for k in range(self.num_batches):
                if self.verbose:
                    print("Processing batch", k, self.num_batches)
                # Define start and end indexes for batch size
                s, e = self.batch_size * k, self.batch_size * (k + 1)
                if k == (self.num_batches - 1):
                    e = self.reference.shape[0]
                
                # Define ``AdversarialIterativePerSampleImputer``
                imputer = AdversarialIterativePerSampleImputer(
                    self.base_classifier,
                    self.reference[perm_indices, :][s:e, :].copy(),
                    self.query[perm_indices, :][s:e, :].copy(),
                    self.num_corrupted_feats,
                    self.num_epochs,
                    self.verbose
                )
                # Correct corrupted features in query for current batch size
                query_imputed_small = imputer.fit_transform()
                idx_update = perm_indices[s:e]
                self.query[idx_update, :] = query_imputed_small

            if self.query.shape[1] < self.query_original.shape[1]:
                # Update original query with corrected corrupted features
                # This condition is triggered when ``max_dims`` is not None
                query = self.query_original
                query[:, : self.num_corrupted_feats] = self.query[
                    :, : self.num_corrupted_feats
                ]
                query_sorted = self.query_original_sorted.copy()
                query_sorted[:, mask==1] = query[:, :self.num_corrupted_feats]
                return query_sorted
            else:
                query_sorted = self.query_original_sorted.copy()
                query_sorted[:, mask==1] = self.query[:, :self.num_corrupted_feats]
                return query_sorted


class AdversarialIterativePerSampleImputer:
    """
    Performs adversarial iterative per-sample imputation.

    Parameters
    ----------
    base_classifier : object
        The base classifier used as discriminator.
    reference : array-like
            The reference dataset is assumed to contain high-quality data.
    query : array-like
        The query dataset might contain some partially or completely corrupted features.
        It must contain the same features as the reference appearing in the same order.
        The first `num_corrupted_feats` must correspond to the corrupted features detected by `DFLocate`.
    num_corrupted_feats : int
        Total number of corrupted features.
    num_epochs : int, default=1
        Number of epochs for iterative adversarial imputation.
    verbose : bool, default=False
        Verbosity.
    """
    def __init__(self, base_classifier, reference, query, num_corrupted_feats, num_epochs=1, verbose=False):
        self.reference = reference
        self.query = query
        self.query_original = query.copy()
        self.num_query_samples = query.shape[0]
        self.num_corrupted_feats = num_corrupted_feats
        self.num_epochs = num_epochs
        self.verbose = verbose

        # Imputed query at each process iteration
        self.query_epoch_log = []

        self.discriminators_list = []
        self.discriminator_base_classifier = base_classifier
        self.eval_classifier = base_classifier

        # Training data to include in discriminator fitting
        self.include_samplewise_shuffle_aug = True
        self.include_featurewise_shuffle_aug = True

        # Supervised Imputers
        self.imputer_list = []
        self.imputed_query_list = []

    def fit_transform(self):
        """
        Performs imputation in the query dataset.
        """
        self.warm_up()
        self.impute_dataset()
        return self.query

    def warm_up(self):
        """
        Initial function that:
        1. Fits initial external imputers including linear regression and k-nn.
        2. Performs imputation with the imputers fitted in step (1).
        3. Gets proposals for starting points e.g. random imputation and imputation from imputers in step (1,2).
        4. Evaluates each initial proposal (step 3) and selects the one with lowest divergence as starting point.
        """
        if self.verbose:
            print("Getting initial proposals")
        self.fit_internal_imputers()
        self.impute_with_sklearn_imputers()
        self.get_initial_proposals()
        self.evaluate_initial_proposals()
        return self

    def _add_proposal_inside_query(self, proposal):
        """
        Helper function for 'get_initial_proposals()'. Replaces the corrupted features 
        in the query with the corresponding features from the proposal.
        """
        _proposal = self.query.copy()
        _proposal[:, 0:self.num_corrupted_feats] = proposal[
            :, 0:self.num_corrupted_feats
        ]
        # Return the query with the corrupted features substituted by the proposal
        return _proposal

    def fit_internal_imputers(self):
        """
        (Warmup) Training of internal imputation done during warmp up using 
        linear regression and k-nn. The imputers are fitted on the reference.                
        """
        # Linear regression imputer
        imputer = SupervisedImputer(LinearRegression(), self.num_corrupted_feats)
        imputer.fit(self.reference)
        self.imputer_list.append(imputer)

        # K-NN imputers
        for k in [1]:
            for weights in ["distance", "uniform"]:
                imputer = KNNImputer(n_neighbors=k, weights=weights)
                imputer.fit(self.reference)
                self.imputer_list.append(imputer)
        return self.imputer_list

    def impute_with_sklearn_imputers(self):
        """
        (Warmup) Inference of internal imputation done during warmp up using 
        linear regression and k-nn.
        """
        que_missing = self.query.copy()
        # Set all corrupted features to missing
        que_missing[:, 0:self.num_corrupted_feats] = np.nan

        for imputer in self.imputer_list:
            # For each imputer trained on the reference...
            # Predict the missing values in the query
            que_imputed = imputer.transform(que_missing.copy())
            self.imputed_query_list.append(que_imputed)
        return self.imputed_query_list
    
    def extend_or_sample_reference(self, reference):
        """
        Extends or samples the reference data to match the desired number of query samples.
        """    
        if self.num_query_samples > len(reference):
            # Calculate the number of additional samples needed
            additional_samples = self.num_query_samples - len(reference)

            # Randomly sample rows from the reference to extend it
            random_indices = np.random.choice(len(reference), additional_samples)
            additional_reference_samples = reference[random_indices, :]

            # Extend the reference by appending the additional samples
            reference = np.vstack([reference, additional_reference_samples])
        elif self.num_query_samples < len(reference):
            # If there are more reference samples than needed, truncate the reference
            reference = reference[:self.num_query_samples, :]

        return reference
    
    def get_initial_proposals(self):
        """
        (warmup) Obtain initial proposals for the warmup imputation process.
        The reference dataset and imputed queries obtained with different 
        imputers are used to generate the proposals.
        """
        proposals = []
        
        # Add reference as initial proposal
        proposals.append(self._add_proposal_inside_query(self.extend_or_sample_reference(self.reference.copy())))

        for proposal in self.imputed_query_list:
            # Add each proposal inside query obtained with a different imputer
            proposals.append(self._add_proposal_inside_query(proposal))

        # Store only the corrupted features from the proposals for evaluation purposes
        proposals_array = np.concatenate(proposals, axis=0)
        proposals_array = proposals_array[:, 0:self.num_corrupted_feats]
        self.initial_proposals = proposals

    def get_iteration_proposals(self):
        """
        (impute_sample) Obtain iteration proposals for the iterative imputation process.
        The proposals are generated using the reference dataset, permutations of the 
        reference, and imputed samples using linear regression.
        """
        proposals = []
        # Add reference as iteration proposal
        proposals.append(self._add_proposal_inside_query(self.extend_or_sample_reference(self.reference.copy())))

        # Generate random permutations of the reference for each corrupted feature
        ref_perm = self.reference.copy()
        for jj in range(self.num_corrupted_feats):
            randperm = np.random.permutation(self.reference.shape[0])
            ref_perm[:, jj] = ref_perm[randperm, jj]
        
        # Add random permutations of the reference as iteration proposals
        proposals.append(self._add_proposal_inside_query(self.extend_or_sample_reference(ref_perm)))
        proposals.append(
            self._add_proposal_inside_query(self.imputed_query_list[0])
        ) # We only add from linear regression imputation

        # Store only the corrupted features from the proposals for evaluation purposes
        proposals_array = np.concatenate(proposals, axis=0)
        proposals_array = proposals_array[:, 0:self.num_corrupted_feats]

        return proposals, proposals_array

    def evaluate_proposal(self, proposal):
        """
        (evaluate_initial_proposals) Evaluate a proposal in terms of Total Variation (TV) 
        distance between the reference and the proposal, as well as the cross-validation (CV) 
        probabilities obtained with the base classifier.
        """
        tv, cv_proba = div_tv_clf(self.reference, proposal, clf=self.eval_classifier)
        return tv

    def evaluate_initial_proposals(self):
        """
        (warmup) Evaluates all initial proposals and selects the proposal with the minimum 
        Total Variation (TV) distance.
        """
        best_tv = 1.0
        for j, proposal in enumerate(self.initial_proposals):
            if self.verbose:
                print(f"Evaluating proposal {j}")
            tv = self.evaluate_proposal(proposal)
            if self.verbose:
                print("TV:", tv, "| Best TV:", best_tv)

            if tv < best_tv:
                # Update the best TV with the new lower TV
                best_tv = tv
                # Update the current query with the new best proposal
                self.query = proposal.copy()

            if best_tv < 0.1:
                # Stop the evaluation process as a satisfactory proposal has been found
                break

    def fit_initial_discriminators(self):
        """
        (impute_dataset) Fit two discriminators: one on the reference and one half of 
        the query, and another on the reference and the other half of the query.
        """
        for i in [1, 0]:
            # Create a matrix with all reference samples and half of the query samples
            x = np.concatenate([self.reference, self.query[i::2, :]], axis=0)
            # Create a list with labels for reference samples as 0 and query samples as 1
            y = np.concatenate(
                [
                    np.zeros(self.reference.shape[0]),
                    np.ones(self.query[i::2, :].shape[0]),
                ],
                axis=0,
            )

            if self.include_samplewise_shuffle_aug:
                # Add augmented samples with sample-wise shuffling
                ref_perm = self.reference.copy()
                randperm = np.random.permutation(self.reference.shape[0])
                ref_perm[:, 0:self.num_corrupted_feats] = ref_perm[
                    randperm, 0:self.num_corrupted_feats
                ]
                x = np.concatenate([x, ref_perm], axis=0)
                y = np.concatenate([y, np.ones(ref_perm.shape[0])], axis=0)

            if self.include_featurewise_shuffle_aug:
                # Add augmented samples with feature-wise shuffling
                ref_perm = self.reference.copy()
                for jj in range(self.num_corrupted_feats):
                    randperm = np.random.permutation(self.reference.shape[0])
                    ref_perm[:, jj] = ref_perm[randperm, jj]
                x = np.concatenate([x, ref_perm], axis=0)
                y = np.concatenate([y, np.ones(ref_perm.shape[0])], axis=0)
        
            for (prev_que) in (self.query_epoch_log):
                # Add previous query epochs to training data
                x = np.concatenate([x, prev_que], axis=0)
                y = np.concatenate([y, np.ones(prev_que.shape[0])], axis=0)

            # Fit discriminator on all training data
            clf = self.discriminator_base_classifier
            clf.fit(x, y)
            self.discriminators_list.append(clf)

    def detect_bad_samples(self):
        """
        (impute_dataset) Perform cross-validation (CV) to evaluate a discriminator
        trained on the reference samples to classify the query samples in terms of
        their probability of being from the query dataset. Samples with higher
        probabilities are considered more corrupt. Return the CV probabilities for
        the query samples.
        """
        cv_pred = cross_val_predict(
            self.eval_classifier,
            np.concatenate([self.reference, self.query], axis=0),
            np.concatenate(
                [np.zeros(self.reference.shape[0]), np.ones(self.query.shape[0])],
                axis=0,
            ),
            cv=4,
            method="predict_proba",
        )
        cv_pred = cv_pred[self.reference.shape[0]:, 1]  # Keep query results only
        return cv_pred

    def impute_sample(self, index):
        """
        (impute_dataset) Obtain different proposals for a query sample and 
        select the proposal with the lowest probability of belonging to the query 
        based on the discriminator that was not fitted with that particular sample.
        """
        x = self.query[index, :]
        _, initial_proposals = self.get_iteration_proposals()
        classifier = self.discriminators_list[index % 2]
        x_new = self.optimize_sample(x, initial_proposals, classifier)
        return x_new

    def impute_dataset(self):
        """
        Main loop performing imputation at each sample.
        """
        for epoch in range(self.num_epochs):
            if self.verbose:
                print("Epoch:", epoch)
            # Fit initial discriminators where each discriminator does not see half 
            # of the query samples
            self.fit_initial_discriminators()
            
            if self.verbose:
                print("Detecting bad samples")
            # Obtain the probability of query samples belonging to the query dataset
            preds = self.detect_bad_samples()
            # Sort the probabilities in descending order
            argsort = np.argsort(preds)[::-1]
            bad_idx = []
            # Select the indices where the probabilities exceed the threshold (0.5)
            for i in argsort:
                if preds[i] > 0.5:
                    bad_idx.append(i)
            bad_idx = np.array(bad_idx)
            
            # Calculate the number of elements per batch based on the excess bad indices
            num_elements_per_batch = max(
                0, int(bad_idx.shape[0] - self.query.shape[0] * 0.5)
            )

            if bad_idx.shape[0] / self.query.shape[0] < 0.53:
                # Check if the ratio of bad indices to total query samples is below the threshold
                # indicating that the model's predictions are close to a random guess
                if self.verbose:
                    print("Finishing process!")
                break

            for j, idx in enumerate(list(bad_idx)):

                # Impute samples that are more corrupted
                self.query[idx, :] = self.impute_sample(idx)

                if j == num_elements_per_batch:
                    break

            print("Re-fitting internal discriminators")
            self.query_epoch_log.append(self.query.copy())

        return self

    def optimize_sample(self, x, initial_proposals, classifier):
        """
        (impute_sample) Optimize a query sample by selecting the proposal with the 
        minimum probability of belonging to the query from a set of initial proposals.
        """
        x = np.repeat(x[np.newaxis, :], initial_proposals.shape[0], axis=0)
        x[:, 0:self.num_corrupted_feats] = initial_proposals[
            :, 0:self.num_corrupted_feats
        ]
        loss = classifier.predict_proba(x)[:, 1]
        x_new = x[np.argmin(loss), :]
        return x_new
