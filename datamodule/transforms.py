import numpy as np
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from omegaconf import OmegaConf

class DataTransforms(object):
    def __init__(self, numerical_features, categorical_features, cog_features):
        self.numerical_features = OmegaConf.to_container(numerical_features, resolve=True)
        self.categorical_features = OmegaConf.to_container(categorical_features, resolve=True)
        self.cog_features = OmegaConf.to_container(cog_features, resolve=True)

        # Create pipelines for both numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
    
    def get_new_cols_name(self):
        onehot_columns = self.preprocessor.named_transformers_[
            'cat'].named_steps['onehot'].get_feature_names_out(
                input_features=self.categorical_features)
        numerical_columns = self.numerical_features 
        all_feature_names = list(numerical_columns) + list(onehot_columns)
        return all_feature_names

    def __call__(self, is_train, clinical_df):
        if is_train:
            clinical_df_transformed = self.preprocessor.fit_transform(clinical_df)
        else:
            clinical_df_transformed = self.preprocessor.transform(clinical_df)
        return clinical_df_transformed
    
    def denormalize_cog(self, cognitive_scores):
        if isinstance(cognitive_scores, np.ndarray):
            cog_scores = cognitive_scores
        else:
            cog_scores = cognitive_scores.clone() # [B, 13]
            cog_scores = cog_scores.detach()
            if cog_scores.is_cuda:
                cog_scores = cog_scores.cpu()
            cog_scores = cog_scores.numpy()
        cog_scores = pd.DataFrame(cog_scores, columns=self.cog_features)
        cog_list = []
        idx = 0
        for col in self.numerical_features:
            if col not in self.cog_features:
                cog_scores[col] = 0
            else:
                cog_list.append(idx)
            idx += 1
        cog_scores = cog_scores[self.numerical_features]

        inverse_transformer = self.preprocessor.named_transformers_['num']
        num_cols_original = inverse_transformer.named_steps['scaler'].inverse_transform(cog_scores)
        num_cols_original = num_cols_original[:,cog_list]

        return num_cols_original
