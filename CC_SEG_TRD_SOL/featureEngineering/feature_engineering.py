import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataCleaning:
        
    def __init__(self):
        pass
          
    def create_dataframe(self, data_file):
        '''
        returns dataframe for a .csv file.
        '''
        self.data_file = data_file
        return pd.read_csv(data_file)
    
    def get_num_vars(self, data_frame):
        '''
        extracts columns for numerical variables from dataframe.
        '''
        self.data_frame = data_frame
        return data_frame._get_numeric_data()
        
    def get_cat_vars(self, data_frame):
        '''
        extracts columns for categorical variables from dataframe.
        '''
        self.data_frame = data_frame
        return data_frame.select_dtypes(include=['object'])
    
    def dataSummary_num(self, data_frame):
        '''
        create data audit report for numerical variables in a data frame
        '''
        self.data_frame = data_frame
        return pd.Series([data_frame.count(), data_frame.isnull().sum(), data_frame.sum(), data_frame.mean(), data_frame.median(),  
                          data_frame.std(), data_frame.var(), data_frame.min(), data_frame.dropna().quantile(0.01), 
                          data_frame.dropna().quantile(0.05),data_frame.dropna().quantile(0.10),
                          data_frame.dropna().quantile(0.25),data_frame.dropna().quantile(0.50),
                          data_frame.dropna().quantile(0.75), data_frame.dropna().quantile(0.90),
                          data_frame.dropna().quantile(0.95), data_frame.dropna().quantile(0.99),
                          data_frame.max()], index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD',
                                        'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])
        
    def outlier_capping(self, data_frame):
        '''
        outlier treatment
        '''
        self.data_frame = data_frame
        data_frame = data_frame.clip(upper = data_frame.quantile(0.99))
        data_frame = data_frame.clip(lower = data_frame.quantile(0.01))
        return data_frame
    
    def outlier_capping2(self, data_frame):
        '''
        skewed distribution - outlier treatment when (P99 and Max) or (P1 and Min) are significantly different
        '''
        self.data_frame = data_frame
        data_frame = data_frame.clip(upper = (data_frame.quantile(0.99) + data_frame.std() * 0.5))
        data_frame = data_frame.clip(lower = (data_frame.quantile(0.10) - data_frame.std() * 0.5))
        return data_frame
    
    def fillna_mean(self, data_frame):
        '''
        fills missing values with mean
        '''
        self.data_frame = data_frame
        data_frame = data_frame.fillna(data_frame.mean())
        return data_frame
    
    def fillna_median(self, data_frame):
        '''
        fills missing values with median
        '''
        self.data_frame = data_frame
        data_frame = data_frame.fillna(data_frame.median())
        return data_frame
    
    def fillna_mode(self, data_frame):
        '''
        fills missing values with mode
        '''
        self.data_frame = data_frame
        data_frame = data_frame.fillna(data_frame.mode())
        return data_frame
    
    def create_dummies(self, data_frame, colname ):
        '''
        creates dummy variables for categorical variables
        '''
        self.data_frame = data_frame
        self.colname = colname
        self.col_dummies = pd.get_dummies(data_frame[colname], prefix=colname)
        self.col_dummies.drop(self.col_dummies.columns[0], axis=1, inplace=True)
        data_frame = pd.concat([data_frame, self.col_dummies], axis=1)
        data_frame.drop( colname, axis = 1, inplace = True )
        return data_frame

class VariableReduction:
    
    def __init__(self):
        pass
    
    def vif(self):
        pass
    
    def data_standardization(self, data_frame):
        self.data_frame = data_frame
        sc = StandardScaler()
        return sc.fit_transform(data_frame)
    
    def get_PCA(self, data_frame, components):
        self.data_frame = data_frame
        self.components = components
        pc = PCA(n_components=components)
        pc.fit(data_frame)
        return pc
        
    def get_cumsum_exp_var_ratio(self, explained_variance_ratio_):
        self.explained_variance_ratio_ = explained_variance_ratio_
        return np.cumsum(np.round(explained_variance_ratio_, decimals=4)*100)
    
    def get_PCA_loadings(self, data_frame, components, explained_variance):
        self.data_frame = data_frame
        self.components = components
        self.explained_variance = explained_variance
        return pd.DataFrame((components.T * np.sqrt(explained_variance)).T, columns=data_frame.columns).T
        
        
        
        
        
        
        