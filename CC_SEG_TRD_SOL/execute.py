from featureEngineering.feature_engineering import DataCleaning,VariableReduction
from modelBuilding.segmentation_algo import DistBasedAlgo
from evaluationMetrices.evaluation_metrices import EMSegmentation
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

dc = DataCleaning() 
vr = VariableReduction()
dba = DistBasedAlgo()
ems = EMSegmentation()
df = dc.create_dataframe('cc_seg_data.csv')
df_num = dc.get_num_vars(df)
df_cat = dc.get_cat_vars(df)

num_summary = df_num.apply(lambda x: dc.dataSummary_num(x)).T
#num_summary.to_csv('num_summary.csv')

df_new = df_num.apply(lambda x: dc.fillna_median(x))

df_new = df_new.apply(lambda x: dc.outlier_capping(x))

num_summary2 = df_new.apply(lambda x: dc.dataSummary_num(x)).T
#num_summary2.to_csv('num_summary2.csv')

df_scaled = vr.data_standardization(df_new)

scaled = pd.DataFrame(df_scaled).describe()

pca = vr.get_PCA(df_scaled, 17)
cumsum_var = vr.get_cumsum_exp_var_ratio(pca.explained_variance_ratio_)

#plt.plot(cumsum_var)
pc_final = vr.get_PCA(df_scaled, 8)

'''
calculate loadings
'''
loadings = vr.get_PCA_loadings(df_num, pc_final.components_, pc_final.explained_variance_)
loadings.to_csv('loadings.csv')

selected_columns = ['PURCHASES','PURCHASES_TRX',
                    'PURCHASES_FREQUENCY','INSTALLMENTS_PURCHASES','ONEOFF_PURCHASES',
                    'CASH_ADVANCE','BALANCE','CASH_ADVANCE_TRX','CASH_ADVANCE_FREQUENCY','TENURE']

select_columns2 = ['PURCHASES','PURCHASES_TRX','PURCHASES_FREQUENCY','CASH_ADVANCE','BALANCE',
                   'PURCHASES_INSTALLMENTS_FREQUENCY','TENURE','CREDIT_LIMIT','PRC_FULL_PAYMENT']


df_scaled_1 = pd.DataFrame(df_scaled, columns = df_num.columns)
df_scaled_final_1 = df_scaled_1[selected_columns]
df_scaled_final_2 = df_scaled_1[select_columns2]

km_3_1 = dba.k_means(df_scaled_final_1, 3)
km_4_1 = dba.k_means(df_scaled_final_1, 4)
km_5_1 = dba.k_means(df_scaled_final_1, 5)
km_6_1 = dba.k_means(df_scaled_final_1, 6)
km_7_1 = dba.k_means(df_scaled_final_1, 7)
km_8_1 = dba.k_means(df_scaled_final_1, 8)
km_9_1 = dba.k_means(df_scaled_final_1, 9)
km_10_1 = dba.k_means(df_scaled_final_1, 10)

df_num['cluster_3'] = km_3_1.labels_
df_num['cluster_4'] = km_4_1.labels_
df_num['cluster_5'] = km_5_1.labels_
df_num['cluster_6'] = km_6_1.labels_
df_num['cluster_7'] = km_7_1.labels_
df_num['cluster_8'] = km_8_1.labels_
df_num['cluster_9'] = km_9_1.labels_
df_num['cluster_10'] = km_10_1.labels_

#
#k_range = range(2, 11)
#score = []
#
#for k in k_range:
#    km = dba.k_means(df_scaled_final_2, k)
#    score.append(ems.check_silhouette_score(df_scaled_final_2, km.labels_))
#    
#plt.plot(k_range, score)
#plt.xlabel('no of clusters')
#plt.ylabel('silhouette co-eff')
#plt.grid(True)


'''
get total size using any cluster
get size for each cluster and each segment
'''
size = pd.concat([pd.Series(df_num.cluster_3.size), pd.Series.sort_index(df_num.cluster_3.value_counts()), 
                  pd.Series.sort_index(df_num.cluster_4.value_counts()),pd.Series.sort_index(df_num.cluster_5.value_counts()),
                  pd.Series.sort_index(df_num.cluster_6.value_counts()),pd.Series.sort_index(df_num.cluster_7.value_counts()), 
                  pd.Series.sort_index(df_num.cluster_8.value_counts()), pd.Series.sort_index(df_num.cluster_9.value_counts()),
                  pd.Series.sort_index(df_num.cluster_10.value_counts())])

Seg_size = pd.DataFrame(size, columns=['Seg_size'])
Seg_pct = pd.DataFrame(size/df_num.cluster_3.size, columns=['Seg_pct'])

'''
get total mean for each column
get mean for each cluster and each segment
'''
Profiling_output = pd.concat([df_num.apply(lambda x: x.mean()).T, df_num.groupby('cluster_3').apply(lambda x: x.mean()).T, 
                             df_num.groupby('cluster_4').apply(lambda x: x.mean()).T,df_num.groupby('cluster_5').apply(lambda x: x.mean()).T, 
                             df_num.groupby('cluster_6').apply(lambda x: x.mean()).T,df_num.groupby('cluster_7').apply(lambda x: x.mean()).T, 
                             df_num.groupby('cluster_8').apply(lambda x: x.mean()).T, df_num.groupby('cluster_9').apply(lambda x: x.mean()).T,
                             df_num.groupby('cluster_10').apply(lambda x: x.mean()).T], axis=1)

Profiling_output_final = pd.concat([Seg_size.T, Seg_pct.T, Profiling_output], axis=0)
Profiling_output_final.columns = ['Overall', 'KM3_1', 'KM3_2', 'KM3_3',
                                'KM4_1', 'KM4_2', 'KM4_3', 'KM4_4',
                                'KM5_1', 'KM5_2', 'KM5_3', 'KM5_4', 'KM5_5',
                                'KM6_1', 'KM6_2', 'KM6_3', 'KM6_4', 'KM6_5','KM6_6',
                                'KM7_1', 'KM7_2', 'KM7_3', 'KM7_4', 'KM7_5','KM7_6','KM7_7',
                                'KM8_1', 'KM8_2', 'KM8_3', 'KM8_4', 'KM8_5','KM8_6','KM8_7','KM8_8',
                                'KM9_1', 'KM9_2', 'KM9_3', 'KM9_4', 'KM9_5','KM9_6','KM9_7','KM9_8','KM9_9',
                                'KM10_1', 'KM10_2', 'KM10_3', 'KM10_4', 'KM10_5','KM10_6','KM10_7','KM10_8','KM10_9','KM10_10']

Profiling_output_final.to_csv('profiling2.csv')






