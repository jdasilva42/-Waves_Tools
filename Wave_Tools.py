"""
Module containing a Wave tools for analyze timeseries.

    plot_frequency():
        
        plot the frequencies of occurrence for each feature and the cumulative frequencies
            Hs -> Wave amplitude meter
            Tp -> Wave period seconde
            Direction -> Direction of the Wave train.
        *The variables Direction can be missing.*     
        
        the function incorporate an option "seasons_split" spliting the features
        by seasons (winter, summer, spring, autumn) if activate.
        
    
    plot_correlogram():
        
        plot the correlation between the features contained in the timeserie.
            Hs / Tp
            Hs / Direction
            Tp / Direction
        as the function plot_frequency(), the "seasons_split" option can be activated.
        

    Kmeans_setup() & GM_setup() 
    
    The module Wave tools contains unsupervised clustering "Kmean" and "GaussianMixture".
    If the "clustering" option is activated the Kmeans_setup() (clustering = Kmean) or
    GM_setup() (clustering=GM) run allowing to plot the informations required for choose the best number of clusters.
    These functions have the option "seasons_split" also.
    
    These functions incorporate a Standardization of the input data.
    
    
    Kmean_run() & GM_run()
    
    These functions return the centroids of each cluster. For Gaussian Mixture (GM_run()) the function incorporate 
    the Anomaly Detection for the timeserie. The keyword p_err controls the anomalie selection. A p_err equal to 1 means that
    approximately 1 % of the instances will be flagged as anomalies.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from math import ceil
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture

class Wave_Tools():

    
    def __init__(self, data, seasons_split=False, clustering = None, n_k=None ,p_err=None):
        
        #import the data
        self.data = data
              
        #option seasons_split
        if seasons_split:
            
            self.seasons_split = True
            
            idx_Seasons = {
                'winter' : [12,1,2],
                'spring' : [3,4,5],
                'summer' : [6,7,8],
                'autumn' : [9,10,11]
                }
            self.seasons = idx_Seasons
            
            self.data_by_season = dict()
            
            for season, months in idx_Seasons.items():
                self.data_by_season[season]=self.data[self.data.index.month.isin(months)]  

        else:
            self.seasons_split = False
            
        #check if the Direction column is present or not
        if 'Direction' in self.data:
            self.Direction = True
        else:
            self.Direction = False
        
       
        #Calculate the frequencies of occurrence and the cumulative frequencies
        self._freq_t()
        
       
        #check if the clustering is required and which clustering
        #The data is automatically Standardized for the clustering algorithms
        if clustering == 'Kmean' or clustering == 'GM':
            
            self.scaler=StandardScaler()

            if self.seasons_split:
                
                self.data_by_season_scaled = dict()
                
                for season in self.seasons:
                    self.data_by_season_scaled[season] =  pd.DataFrame(self.scaler.fit_transform(self.data_by_season[season]))
                    
            else:
                self.data_scaled = pd.DataFrame(self.scaler.fit_transform(self.data))
                
            if n_k :
                self.n_k=n_k
            else:
                self.n_k=range(2, 10)
          
        #run the clustering algorithm. if the option clustering is activated the p_err is set to 1 for the function Gm_run()
        # the keyword p_err can be modified by the user.        
        if clustering == 'Kmean':
            
            self.Kmeans_setup()
        
        if clustering =='GM':
            
            self.GM_setup()
            
            if p_err:
                self.p_err = p_err
            else:
                self.p_err = 1
        


    def _freq_t(self):
        
        
        if self.seasons_split:
            
            
            self.freq_by_season= dict()
            
            if self.Direction:
                column_name = ['Hs','Tp','Direction']
            else:
                column_name = ['Hs','Tp']
                
            for season in self.seasons:
                self.freq_by_season[season]={}
                for column in column_name:
                    if column == 'Hs':
                        bins=np.arange(0,float(ceil(self.data_by_season[season][column].max()))+0.5,0.5)
                    if column =='Tp':
                        bins=np.arange(0,float(ceil(self.data_by_season[season][column].max()))+1,1)
                    if self.Direction:
                        if column == 'Direction':
                            bins=np.arange(0,360,15)
                    
                    self.freq_by_season[season][column] = pd.DataFrame(self.data_by_season[season][column].value_counts(bins=bins))
                    self.freq_by_season[season][column]['freq'] = round(self.freq_by_season[season][column]*100/ self.freq_by_season[season][column].sum())
                    self.freq_by_season[season][column]['cum_freq']= np.cumsum(self.freq_by_season[season][column]['freq'])
            
            return self.freq_by_season
        
        else:
            self.freq_all= dict()
            
            if self.Direction:
                column_name = ['Hs','Tp','Direction']
            else:
                column_name = ['Hs','Tp']
                
            for column in column_name:
                if column == 'Hs':
                    bins=np.arange(0,float(ceil(self.data[column].max()))+0.5,0.5)
                if column =='Tp':
                    bins=np.arange(0,float(ceil(self.data[column].max()))+1,1)
                if self.Direction:
                    if column == 'Direction':
                        bins=np.arange(0,360,15)
                    
                self.freq_all[column] = pd.DataFrame(self.data[column].value_counts(bins=bins))
                self.freq_all[column]['freq'] = round(self.freq_all[column]*100/ self.freq_all[column].sum())
                self.freq_all[column]['cum_freq']= np.cumsum(self.freq_all[column]['freq'])            
            
            return self.freq_all 
            
    def plot_frequency(self):
        
   
        if self.seasons_split:

            for season in self.seasons:
                
                fig,axes = plt.subplots(1,3,figsize=(16,6))
                
                #Hs plot
                ax1 = axes[0].twinx()
                axes[0].set_title('Hs [m] frequency during the %s'%season + ' period')
                sns.barplot(np.arange(0.5,float(ceil(self.data_by_season[season]['Hs'].max()))+0.5,0.5),
                            self.freq_by_season[season]['Hs']['freq'],
                            color='#8d8d8d',
                            edgecolor= '0.3',
                            ax=axes[0])
                    
                sns.pointplot(np.arange(0.5,float(ceil(self.data_by_season[season]['Hs'].max()))+0.5,0.5),
                            self.freq_by_season[season]['Hs']['cum_freq'],
                            ax=ax1,
                            color='k',
                            scale = 0.5,
                            linestyles=':')
                    
                ax1.set_ylim([0, 105]), ax1.set_ylabel('')
                ax1.set_yticklabels(['%1.f%%' %i for i in ax1.get_yticks()])
               
                axes[0].set_yticklabels(['%1.f%%' %i for i in axes[0].get_yticks()])
                axes[0].set_ylabel('Frequency')
                plt.setp(axes[0].get_xticklabels(), rotation='vertical')
            
                #Tp plt
                ax2 = axes[1].twinx() 
                axes[1].set_title('Tp [s] frequency during the %s'%season + ' period')       
                    
                sns.barplot(np.arange(1,float(ceil(self.data_by_season[season]['Tp'].max()))+1,1),
                            self.freq_by_season[season]['Tp']['freq'],
                            color='#8d8d8d',
                            edgecolor= '0.3',
                            ax=axes[1])
                    
                sns.pointplot(np.arange(1,float(ceil(self.data_by_season[season]['Tp'].max()))+1,1),
                              self.freq_by_season[season]['Tp']['cum_freq'],
                              ax=ax2,
                              color='k',
                              scale = 0.5,
                              linestyles=':')
                    
                ax2.set_ylim([0, 105]), ax2.set_ylabel('')      
                ax2.set_yticklabels(['%1.f%%' %i for i in ax2.get_yticks()])
                ax2.set_ylabel('')
                    
                axes[1].set_yticklabels(['%1.f%%' %i for i in axes[1].get_yticks()])
                axes[1].set_ylabel('')
                axes[1].set_xticklabels(['%1.f' %i for i in np.arange(1,float(ceil(self.data_by_season[season]['Tp'].max()))+1,1)])
                plt.setp(axes[1].get_xticklabels(),rotation='vertical')
                    
                if self.Direction:

                    #Direction 
                    ax3 = axes[2].twinx()
                    axes[2].set_title('Direction [°] frequency during the %s'%season+' period')
                        
                    sns.barplot(np.arange(15,360,15),
                                self.freq_by_season[season]['Direction']['freq'],
                                color='#8d8d8d',
                                edgecolor= '0.3',
                                ax=axes[2])
                        
                    sns.pointplot(np.arange(15,360,15),
                                  self.freq_by_season[season]['Direction']['cum_freq'],
                                  ax=ax3,
                                  color='k',
                                  scale = 0.5,
                                  linestyles=':') 
                
                    ax3.set_ylim([0, 105]), ax3.set_ylabel('cumulative frequency')
                    ax3.set_yticklabels(['%1.f%%' %i for i in ax3.get_yticks()])
                        
                    axes[2].set_yticklabels(['%1.f%%' %i for i in axes[2].get_yticks()])
                    axes[2].set_ylabel('')
                    axes[2].set_xticklabels(['%1.f' %i for i in np.arange(15,360,15)])
                    plt.setp(axes[2].get_xticklabels(),rotation='vertical')
          
                fig.tight_layout()
                
        
        else:

            fig,axes = plt.subplots(1,3,figsize=(16,6))
                        
            #Hs plot
            ax1 = axes[0].twinx()
            axes[0].set_title('Hs [m] frequency all the period')
            sns.barplot(np.arange(0.5,float(ceil(self.data['Hs'].max()))+0.5,0.5),
                        self.freq_all['Hs']['freq'],
                        color='#8d8d8d',
                        edgecolor= '0.3',
                        ax=axes[0])
                        
            sns.pointplot(np.arange(0.5,float(ceil(self.data['Hs'].max()))+0.5,0.5),
                          self.freq_all['Hs']['cum_freq'],
                          ax=ax1,
                          color='k',
                          scale = 0.5,
                          linestyles=':')
                        
            ax1.set_ylim([0, 105]), ax1.set_ylabel('')
            ax1.set_yticklabels(['%1.f%%' %i for i in ax1.get_yticks()])

            axes[0].set_yticklabels(['%1.f%%' %i for i in axes[0].get_yticks()])
            axes[0].set_ylabel('Frequency')
            plt.setp(axes[0].get_xticklabels(),rotation='vertical')
                
            #Tp plt
            ax2 = axes[1].twinx() 
            axes[1].set_title('Tp [s] frequency during all the period')       
                        
            sns.barplot(np.arange(1,float(ceil(self.data['Tp'].max()))+1,1),
                        self.freq_all['Tp']['freq'],
                        color='#8d8d8d',
                        edgecolor= '0.3',
                        ax=axes[1])
                        
            sns.pointplot(np.arange(1,float(ceil(self.data['Tp'].max()))+1,1),
                          self.freq_all['Tp']['cum_freq'],
                          ax=ax2,
                          color='k',
                          scale = 0.5,
                          linestyles=':')
                        
            ax2.set_ylim([0, 105]), ax2.set_ylabel('')
            ax2.set_yticklabels(['%1.f%%' %i for i in ax2.get_yticks()])
                        
            axes[1].set_yticklabels(['%1.f%%' %i for i in axes[1].get_yticks()])
            axes[1].set_ylabel('')
            axes[1].set_xticklabels(['%1.f' %i for i in np.arange(1,float(ceil(self.data['Tp'].max()))+1,1)])
            plt.setp(axes[1].get_xticklabels(),rotation='vertical')
                        
            if self.Direction:

                #Direction 
                ax3 = axes[2].twinx()
                axes[2].set_title('Direction [°] frequency during all the period')
                            
                sns.barplot(np.arange(15,360,15),
                            self.freq_all['Direction']['freq'],
                            color='#8d8d8d',
                            edgecolor= '0.3',
                            ax=axes[2])
                            
                sns.pointplot(np.arange(15,360,15),
                              self.freq_all['Direction']['cum_freq'],
                              ax=ax3,
                              color='k',
                              scale = 0.5,
                              linestyles=':') 
                    
                ax3.set_ylim([0, 105]), ax3.set_ylabel('cumulative frequency')
                ax3.set_yticklabels(['%1.f%%' %i for i in ax3.get_yticks()])
                            
                axes[2].set_yticklabels(['%1.f%%' %i for i in axes[2].get_yticks()])
                axes[2].set_ylabel('')
                axes[2].set_xticklabels(['%1.f' %i for i in np.arange(15,360,15)])
                plt.setp(axes[2].get_xticklabels(),rotation='vertical')
        
            fig.tight_layout()
            
            
    def plot_correlogram(self):
        
        if self.seasons_split:
            
            for season in self.seasons:
                              
                sns.set(style="white")
                g=sns.jointplot(x= self.data_by_season[season]['Tp'], y= self.data_by_season[season]['Hs'],kind='hex',color='red')
                g.fig.suptitle('Correlogram Hs / Tp during the %s'%season + ' period',y=0.999)
                g.set_axis_labels('Parameter Tp [s]', 'Parameter Hs [m]')
                
                if self.Direction:
    
                    g=sns.jointplot(x=self.data_by_season[season]['Direction'], y=self.data_by_season[season]['Hs'],kind='hex',color='red')
                    g.fig.suptitle('Correlogram Hs / Direction during the %s'%season + ' period',y=0.999)
                    g.set_axis_labels('Parameter Direction [°]', 'Parameter Hs [m]')
                     
                    g=sns.jointplot(x=self.data_by_season[season]['Tp'], y=self.data_by_season[season]['Direction'],kind='hex',color='red')
                    g.fig.suptitle('Correlogram Tp / Direction during the %s'%season + ' period',y=0.999)
                    g.set_axis_labels('Parameter Tp [s]', 'Parameter Direction [°]')
    
        else:
            
            sns.set(style="white")
            g=sns.jointplot(x=self.data['Tp'], y=self.data['Hs'],kind='hex',color='red')
            g.fig.suptitle('Correlogram Hs / Tp during all seasons' ,y=0.999)
            g.set_axis_labels('Parameter Tp [s]', 'Parameter Hs [m]')
                 
            if self.Direction:
                 
                g=sns.jointplot(x=self.data['Direction'], y=self.data['Hs'],kind='hex',color='red')
                g.fig.suptitle('Correlogram Hs / Direction during all seasons',y=0.999)
                g.set_axis_labels('Parameter Direction [°]', 'Parameter Hs [m]')
                     
                g=sns.jointplot(x=self.data['Tp'], y=self.data['Direction'],kind='hex',color='red')
                g.fig.suptitle('Correlogram Tp / Direction during all seasons',y=0.999)
                g.set_axis_labels('Parameter Tp [s]', 'Parameter Direction [°]')
    
    
    def Kmeans_setup(self):
        
        if self.seasons_split:
            
            kmeans_per_k = dict()
            inertias = dict()
            silouhettes = dict()

            for season in self.seasons:               

                kmeans_per_k[season]=[KMeans(n_clusters=k, random_state=42,n_init=10,n_jobs=-1,init='k-means++').fit(
                    self.data_by_season_scaled[season]) for k in self.n_k]
                inertias[season] = [model.inertia_ for model in kmeans_per_k[season]]
                silouhettes[season] = [silhouette_score(self.data_by_season_scaled[season],model.labels_) for model in kmeans_per_k[season]]
                
                fig = plt.figure(figsize=(16,6))
                plt.subplot(1,2,1)
                plt.plot(self.n_k,inertias[season],'o-',color='k')
                plt.title('Inertia vs number of clusters k for %s'%season)
                plt.xlabel('number of clusters k')
                plt.ylabel('inertia')
                plt.xticks(self.n_k)
                plt.grid()
                
                plt.subplot(1,2,2)
                plt.plot(self.n_k,silouhettes[season],'o-',color='k')
                plt.title('Silouette_score vs number of clusters k for %s'%season)
                plt.xlabel('number of clusters k')
                plt.ylabel('Silouette_score')
                plt.xticks(self.n_k)
                plt.grid()
                fig.tight_layout()
                
                fig = plt.figure(figsize=(8.27,11.69)) #A4
                fig.suptitle('silhouette diagram for %s' %season)
                for k in self.n_k:
                    plt.subplot(3,ceil(max()/3),k-min(self.n_k)+1)
                    y_pred = kmeans_per_k[season][k-min(self.n_k)].labels_
                    silhouette_coefficients = silhouette_samples(self.data_by_season_scaled[season], y_pred)
                    
                    padding = len(self.data_by_season_scaled[season]) // 30
                    pos = padding
                    ticks = []
                    for i in range(k):
                        coeffs = silhouette_coefficients[y_pred == i]
                        coeffs.sort()
                        color = plt.cm.Spectral(i / k)
                        
                        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                                          facecolor=color, edgecolor=color, alpha=0.7)
                        ticks.append(pos + len(coeffs) // 2)
                        pos += len(coeffs) + padding
        
                    plt.axvline(x=silouhettes[season][k-min(self.n_k)], color="red", linestyle="--")
                    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
                    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
                    plt.gca().set_xticklabels([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                    plt.xticks(fontsize=5)
                    plt.yticks(fontsize=5)
                    plt.gca().set_yticks
                    plt.title("$k={}$".format(k), fontsize=5)
                    plt.xlabel("Silhouette Coefficient",fontsize=5)
                    if k in np.arange(2,30,3):
                        plt.ylabel("Cluster",fontsize=5)

        else:
            
            kmeans_per_k=[KMeans(n_clusters=k, random_state=42,n_init=10,n_jobs=-1,init='k-means++').fit(self.data_scaled)
                          for k in self.n_k]
            inertias = [model.inertia_ for model in kmeans_per_k]
            silouhettes = [silhouette_score(self.data_scaled,model.labels_) for model in kmeans_per_k]
            
            fig = plt.figure(figsize=(16,6))
            plt.subplot(1,2,1)
            plt.plot(self.n_k,inertias,'o-',color='k')
            plt.title('Inertia vs number of clusters k')
            plt.xlabel('number of clusters k')
            plt.ylabel('inertia')
            plt.xticks(self.n_k)
            plt.grid()
            
            plt.subplot(1,2,2)
            plt.plot(self.n_k,silouhettes,'o-',color='k')
            plt.title('Silouette_score vs number of clusters k')
            plt.xlabel('number of clusters k')
            plt.ylabel('Silouette_score')
            plt.xticks(self.n_k)
            plt.grid()
            
            fig = plt.figure(figsize=(8.27,11.69)) #A4
            for k in self.n_k:
                
                plt.subplot(3,ceil(max(self.n_k)/3),k-min(self.n_k)+1)
                y_pred = kmeans_per_k[k-min(self.n_k)].labels_
                silhouette_coefficients = silhouette_samples(self.data_scaled, y_pred)
                
                padding = len(self.data_scaled) // 30
                pos = padding
                ticks = []
                for i in range(k):
                    coeffs = silhouette_coefficients[y_pred == i]
                    coeffs.sort()
                    color = plt.cm.Spectral(i / k)
                    
                    plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                                      facecolor=color, edgecolor=color, alpha=0.7)
                    ticks.append(pos + len(coeffs) // 2)
                    pos += len(coeffs) + padding
    
                plt.axvline(x=silouhettes[k-min(self.n_k)], color="red", linestyle="--")
                plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
                plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
                plt.gca().set_xticklabels([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                plt.xticks(fontsize=5)
                plt.yticks(fontsize=5)
                plt.gca().set_yticks
                plt.title("$k={}$".format(k), fontsize=5)
                plt.xlabel("Silhouette Coefficient",fontsize=5)
                if k in np.arange(2,30,3):
                    plt.ylabel("Cluster",fontsize=5)

    def Kmean_run(self, k):
        
        if self.seasons_split:
            
            kmeans_model = dict()
            clusters = dict()
                       
            for season in self.seasons:
                kmeans_model[season]=KMeans(n_clusters=k, random_state=42,n_init=10,n_jobs=-1,init='k-means++').fit(
                    self.data_by_season_scaled[season])
                
                if self.Direction:
                    clusters[season] = pd.DataFrame(self.scaler.inverse_transform(kmeans_model[season].cluster_centers_), columns=['Hs', 'Tp', 'Direction'])
                else:
                    clusters[season] = pd.DataFrame(self.scaler.inverse_transform(kmeans_model[season].cluster_centers_), columns=['Hs', 'Tp'])
            
            self.clusters = clusters
        
        else :
            kmeans_model = KMeans(n_clusters=k, random_state=42,n_init=20,n_jobs=-1,init='k-means++').fit(self.data_scaled)
        
            if self.Direction:
                cluster = pd.DataFrame(self.scaler.inverse_transform(kmeans_model.cluster_centers_), columns=['Hs', 'Tp', 'Direction'])
            else:
                cluster = pd.DataFrame(self.scaler.inverse_transform(kmeans_model.cluster_centers_), columns=['Hs', 'Tp'])
        
        self.clusters = cluster

    
    def GM_setup(self):
        
        if self.seasons_split:
                   
            gms_per_k = dict()
            bics = dict()
            aics = dict()
            
            for season in self.seasons:
                
                gms_per_k[season]= [GaussianMixture(n_components=k, n_init=10, random_state=42,covariance_type = 'full').fit(self.data_by_season_scaled[season])
                                     for k in self.n_k]
                bics[season] = [model.bic(self.data_by_season_scaled[season]) for model in gms_per_k[season]]
                aics[season] = [model.aic(self.data_by_season_scaled[season]) for model in gms_per_k[season]]
                
                fig = plt.figure()
                
                for k in self.n_k:
                               
                    plt.title('Information Criterion for %s'%season)
                    plt.plot(self.n_k, bics[season], "ko-", label="BIC")
                    plt.plot(self.n_k, aics[season], "o--", color='dimgray', label="AIC")
                    plt.xlabel("$k$", fontsize=14)
                    plt.ylabel("Information Criterion", fontsize=14)
                    plt.axis([1, 9.5, np.min(bics[season]) - 50, max(bics[season]) + 50])
                    plt.grid()

         
        else:
            gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42,covariance_type = 'full').fit(self.data_scaled)
                         for k in self.n_k]
            bics = [model.bic(self.data_scaled) for model in gms_per_k]
            aics = [model.aic(self.data_scaled) for model in gms_per_k]
        
        
            plt.figure()
            plt.plot(self.n_k, bics, "ko-", label="BIC")
            plt.plot(self.n_k, aics, "o--",color = 'dimgray', label="AIC")
            plt.xlabel("$k$", fontsize=14)
            plt.ylabel("Information Criterion", fontsize=14)
            plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
            plt.grid()
            plt.legend()
    
    def GM_run(self,k):
                
        if self.seasons_split:
            
            GM_model = dict()
            clusters = dict ()
            anomalies = dict()
            densities = dict()
            density_threshold = dict()
        
            for season in self.seasons:
                
                
                GM_model[season]=GaussianMixture(n_components=k, n_init=10, random_state=42,covariance_type = 'full').fit(
                    self.data_by_season_scaled[season])
                 
                if self.Direction:
                    
                    clusters[season] = pd.DataFrame(self.scaler.inverse_transform(GM_model[season].means_), columns=['Hs', 'Tp', 'Direction'])
                else :
                    clusters[season] = pd.DataFrame(self.scaler.inverse_transform(GM_model[season].means_), columns=['Hs', 'Tp'])
                 
                densities[season] = GM_model[season].score_samples(self.data_by_season_scaled[season])
                density_threshold[season] = np.percentile(densities[season], self.p_err)
                 
                if self.Direction:
                    
                    anomalies[season] = pd.DataFrame(self.scaler.inverse_transform(self.data_by_season_scaled[season][densities[season] < density_threshold[season]]),
                                                       columns= ['Hs', 'Tp', 'Direction'])
                else:
                    anomalies[season] = pd.DataFrame(self.scaler.inverse_transform(Data_range_scaled[seasons][densities[seasons] < density_threshold[seasons]]),
                                                     columns= ['Hs', 'Tp'])                                  
     
                 
            self.clusters = clusters
            self.anomalies = anomalies
        
        else:
            GM_model = GaussianMixture(n_components=k, n_init=10, random_state=42,covariance_type = 'full').fit(self.data_scaled)
        
            if self.Direction:
                cluster = pd.DataFrame(self.scaler.inverse_transform(GM_model.means_), columns=['Hs', 'Tp', 'Direction'])
            else:
                cluster = pd.DataFrame(self.scaler.inverse_transform(GM_model.means_), columns=['Hs', 'Tp'])            
        
            densities = GM_model.score_samples(self.data_scaled)
            density_threshold = np.percentile(densities, self.p_err)
        
            if self.Direction:
                anomalies = pd.DataFrame(self.scaler.inverse_transform(self.data_scaled[densities < density_threshold]),
                                         columns= ['Hs', 'Tp', 'Direction'])
            else:
                anomalies = pd.DataFrame(self.scaler.inverse_transform(self.data_scaled[densities < density_threshold]),
                                         columns= ['Hs', 'Tp'])
            
            self.clusters = cluster
            self.anomalies = anomalies
            
        
            
        
        
    
        
        
        

                
