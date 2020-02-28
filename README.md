# Waves_Tools

<i> Module contains tools for analyze ocean wave timeseries. Unsurpervised clustering algorithms have been also implemented for classify the ocean wave, according to the caracteristics Hs (Wave amplitude [m]), Tp (Wave period [s]) and Direction (direction of the train wave [°]).</i>

# How it works

<blockquote> class <b>Wave_Tools</b>(data, seasons_split=False,clustering=None, n_k=None, p_err = None) </blockquote>
<h3> Parameters </h3>
</br>
<blockquote> data :: (DataFrame)</blockquote> 
 <i> DataFrame contains two or Three columns Hs Tp  and Direction, need to be analysed. The variables Direction can be missing.</i>
</br>
</br>
<blockquote> seasons_split :: (Bool)</blockquote>
<i> Activate the seasons_split option. If activate the timeserie will be grouped by season : winter, summer, spring and autumn.</i>
</br>
</br>
<blockquote> clustering :: (str)</blockquote>
<i> Activate the clustering algorithms. <b>clustering = 'Kmean'</b> for the Kmean method and <b>clustering = 'GM'</b> for the Gaussian mixture model.
The Gaussian mixture model incorporates the anomaly Detection for the timeserie. The keyword <b>p_err</b> controls the anomalie selection.</i>
</br>
</br>
<blockquote> n_k :: (range)</blockquote>
number of k clusters tested for fint the correct one. if the option clustering is activated by default n_k = range(2, 10).</i>
</br>
</br>
<blockquote> p_err :: (int or float)</blockquote>
<i> p_err controls the anomalie selection. A p_err equal to 1 means that approximately 1 % of the instances will be flagged as anomalies. by default p_err = 1 </i>
</br>
</br>
</br>
<h3> Attributes </h3>
</br>
<blockquote>_freq_t()</blockquote>
<i>Return the frequencies of occurrence for each feature and the cumulative frequencies.</i>
</br>
</br>
<blockquote>plot_frequency()</blockquote>
<i>Return the plots of occurrences and the cumulative frequencies.</i>
</br>
</br>
<blockquote>plot_correlogram()</blockquote>
<i>Return the correlogram plots between Hs/Tp, Hs/Direction and Tp/Direction.</i>
</br>
</br>
<blockquote>Kmeans_setup()</blockquote>
<i>Return plots Inertia vs number of clusters k and Silouette_score vs number of clusters k.</i>
</br>
</br>
<blockquote>Kmeans_run( k )</blockquote>
<i>Return DataFrame containing the controids of each cluster (k).</i>
</br>
</br>
<blockquote>GM_setup()</blockquote>
<i>Return plots Information Criterion for each number of cluster.</i>
</br>
</br>
<blockquote>GM_run( k )</blockquote>
<i>Return DataFrame containing the mean controids of each cluster(k) and DataFrame containing the anomalies detected</i>

# User Guide

Let's using a dataset from wave modelling propagation from IFREMER (French institut).

<pre><code> 
print(df_Data)

Out[]: 
                           Hs        Tp  Direction
Time                                              
2016-01-01 00:00:00  2.355760  13.87020    245.966
2016-01-01 01:00:00  2.327610  13.87020    245.675
2016-01-01 02:00:00  2.317590  13.87020    245.664
2016-01-01 03:00:00  2.317530  13.88890    245.944
2016-01-01 04:00:00  2.335020  13.88890    246.426
                      ...       ...        ...
2018-12-30 19:00:00  0.468191  10.00000    254.370
2018-12-30 20:00:00  0.489997   9.90099    254.241
2018-12-30 21:00:00  0.509803   9.80392    254.221
2018-12-30 22:00:00  0.525608   9.70874    254.293
2018-12-30 23:00:00  0.537219  12.65820    254.281

[26280 rows x 3 columns]</code></pre>

<i>Then, for plot the occurrences and the cumulative frequencies whitout the classification and for all the data.</i>

<pre><code> 
Wave_analyse= Wave_Tools(df_Data, seasons_split = False)
Wave_analyse.plot_frequency()
</code></pre>
<img src ="Images/Figure_1.png">

<i> plot the correlogram </i>
<pre><code> 
Wave_analyse.plot_correlogram()
</code></pre>
<img src ="Images/Figure_2.png" width="20" >
<img src ="Images/Figure_3.png">
<img src ="Images/Figure_4.png">

