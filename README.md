# Waves_Tools

<i> Module contains tools for analyze ocean wave timeseries. Unsurpervised clustering algorithms have been also implemented for classify the ocean wave, according to the caracteristics Hs (Wave amplitude [m]), Tp (Wave period [s]) and Direction (direction of the train wave [°]).</i>

# How it works

<blockquote> class <b>Wave_Tools</b>(data, seasons_split=False,clustering=None, n_k=None, p_err = None) </blockquote>
<h3> Parameters </h3>
</br>
<blockquote> data :: (DataFrame)</blockquote> 
 <i> DataFrame contains two or Three columns Hs Tp  and Direction, need to be analysed </i>
</br>
<blockquote> seasons_split :: (Bool)</blockquote>
<i> Activate the seasons_split option. If activate the timeserie will be grouped by season : winter, summer, spring and autumn.</i>
</br>
<blockquote> clustering :: (str)</blockquote>
<i> Activate the clustering algorithms. <b>clustering = 'Kmean'</b> for the Kmean method and <b>clustering = 'GM'</b> for the Gaussian mixture model.
The Gaussian mixture model incorporates the anomaly Detection for the timeserie. The keyword <b>p_err</b> controls the anomalie selection. 
</br>
<blockquote> n_k :: (range)</blockquote>
number of k clusters tested for fint the correct one. if the option clustering is activated by default n_k = range(2, 10).
</br>
<blockquote> p_err :: (int or float)</blockquote>
<i> p_err controls the anomalie selection. A p_err equal to 1 means that approximately 1 % of the instances will be flagged as anomalies. by default p_err = 1

