----------------------------------------------------------------------------
Does Air Quality Really Impact COVID-19 Clinical Severity:
Coupling NASA Satellite Datasets with Geometric Deep Learning
----------------------------------------------------------------------------

![alt text](https://github.com/Covid-19-papers/Does-Air-Quality-Really-Impact-COVID-19-Clinical-Severity/blob/main/Atmospheric_Variables.png?raw=true)

*Given that persons with a prior history of respiratory diseases tend to demonstrate more severe illness from COVID-19 and, hence, are at higher risk of serious symptoms, ambient air quality data from NASA’s satellite observations might provide a critical insight into how the expected severity of COVID-19 and associated survival rates may vary across space in the future.* 

*The goal of this project is to glean a deeper insight into sophisticated spatio-temporal dependencies among air quality, atmospheric conditions, and COVID-19 clinical severity using the machinery of Geometric Deep Learning (GDL), while providing quantitative uncertainty estimates. Our results based on the GDL model on a county level in three US states, California, Pennsylvania and Texas, indicate that AOD attributes to COVID-19 clinical severity. Our findings do not only contribute to understanding of latent factors behind COVID-19 progression but open new perspectives for innovative use of NASA’s datasets for biosurveillance and social good.*

This package includes the source codes and datasets used in this research. 
We encourage the reader to review [our open access paper](https://doi.org/10.1145/3447548.3467207) presented in [KDD 2021](https://kdd.org/kdd2021/).

The complete software list and requirements are included in the file "Requirements.txt".
Many thanks to [Benedek Rozemberczki et al](https://arxiv.org/abs/2104.07788), head of project [Pytorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal).

Since the methodology was tested on three different US States (California, Pennsylvania and Texas), we include three folders in this package. Each folder contains the datasets of each variable used in the paper.

Results are reproducible by running python source code: Source_Code.py



