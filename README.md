
Kaizan Haque


## Problem Statement

The goal of this project is to utilize public datasets that have been used for 
heart disease prediction and try and apply machine learning techniques such as: 

- K-Nearest Neighbors (KNN)

- Decision Trees (DT)

- Random Forests (RF)

- Naive Bayes (NB)
    
- Support Vector Machines (SVM)

in order to replicate these results and better cement my understanding of these 
algorithms and how to use them.

(disclaimer: this model should not be used for actual medical diagnosis, this
is just based off of a dataset a computer engineering student got their hands on
in order to further their understanding of how to properly use different 
machine learning models. Please go see an actual doctor if you have heart concerns)


## Motivation

After learning about various machine learning techniques through independent 
study, it came the the time for me to apply this knowledge in projects in order
to have project experience as well as practical application knowledge of how to 
use these algorithms.  

I knew I wanted to do a medical application of ML because I find it way more 
fulfilling to use ML for these purposes rather than for algorithms that increase
a consumer's engagement on some social media app or to recommend the next show
to binge on a streaming service. I picked heart disease specifically because
there are many people in my life that have heart disease issues, so this was a
relevant topic that ML could be applied to in an actually practical way.  

Once the topic was decided, I combed through 9 research papers online to see
what datasets and ML techniques they used, and also to familiarize myself with
the technical language that these ML researchers use to get a deeper 
understanding of what's going on. The conclusions from this reading was that
the algorithms listed in the Problem Statement were the most commonly tested
ones, so those are the algorithms I decided to use. The papers I used are 
linked at the bottom in the References section.


## Datasets
    
The [Cleveland](https://archive.ics.uci.edu/dataset/45/heart+disease) dataset 
from University of Chicago - Irvine was by far the most popular dataset used in
papers, so this is the one I chose to use. It was used by two papers in IEEE
and two other papers as well. 

Some other notable datasets that were used in these papers include:
        
- [Statlog](https://archive.ics.uci.edu/dataset/145/statlog+heart) which is also from UCI

- [Framingham](https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data)
from Kaggle

- [SPECTF](https://archive.ics.uci.edu/dataset/96/spectf+heart) from UCI

- [Ulianova](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
from Kaggle


## Data Exploration

After importing the dataset with the Python code on the UCI website of the 
dataset, now came the time for data cleaning and getting an overall feel for 
the data. There are 13 columns of various features and 303 rows according to 

```X.describe``` 

where X was the dataset with 13 features: 

- age (in years)
- sex (1=male, 0=female)
- cp (chest pain type, 1=typical angina, 2=atypical, 3=non-anginal pain, 4=asymptomatic)
- trestbps (resting blood pressure in mm Hg)
- chol (serum cholesterol in mg/dl
- fbs (fasting blood sugar > 120 mg/dl, 1=true, 0=false)
- restecg (resting ecg results, 0=normal, 1=ST-T wave abnormality, 2=probable or definite left ventricular hypertrophy)
- thalach (maximum heart rate achieved)
- exang (exercise induced angina, 1=yes, 0=no)
- oldpeak (ST depression induced by exercise relative to rest) 
- slope (slope of the peak exercise ST segment, 1=upward slope, 2=flat, 3=down)
- ca (number of major vessels 0-3 colored by fluoroscopy)
- thal (3=normal, 6=fixed defect, 7=reversable defect)

The variable ```y``` on the other hand represents a 14th feature, num, which is
going to be the output of the model to diagnose heart disease . Here, a value 
of 0 indicates <50% diameter narrowing in any major vessel of the cardiovascular
system, thus a negative heart disease prediction. A value of 1 would indicate >50%
diameter narrowing and thus would indicate a prediction of heart disease for the individual.


## K-Nearest Neighbors

The strategy for this one was to clean the dataset by removing the na values 
then use k-fold cross-validation to train the sets. For the value of k, we tried
5 and 10 because, after some research, it looked like those were the most common
values for k to use. 

Next, we set up a for loop to try different values of k for KNN (which uses 
a different number of the nearest neighbors to classify a data point) and put
the upper limit of k to the square root of the total number of points because my
research said that was the default assumption if nothing else is known and I 
wanted to make sure I avoided overfitting. This for loop will be run twice: once
each for 5-fold and 10-fold.

The program produces a new confusion matrix every time the file is run, but the
sample one in the GitHub is a solid representation of what the results are every
time.

I ran the model a few times with 5 and 10 folds and got a little over
54% accuracy every time with 10-fold cross-validation whlie 5-fold was around
53%, so the final model uses 10-fold. The 10-fold model usually used around 15 
neighbors. Looking at the confusion matrix, the model often predicted 0s but got 
a lot of those right so that just means that the dataset has a lot of 0 datapoints
which represents people who don't have cardiovascular disease.


## Decision Tree

We used the same strategy as KNN with the cleaning and k-fold cross-validation
to keep things consistent.

We changed around different parameters in the constructor for the DecisionTreeClassifier, 
experimenting to see how they impacted results. We also found that 10-fold cross-validation 
gave better accuracy than 5-fold. When changing around min_samples_leaf, we found
that at a low number like 5, the predictions were more spread out but the accuracy
went down to 48-49%. When the number was high like 200, we got a higher accuracy 
around 55% but pretty much all of the predictions are 0. Part of this is probably
due to the observation from earlier where most of the targets are actually 0,
so the safest guess for the model is 0 and that gives the highest accuracy which
is unfortunate. If we were to combat this, we would probably remove some of the
features with 0 so there was an equal number of 0, 1, 2, 3, 4 or at least closer
to an equal amount. 

We settled on a min_samples_leaf value of 11 because it kind of balanced the 
spread out predictions, instead of only predicting 0's, and had decent accuracy 
(roughly 53%)


## References

1. [S. Mohan, C. Thirumalai and G. Srivastava, "Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques," in IEEE Access, vol. 7, pp. 81542-81554, 2019, doi: 10.1109/ACCESS.2019.2923707](https://ieeexplore.ieee.org/abstract/document/8740989)

2. [Marelli, A, Li, C, Liu, A. et al. Machine Learning Informed Diagnosis for Congenital Heart Disease in Large Claims Data Source. JACC Adv. 2024 Feb, 3 (2)](https://doi.org/10.1016/j.jacadv.2023.100801)

3. [J. P. Li, A. U. Haq, S. U. Din, J. Khan, A. Khan and A. Saboor, "Heart Disease Identification Method Using Machine Learning Classification in E-Healthcare," in IEEE Access, vol. 8, pp. 107562-107582, 2020, doi: 10.1109/ACCESS.2020.3001149.](https://ieeexplore.ieee.org/abstract/document/9112202)

4. [Shah, D., Patel, S. & Bharti, S.K. Heart Disease Prediction using Machine Learning Techniques. SN COMPUT. SCI. 1, 345 (2020).](https://doi.org/10.1007/s42979-020-00365-y)

5. [Rani, P., Kumar, R., Jain, A. et al. An Extensive Review of Machine Learning and Deep Learning Techniques on Heart Disease Classification and Prediction. Arch Computat Methods Eng (2024).](https://doi.org/10.1007/s11831-024-10075-w)

6. [Bhatt CM, Patel P, Ghetia T, Mazzeo PL. Effective Heart Disease Prediction Using Machine Learning Techniques. Algorithms. 2023; 16(2):88.](https://doi.org/10.3390/a16020088)

7. [Das, S., Nayak, S.P., Sahoo, B. et al. Machine Learning in Healthcare Analytics: A State-of-the-Art Review. Arch Computat Methods Eng (2024).](https://doi.org/10.1007/s11831-024-10098-3)
    
8. [Luca Brunese, Fabio Martinelli, Francesco Mercaldo, Antonella Santone, Machine learning for coronavirus covid-19 detection from chest x-rays, Procedia Computer Science, Volume 176, 2020, Pages 2212-2221, ISSN 1877-0509,](https://doi.org/10.1016/j.procs.2020.09.258)

9. [Wankhede J, Kumar M, Sambandam P. Efficient heart disease prediction-based on optimal feature selection using DFCSS and classification by improved Elman-SFO. IET Syst Biol. 2020 Dec;14(6):380-390. doi: 10.1049/iet-syb.2020.0041. PMID: 33399101; PMCID: PMC8687167.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8687167/)
