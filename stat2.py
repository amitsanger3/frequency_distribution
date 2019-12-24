import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import math
import scipy as stats
import scipy.stats

from pystat import *


class Sampling_Distibution(Probability_Distribution):
    """
    Stratified Sampling: use when each group has small variation
        within itself but there is a wide variation between the 
        groups.
        
    Cluster Sampling: use when there is considerable variation
        within each group but the groups are essentially similar
        to each other.
    """
    
    def __init__(self):
        pass
    
    
    def standard_error(self, sigma, n):
        """
        To  get sampling distribution standard deviation
        for infinite population or from finite sample with
        replacements.
        
        
        sigma: population standard deviation
        n: sample size
        """
        return sigma/(math.sqrt(n))
    
    
    def probability(self, mu, sigma, n, N=0, X1=-math.inf, X2=math.inf, infinite=True):
        """
        mu = population mean
        sigma = population standard deviation
        infinte = bool, if true then find probaility for infinite 
        population, otherwise for finite population

        """
        if infinite is True:
            std_error = self.standard_error(sigma, n)
        elif infinite is False:
            std_error = self.standard_error_finite(sigma, N, n)
        print("std error:", std_error)
        
        return self.normal_distribution(mu, std_error, X1, X2)
    
    
    """
    CENTRAL LIMIT THEOREM:
    
    1. The mean of the sampling distribution of the mean
        will be equal to the population mean.
    2. As the sample size increases, the sampling distribution
        of the mean will approach normality.
        
    Significance:
        it permits use sample statics to make inferences about
        population parameterswithout knowing anything about
        the shape of the frequency distribution of that population
        other than what we can get from the sample.
    """
        
    def finite_population_multiplier(self, N, n):
        """
        N: Size of the population
        n: Size of sample
        
        Note: When sampling fraction(n/N) is less than 0.05, the
            finite population multiplier need not be used.
        """
        return math.sqrt((N-n)/(N-1))
    
    """
    Ststistician reccommend that in estimation, n be large
    enough, atlest n/N > 0.5 to use Normal distribution
    as a substitute for the binomial distribution.
    """
    
    def standard_error_finite(self, sigma, N, n):
        """
        To  get sampling distribution standard deviation
        for finite sample without replacements
        
        sigma: population standard deviation
        N: Size of the population
        n: Size of sample
        
        if n/N > 0.05 is True then use standard_error_finite
        otherwise use standard_error
        """
        return self.standard_error(sigma, n) * self.finite_population_multiplier(N, n)
    

class Estimation(Sampling_Distibution):
    
    def __init__(self, x):
        """
        x = np.array of samples
        """
        self.sample = x
    
    
    def point_estimate_mean(self):
        return (self.sample.sum())/len(self.sample)
        
        
    def point_estimate_variance(self):
        variance = ((self.sample - self.point_estimate_mean())**2).sum()/(len(self.sample)-1)
        return variance
        
    
    def point_estimate_deviation(self):
        return math.sqrt(self.point_estimate_variance())
    
    
    def interval_estimate(self, mu, sigma, n, N=0, infinite=True):
        """
        68–95–99.7 rule:
        
        In statistics, the 68–95–99.7 rule, also known as the empirical rule,
        is a shorthand used to remember the percentage of values that lie 
        within a band around the mean in a normal distribution with a width 
        of two, four and six standard deviations, respectively; more accurately,
        68.27%, 95.45% and 99.73% of the values lie within one, two and three 
        standard deviations of the mean, respectively.

        In mathematical notation, these facts can be expressed as follows, where
        Χ is an observation from a normally distributed random variable, μ is 
        the mean of the distribution, and σ is its standard deviation:
        
        Pr(μ - σ ≤ X ≤ μ + σ) ≈ 0.6827
        Pr(μ - 2σ ≤ X ≤ μ + 2σ) ≈ 0.9545
        Pr(μ - 2σ ≤ X ≤ μ + 2σ) ≈ 0.9973
        """
        #n = len(self.sample)
        result = []
        for i in range(1,4):
            if infinite is True:
                t = (mu - i*self.standard_error(sigma,n), mu + i*self.standard_error(sigma,n))
            elif infinite is False:
                t = (mu - i*self.standard_error_finite(sigma, N, n), mu + i*self.standard_error_finite(sigma, N, n))
                
            result.append(t)
        
        return result
    
    
    def standard_error_from_mean(self, confidence_lavel):
        """
        confidence_lavel = percentage of confidence
        """
        
        file = 'cumulative_from_mean_0toZ.csv'
        standard_table = self.standard_normal_table(file, 'z')


        dfr = standard_table[standard_table < confidence_lavel/200]
        max_columns = dfr.idxmax(axis=1)
        index = max_columns.last_valid_index()
        column = max_columns.loc[index]

        return round(index + float(column), 2)
    
    
    def confidence_interval_by_std_error(self, confidence, mean, std_error):
        """
        confidence = percentage
        mean = sample mean or population mean
        std_error = 
        """
        return (mean - (confidence * std_error), mean + (confidence * std_error)) 
    
    
    def confidence_interval(self, confidence_lavel, x_bar, sigma, n, N=0, infinite=True):
        """
        confidence_lavel = percentage of confidence
        x_bar: sample mean
        sigma: standard deviation
        """

        conf = self.standard_error_from_mean(confidence_lavel)
        
        if infinite is True:
            std_err = self.standard_error(sigma,n)
        elif infinite is False:
            std_err = self.standard_error_finite(sigma, N, n)
        
        return self.confidence_interval_by_std_error(conf, x_bar, std_err)
    
    """
    Ststistician reccommend that in estimation, n be large
    enough, atlest n/N > 0.5 to use Normal distribution
    as a substitute for the binomial distribution.
    """
    
    #========================================================
    # INTERVAL ESTIMATES OF THE PROPORTION FROM LARGE SAMPLES
    #========================================================
    
    def estimate_mean_of_proportion(self, p_bar):
        """
        p_bar = sample proportion in favour
        """
        return p_bar
    
    
    def estimate_standard_error_of_proportion(self, p_bar, q_bar, n, N=0, infinite=True):
        """
        p_bar = sample proportion in favour
        q_bar = sample proportion not in favour
        n = sample size
        N = Population mean
        """
        
        if infinite is True:
            return math.sqrt((p_bar * q_bar)/n)
        elif infinite is False:
            return math.sqrt((p_bar * q_bar)/n) * self.finite_population_multiplier(N, n)
    
    
    
    def confidence_interval_of_proportion(self, confidence, p_bar, q_bar, n, N=0, infinite=True):
        """
        confidence = percentage
        p_bar = sample proportion in favour
        q_bar = sample proportion not in favour
        n = sample size
        """
        return self.confidence_interval_by_std_error(self.standard_error_from_mean(confidence), p_bar, self.estimate_standard_error_of_proportion(p_bar, q_bar, n, N, infinite))
    
    
    def t_distribution(self, x_bar, sigma_hat, t):
        """
        x_bar = sample mean
        sigma_hat = sample standard deviation
        t = value
        
        Condition of usage:
            1. sample <= 30
            2. population standard deviation is unknown
            3. assume: population is nomal or approximately normal
        """
        return x_bar - (sigma_hat * t), x_bar + (sigma_hat * t)
    
    
    def degree_of_freedom(self, n):
        return n-1
    
    
    def t_value(self, file, n, confidence_interval):
        
        t_table = self.standard_normal_table(file, 'df')
        
        return t_table.loc[str(self.degree_of_freedom(n))][str(confidence_interval)]
    
    
    def t_value_confidence(self, n, confidence_interval):
        conf = str(confidence_interval)+"%"
        return self.t_value('t-table-confidence.csv', n, conf)
    
    
    def t_value_cumulative(self, n, confidence_interval):
        return self.t_value('t-table-cumulative.csv', n, confidence_interval)
    
    
    def t_value_one_tail(self, n, confidence_interval):
        return self.t_value('t-table-one-tail.csv', n, confidence_interval)
    
    
    def t_value_two_tail(self, n, confidence_interval):
        return self.t_value('t-table-two-tail.csv', n, confidence_interval)
    
    
    def confidence(self, n, t_val, file):
        t_table = self.standard_normal_table(file, 'df')
        
        t_val_series = round(t_table.loc[str(self.degree_of_freedom(n))], 3)
        t_index = t_val_series[t_val_series == t_val].index

        return t_index[0]
    
    
    def confidence_without_sample_size(self, t_val, file):
        t_table = round(self.standard_normal_table(file, 'df'), 3)
        
        t_table_na = t_table[t_table == t_val]
        t_tab = t_table_na.fillna(0.0)
        
        t_max = t_tab.max()
        t_index = t_max[t_max == t_val].index

        return int(t_tab.idxmax().max())+1, t_index[0]
    
        
    def confidence_lavel_from_t_value(self, n, t_val):
        file = 't-table-confidence.csv'
        return self.confidence(n, t_val, file)
    
    def confidence_lavel_from_t_value_without_sample_size(self, t_val):
        file = 't-table-confidence.csv'
        return self.confidence_without_sample_size(t_val, file)
    
    
    def one_tail_confidence_level_from_t_value(self, n, t_val):
        file = 't-table-one-tail.csv'
        return self.confidence(n, t_val, file)
    
    def one_tail_confidence_level_from_t_value_without_sample_size(self, t_val):
        file = 't-table-one-tail.csv'
        return self.confidence_without_sample_size(t_val, file)
    
    
    def two_tail_confidence_level_from_t_value(self, n, t_val):
        file = 't-table-two-tail.csv'
        return self.confidence(n, t_val, file)
    
    def two_tail_confidence_level_from_t_value_without_sample_size(self, t_val):
        file = 't-table-two-tail.csv'
        return self.confidence_without_sample_size(t_val, file)
    
    
    def cumulative_confidence_level_from_t_value(self, n, t_val):
        file = 't-table-cumulative.csv'
        return self.confidence(n, t_val, file)
    
    def cumulative_confidence_level_from_t_value_withou_sample_size(self, t_val):
        file = 't-table-cumulative.csv'
        return self.confidence_without_sample_size(t_val, file)
    
    
# ===========================================  
 # Determining the sample size of estimation
# ===========================================

    def sample_size_mean(self, t_val_mean, z_val, sigma):
        
        return ((z_val*sigma)/t_val_mean)**2
    
    
    def sample_size_proportion(self, t_val, z_val, p, q):
        
        return ((z_val * math.sqrt(p*q))/t_val)**2
    
    
# ===========================================  
 # Confidence level associated with interval
# ===========================================

    def confidence_level_from_z(self, z_val):
        
        return self.probability_of_normal_distribution(0, z_val) * 200
