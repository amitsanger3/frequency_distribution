#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import math
import scipy as stats



#get_ipython().run_line_magic('matplotlib', 'inline')
#####################################################################################################################################

class Relative_Frequency(object):
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.width_of_class_interval = None
        self.data_range = None
        self.number_of_intervals = None

        
    def get_data_range(self, number_of_intervals):
        self.width_of_class_interval = self. get_width_of_class_interval(number_of_intervals)
        self.data_range = np.arange(min(self.dataset), max(self.dataset)+self.width_of_class_interval, self.width_of_class_interval).round(decimals=1)
        return self.data_range
        
        
    def get_custom_data_range(self, start_point, end_point, width_of_class_interval):
        self.data_range = np.arange(start_point, end_point+width_of_class_interval, width_of_class_interval)
        return self.data_range
        
        
    def get_width_of_class_interval(self, number_of_intervals):
        self.number_of_intervals = number_of_intervals
        self.width_of_class_interval = round(((math.ceil(max(self.dataset))-min(self.dataset))/(self.number_of_intervals)), ndigits=1)
        if self.width_of_class_interval < 0.2:
            self.get_width_of_class_interval(number_of_intervals-1)
        else:
            return self.width_of_class_interval
        
        
    def get_custom_width_of_class_interval(self, start_point, end_point, number_of_intervals):
        self.width_of_class_interval = round((end_point - start_point)/(number_of_intervals))
        return self.width_of_class_interval
        
        
    def get_number_of_intervals(self, width_of_class_interval=0.0):
        self.width_of_class_interval = width_of_class_interval
        if self.width_of_class_interval == 0.0:
            return len(self.dataset)
        else:
            return round(((math.ceil(max(self.dataset))-min(self.dataset))/(width_of_class_interval)), ndigits=1)
        
        
    def frequency(self, data_range):
        self.data_range = data_range
        result = []
        for i in range(len(self.data_range)-1):
            if i ==0:
                x1 = self.dataset[self.data_range[i]<=self.dataset]
                x2 = x1[x1<=self.data_range[i+1]]
                result.append(len(x2))
            else:
                x1 = self.dataset[self.data_range[i]<self.dataset]
                x2 = x1[x1<=self.data_range[i+1]]
                result.append(len(x2))
        return result

    
    def open_frequency(self, data_range):
        self.data_range = data_range
        result = []
        for i in range(len(self.data_range)+1):
            if i == 0:
                x1 = self.dataset[self.data_range[i]>self.dataset]
                result.append(len(x1))
            elif i == len(self.data_range):
                x1 = self.dataset[self.data_range[i-1]<self.dataset]
                result.append(len(x1))
            else:
                x1 = self.dataset[self.data_range[i-1]<=self.dataset]
                x2 = x1[x1<=self.data_range[i]]
                result.append(len(x2))
        
        return result
    
    
    def interval(self):#, data_range):
        #self.data_range = data_range
        result = []
        for i in range(len(self.data_range)-1):
            if i == 0:
                result.append(str(round(self.data_range[i], ndigits=1))+'-'+str(round(self.data_range[i+1], ndigits=1)))
            elif i == len(self.data_range)-2:
                result.append(str(round(self.data_range[i], ndigits=1))+'-'+str(round(max(self.dataset), ndigits=1)))
            else:
                result.append(str(round(self.data_range[i]+0.1, ndigits=1))+'-'+str(round(self.data_range[i+1], ndigits=1)))
        
        return result
    
    
    def open_interval(self):#, data_range):
        #self.data_range = data_range
        result = []
        for i in range(len(self.data_range)+1):
            if i == 0:
                result.append('infinity-'+str(round(self.data_range[i], ndigits=1)))
            elif i == len(self.data_range):
                result.append(str(round(max(self.dataset), ndigits=1))+'-infinity')
            else:
                result.append(str(round(self.data_range[i-1]+0.1, ndigits=1))+'-'+str(round(self.data_range[i], ndigits=1)))
        
        return result
    
    
    def classification(self, data_range, close=True):
        if close is True:
            f = self.frequency(data_range)
            result = {
                "Interval": self.interval(),
                "Frequency": f,
                "%": np.array(f)/len(self.dataset),
            }
        else:
            f = self.open_frequency(data_range)
            result = {
            "Interval": self.open_interval(),
            "Frequency": f,
            "%": np.array(f)/len(self.dataset),
        }
            
        return pd.DataFrame(result)
    
    
    def evaluation(self, data_range, close=True):
        rep = self.classification(data_range, close) # change
        fig = plt.figure(figsize=(15.0, 6.0))
        axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
        axes1.plot(rep["Interval"], rep["Frequency"])
        #axes1.bar(np.arange(0,len(rep['Interval']),1), height=rep['Frequency'])
        #sns.barplot(rep.index.to_list(), rep['Frequency'])
        #print(" ")
        #print("Frequency Distribution Plot")
        #sns.distplot(rep['Frequency'])
        
        
    def commulative_evaluation(self, data_range, more_than=True):
        self.data_range = data_range
        data_frequency = np.array(self.frequency(self.data_range))
        #more_than 
        if more_than is True:
            total = data_frequency.sum()
            commulative_frequency_distribution = [total]
            for i in data_frequency:
                total -= i
                commulative_frequency_distribution.append(total)
        # less than
        else:
            start_total = 0
            commulative_frequency_distribution = [start_total]
            for i in data_frequency:
                start_total += i
                commulative_frequency_distribution.append(start_total)

        # presentation
        fig = plt.figure(figsize=(15.0, 6.0))
        axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
        axes1.plot(data_range, commulative_frequency_distribution)



#####################################################################################################################################

class Measure_of_Central_Tendency(Relative_Frequency):
    
    def __init__(self, dataset):
        Relative_Frequency.__init__(self, dataset)
        #relative_frequency.__init__(self, dataset)
        self.data_range_mid_point = None
        self.weights = None
        
    def get_data_range_mid_point(self, data_range):
        """
        if class are given in a manner like
        [1-3,4-6,7-9,10-12,13-15,16-18,19-21,22-24]
        
        then, our datarange is
        [1,4,7,10,13,16,19,22,25]
        """
        self.data_range_mid_point = [round((data_range[i]+data_range[i+1])/2) for i in range(len(data_range)-1)]
        return np.array(self.data_range_mid_point)
    
    def arithmetic_mean_by_frequency_distribution(self, data_range):
        x = self.get_data_range_mid_point(data_range)
        f = np.array(self.frequency(data_range))
        sum_fx = (f*x).sum()
        print("fx: ", sum_fx)
        sum_f = f.sum()
        print("f: ", sum_f)
        
        return sum_fx/sum_f
        
        
    def arithmetic_mean_by_raw_dataset(self):
        return self.dataset.mean()
    
    
    def weighted_mean(self, weights):
        self.weights = weights
        return (self.dataset*self.weights).sum()/self.weights.sum()
    
    
    def geometric_mean(self, change_factors, increase=True):
        """
        if direct % is given then, first multiply the array with 0.01
        """
        if increase is True:
            change_factors = change_factors + 1
        
        product_change_factors = change_factors.cumprod()
                
        return (product_change_factors)**(1/len(change_factors)) - 1
    
    
    
    def get_change_factor(self, increase=True):
        change_factors = []
        for i in range(len(self.dataset)-1):
            change_factors.append(self.dataset[i+1]/self.dataset[i])
            
        if increase is True:
            return np.array(change_factors) - 1
        else:
            return np.array(change_factors)
        
        
    def estimate_percentage_change(self, geometric_mean, time):
        return (1 + geometric_mean)**(time) - 1
        
            
    def median_by_raw_data(self):     
        return np.median(self.dataset)
        
        
    def median_by_frequency_distribution(self):
        f = np.array(self.frequency(self.data_range))
        cum_f = f.cumsum() # Cumulative sum of frequency
        n = np.array(f).sum()
        center_elemnt = (n+1)/2
        if n%2 == 0:
            center1,center2 = math.floor(center_elemnt), math.ceil(center_elemnt)
            class_num = len(cum_f[cum_f<center_elemnt]) # Class index number of which median belong
            fnum = cum_f[class_num] # frequency num of which median belong
            data_range_lower_num, data_range_upper_num = self.data_range[class_num], self.data_range[class_num+1]
            step_width = (data_range_upper_num - data_range_lower_num)/fnum
            center1_item = (step_width * (center1 - 1)) + data_range_lower_num
            center2_item = (step_width * (center2 - 1)) + data_range_lower_num
            median = (center1_item + center2_item)/2
    
        else:
            center = center_elemnt
            class_num = len(cum_f[cum_f<center_elemnt]) # Class index number of which median belong
            fnum = cum_f[class_num] # frequency num of which median belong
            data_range_lower_num, data_range_upper_num = self.data_range[class_num], self.data_range[class_num+1]
            step_width = (data_range_upper_num - data_range_lower_num)/fnum
            median = (step_width * (center - 1)) + data_range_lower_num
            
        return median
    
    
    def mode_by_raw_data(self):
        """
        if dataset takes smaller value having larger number of repeats
        """
        unique_data_set = np.unique(self.dataset, return_counts=True)
        return unique_data_set[0][unique_data_set[1].tolist().index(unique_data_set[1].max())], unique_data_set[1].max()
        
    
    def mode_by_frequency_distribution(self):
        """
        if frequency take the larger value
        """
        f = np.array(self.frequency(self.data_range))
        cum_f = f.cumsum() # Cumulative sum of frequency
        n = np.array(f).sum()
        center_elemnt = (n+1)/2
       
        class_num = len(cum_f[cum_f<center_elemnt]) # Class index number of which median belong
        fnum = cum_f[class_num] # frequency num of which median belong
        data_range_lower_num, data_range_upper_num = self.data_range[class_num], self.data_range[class_num+1]
        
        data_range_width = data_range_upper_num - data_range_lower_num
        d1 = f[class_num] - f[class_num+1]
        d2 = f[class_num] - f[class_num-1]
        mode = data_range_lower_num + ((d1/(d1+d2))*data_range_width)
        
        return mode
            



#####################################################################################################################################


class Measure_of_Dispersion(object):
   

    def __init__(self, dataset):
        self.dataset = dataset
        
        
    def get_dataset(self):
        return self.dataset
    
    
    def set_dataset(self, new_dataset):
        self.dataset = new_dataset
    #==============================
        # Range 
    #==============================

    
    def get_range(self):
        return max(self.dataset)-min(self.dataset)
    
    
    def percentile(self, fraction):
        return sorted(self.dataset)[math.floor(len(self.dataset)*fraction) - 1]
    
    
    def interfractile_range(self, lower_fractile, upper_fractile):
        return self.percentile(upper_fractile) - self.percentile(lower_fractile)
    
    
    def interquartile_range(self):
        return self.interfractile_range(0.25, 0.75)
        
    
    #==============================
        # Average Deviation Measure
    #==============================

    
    def population_variance(self):
        return ((self.dataset**2).sum() / len(self.dataset)) - (self.dataset.mean())**2
    
    
    def population_standard_daviation(self):
        return (self.population_variance())**(1/2)
    
    
    def standard_score(self, x):
        return (x - self.dataset.mean()) / self.population_standard_daviation()
    
    
    def population_mean_by_frequency_distribution(self, frequency):
        return ((self.dataset * frequency).sum())/(frequency.sum())
    
    
    def population_variance_by_frequency_distribution(self, frequency):
        return ((frequency * (self.dataset**2)).sum() / (frequency.sum())) - (self.population_mean_by_frequency_distribution(frequency))**2
        
        
    def population_standard_daviation_by_frequency_distribution(self, frequency):
        return (self.population_variance_by_frequency_distribution(frequency))**(1/2)
    
    
    def sample_variance(self):
        return ((self.dataset - self.dataset.mean())**2).sum() / (len(self.dataset) - 1)
    
    
    def sample_standard_deviation(self):
        return (self.sample_variance())**(1/2)
    
    
    def sample_standard_score(self, x):
        return (x - self.dataset.mean()) / self.sample_standard_deviation()
    
    
    #==============================
        # Relative Dispersion
    #==============================

    
    def coefficient_of_variation(self, sigma, mu):
        """
        lesser is better
        """
        return (sigma/mu)*100



#####################################################################################################################################

class Probability(object):
    
    def __init__(self):
        pass
    
    
    def get_classical_probability(self, A, N):
        """
        A: Number of favourable outcomes
        N: Total number of possible outcomes
        """
        return round(A/N, 2)
    
    
    #====================================
    # UNDER STATISTICAL INDEPENDENCE 
    #====================================
    def marginal_probability(self, probabilities_of_events_happening, probility_of_events_happening_together=0.0):
        """
        probabilities_of_events_happening: numpy array or list
        """
        return round(np.array(probabilities_of_events_happening).sum() - probility_of_events_happening_together, 2)
    
    
    def joint_probability(self, probabilities_of_events_happening):
        return round(np.array(probabilities_of_events_happening).prod(),2)
    
    
    def conditional_probabilities(self, probability_of_required_event_happening):
        """
        as events are independents 
        """
        return probability_of_required_event_happening
    
    
    #====================================
    # UNDER STATISTICAL DEPENDENCE 
    #====================================
    def conditional_probabilities_under_statistical_dependence(self, probability_of_given_event_happening, probability_of_given_and_required_event_happening_together):
        return round(probability_of_given_and_required_event_happening_together/probability_of_given_event_happening, 2)
    
    
    def joint_probabilities_under_statistical_dependence(self, probability_of_given_event_happening, probability_of_conditional_event_happening_together):
        return round(probability_of_conditional_event_happening_together*probability_of_given_event_happening, 2)
    
    
    def marginal_probability(self, probabilities_of_all_joint_events_happening):
        """
        probabilities_of_events_happening: numpy array or list
        """
        return round(np.array(probabilities_of_events_happening).sum(), 2)
    
    
    #====================================
    # PRIOR ESTIMATES OF PROBABILITIES
    #====================================
    def bayes_therorem_probability(self, probabilities_of_events, conditional_probabilities):
        """
        return: tuple of probability_of_condition & posterior_probabilities of events
        """
        probabilities_of_conditions_and_event = np.array(probabilities_of_events) * np.array(conditional_probabilities)
        probability_of_condition = probabilities_of_conditions_and_event.sum()
        posterior_probabilities = probabilities_of_conditions_and_event/probability_of_condition
        
        return probability_of_condition, posterior_probabilities



#####################################################################################################################################

class Probability_Distribution(object):
    """
    APPLICATIONS OF DISTRIBUTIONS
    Binomial Distribution: applied when number of trials is fixed before
                        the experiments begins, and each trial is indipendent
                        and can result in only two mutually exclusive outcome
                        (success/failure, either/or, yes/no)
    Possion Distribution: when each trial is independent. But although the
                        probabilities approach 0 after the first few values,
                        the number of possible values is finite. The result
                        are not limited to two mutually exclusive outcomes.
    Normal Distribution: where distribution is continues.
    """
    
    def __init__(self):
        self.X = None
        self.probabilities = None
        
        
    def get_random_variable(self):
        return self.X
    
    
    def set_random_variable(self, random_variable):
        self.X = random_variable
        
        
    def get_probabilities(self):
        return self.probabilities
    
    
    def set_probabilities(self, probabilities):
        self.probabilities = probabilities
        
        
    def possible_value_of_discrete_random_variable(self):
        """
        weighted average of the outcomes expected in future.
        As condition cahnge over time, the value would recompute
        and use this new value as the basis of future decision
        making.
        """
        return (self.X * self.probabilities).sum()
        
        
        """
        check the need of init variables
        """
        
    def conditional_table(self, obs, opp):
        """
        obs: obsolescence losses: caused by stocking too much than demand
        opp: opportunity losses: caused by shortage of stock when demand is high.
        
        optimal stock action: is the one that will minimize expected losses.
        """
        result_table = np.zeros((len(self.X),len(self.X)))
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                if i>j:
                    result_table[i,j] = (i-j)*opp
                elif i<j:
                    result_table[i,j] = (j-i)*obs
                    
        return result_table
        
        
    def expected_value(self, obs, opp):
        conditions = np.transpose(self.conditional_table(obs, opp))
        expectations = []
        for condition in conditions:
            expectations.append((condition * self.probabilities).sum())
        
        return expectations
    
    
    def probability_distribution_graph(self):
        plt.plot(self.X, self.probabilities)
        
    
    def binomial_distribution(self, p, n, r):
        """
        use for discrete probailities
        p: characteristic probability or probability of success
        n: number of trials undertaken
        r: number of success desired
        """
        return (math.factorial(n)/(math.factorial(r)*(math.factorial(n-r))))*(p**r)*((1-p)**(n-r))
    
    
    def binomial_distribution_evaluation(self, p, n):
        binomial_probability_distribution = []
        range_of_success_desired = np.arange(0, n+1)
        for i in range_of_success_desired:
            binomial_probability_distribution.append(self.binomial_distribution(p, n, i))
            
        plt.plot(range_of_success_desired, binomial_probability_distribution)
        
        result = {
            "r": range_of_success_desired,
            "Binomial Probability": binomial_probability_distribution,
        }
        
        return pd.DataFrame(result)
        
    def binomial_distribution_mean(self, n, p):
        return n*p
    
    
    def binomial_distribution_standard_deviation(self, n, p):
        return math.sqrt(n*p*(1-p))
    
        
    def poisson_distribution(self, mean, x):
        """
        use for discrete probailities
        mean: the mean number of occurrences per interval of time
        x: probability of exactly x occurrences
        """
        return ((mean**x)*((math.e)**(-mean)))/(math.factorial(x))
    
    
    def poisson_distribution_evaluation(self, mean):
        """
        we will calulate this on random variable
        """
        poisson_probability_distribution = []
        for i in self.X:
            poisson_probability_distribution.append(self.poisson_distribution(mean,i))
        
        plt.plot(self.X, poisson_probability_distribution)
        
        result = {
            "x": self.X,
            "P(x)": poisson_probability_distribution,
        }
        
        return pd.DataFrame(result)
    
    
    def standard_normal_table(self, file, column_drop):
        df_file = pd.read_csv(file)
        df_file.index = df_file[column_drop]
        df_file.drop(column_drop, axis=1, inplace=True)

        return df_file
    
    
    def number_of_standard_deviations(self, x, mu, sigma):
        """
        To get number of standard deviations from x to the mean 
        of this ditribution
        x = value of random variable with which concerned
        mu = mean if the distribution of this random variable
        sigma = standard deviation of thistribution
        """
        return round((x - mu)/sigma, 2)
    
    
    def probability_of_num_standard_deviations(self, file, z):
        """
        file = csv file to see standard normal distribution
        z = number of standard deviations from x to the mean 
        of this ditribution
        """
        z_str = str(round(z, 2))
        print('z string:', z)
        if len(z_str)>3:
            first_num, last_num = float(z_str[:-1]), int(z_str[-1])
        else:
            first_num, last_num = float(z_str), 0
        print("first, last:", first_num, last_num)
        
        if first_num <= 4.0 or first_num >=100:
            normal_table = self.standard_normal_table(file, 'z')
            probability_value = normal_table.loc[first_num][last_num]
        else:
            probability_value = 0.0
    
        return probability_value
    
    
    def probability_of_normal_distribution(self, z1, z2):
        if abs(z1) == abs(z2) == math.inf:
            return None
        
        elif abs(z1) == math.inf and abs(z2) != math.inf:
            if z2 > 0:
                print("a1")
                return self.probability_of_num_standard_deviations('cumulative.csv', z2)
            else:
                print("a2")
                return self.probability_of_num_standard_deviations('complementary_cumulative.csv', abs(z2))
        
        elif abs(z1) != math.inf and abs(z2) == math.inf:
            if z1 > 0:
                print("a3")
                return self.probability_of_num_standard_deviations('complementary_cumulative.csv', z1)
            else:
                print("a4")
                return self.probability_of_num_standard_deviations('cumulative.csv', abs(z1))
        
        else:
            if abs(z1) == 0 and z2 > 0:
                print("a5")
                return self.probability_of_num_standard_deviations('cumulative_from_mean_0toZ.csv', z2)
            
            elif z1>0 and z2>0:
                print("a6")
                prob1 = self.probability_of_num_standard_deviations('cumulative.csv', z2)
                prob2 = self.probability_of_num_standard_deviations('cumulative.csv', z1)
                print("probabilities 1 & 2", prob1, prob2)
                return prob1 - prob2
            
            elif (z1<0 and z2>0) or (z1>0 and z2<0):
                print("a7")
                #prob1 = self.probability_of_num_standard_deviations('cumulative.csv', abs(z2))
                #prob2 = self.probability_of_num_standard_deviations('cumulative.csv', abs(z1))
                
                prob1 = self.probability_of_num_standard_deviations('cumulative_from_mean_0toZ.csv', abs(z2))
                prob2 = self.probability_of_num_standard_deviations('cumulative_from_mean_0toZ.csv', abs(z1))
                print("probabilities 1 & 2", prob1, prob2)
                return prob1 + prob2
            
            elif z1<0 and z2<0:
                print("a8")
                prob1 = self.probability_of_num_standard_deviations('complementary_cumulative.csv', abs(z2))
                prob2 = self.probability_of_num_standard_deviations('complementary_cumulative.csv', abs(z1))
                print("probabilities 1 & 2", prob1, prob2)
                return prob1 - prob2


    def normal_distribution(self, mu, sigma, X1=-math.inf, X2=math.inf):
        """
        Also called Gausssian Distribution
        use for continues probabilities
        
        NOTE: 1. while taking random variable(rv) if rv's are inclusive
        then x1-0.5, x2+0.5, and if not inclusive then x1+0.5 & x2-0.5 
        only in the case of discontinoues distribution. In normal
        values remains the same.
        
        X1 = Smaller value of random variable with which concerned
        X2 = Larger value of random variable with which concerned
        mu = mean if the distribution of this random variable
        sigma = standard deviation of thistribution
        """
        z1 = self.number_of_standard_deviations(X1, mu, sigma)
        z2 = self.number_of_standard_deviations(X2, mu, sigma)
        print(z1, z2)
        return self.probability_of_normal_distribution(z1, z2)
        
    
    def __str__(self):
        return "Random Variables(X): "+str(self.X)+"\n \nProbabilities: "+str(self.probabilities)

