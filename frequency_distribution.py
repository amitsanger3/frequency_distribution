#!/usr/bin/env python
# coding: utf-8

# Imports some useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
# import seaborn as sns
# import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
# from sklearn.model_selection import train_test_split
import math


class relative_frequency(object):
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.width_of_class_interval = None
        self.data_range = None
        self.number_of_intervals = None

        
    def get_data_range(self, number_of_intervals):
        self.width_of_class_interval = self. get_width_of_class_interval(number_of_intervals)
        return np.arange(min(self.dataset), max(self.dataset)+self.width_of_class_interval, self.width_of_class_interval).round(decimals=1)
        
        
    def get_custom_data_range(self, start_point, end_point, width_of_class_interval):
        return np.arange(start_point, end_point+width_of_class_interval, width_of_class_interval)
        
        
    def get_width_of_class_interval(self, number_of_intervals):
        self.number_of_intervals = number_of_intervals
        width_of_class_interval = round(((math.ceil(max(self.dataset))-min(self.dataset))/(self.number_of_intervals)), ndigits=1)
        if width_of_class_interval < 0.2:
            self.get_width_of_class_interval(number_of_intervals-1)
        else:
            return width_of_class_interval
        
        
    def get_custom_width_of_class_interval(self, start_point, end_point, number_of_intervals):
        return round((end_point - start_point)/(number_of_intervals))
        
        
    def get_number_of_intervals(self, width_of_class_interval=0.0):
        if width_of_class_interval == 0.0:
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
    
    
    def interval(self, data_range):
        self.data_range = data_range
        result = []
        for i in range(len(self.data_range)-1):
            if i == 0:
                result.append(str(round(self.data_range[i], ndigits=1))+'-'+str(round(self.data_range[i+1], ndigits=1)))
            elif i == len(self.data_range)-2:
                result.append(str(round(self.data_range[i], ndigits=1))+'-'+str(round(max(self.dataset), ndigits=1)))
            else:
                result.append(str(round(self.data_range[i]+0.1, ndigits=1))+'-'+str(round(self.data_range[i+1], ndigits=1)))
        
        return result
    
    
    def open_interval(self, data_range):
        self.data_range = data_range
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
            result = {
                "Interval": self.interval(data_range),
                "Frequency": self.frequency(data_range),
                "%": np.array(self.frequency(data_range))/len(self.dataset),
            }
        else:
            result = {
            "Interval": self.open_interval(data_range),
            "Frequency": self.open_frequency(data_range),
            "%": np.array(self.open_frequency(data_range))/len(self.dataset),
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

