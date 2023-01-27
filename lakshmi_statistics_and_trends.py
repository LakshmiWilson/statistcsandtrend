# Loading the required libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


# Reading the data from a csv file and converting the data to a dataframe.
def get_data(file_name):
    ''' This function takes in name of the csv file that contains the data, loads data into a dataframe and returns 
        the dataframe and transpose of the dataframe. '''
    df = pd.read_csv(file_name)
    return df, df.T


# Plotting a line plot of green house gas emitted data over the years.
def plot_green_house_gas_emission_data(X, Y):
    ''' This function takes in X and Y values and plots a line plot. '''
    plt.plot(X, Y)
    plt.xlabel('Year')
    plt.ylabel('Green House gas emission in terms of kilo tonnes of CO2 equivalent')
    plt.title('Plot showing the amount of green house gases emitted \n all over the world over the years 1990 to 2019.')
    plt.show()


# Plotting a line plot of percentage of urban population over the years.
def plot_urban_population_data(X, Y):
    ''' This function takes in X and Y values and plots a line plot. '''
    plt.plot(X, Y)
    plt.xlabel('Year')
    plt.ylabel('Percentage of population living in urban areas')
    plt.title('Plot showing the percentage of world population living \n in urban areas over the years 1990 to 2019.')
    plt.show()


# Loading the data into a dataframe.
green_house_gas_emission_data, green_house_gas_emission_data_transpose = get_data('green_house_gas_emissions.csv')
urban_population_data, urban_population_data_transpose = get_data('urban_population.csv')

# Creating a dataframe which consists of data required for our analysis from the entire data.
data = pd.DataFrame({'Green House Gas Emissions' : green_house_gas_emission_data.iloc[259, 34:64],
                    'Urban Population' : urban_population_data.iloc[259, 34:64]})

# Printing the dataframe.
print(data)

# Plotting line plots.
plot_green_house_gas_emission_data(range(1990, 2020), data.iloc[:, 0])
plot_urban_population_data(range(1990, 2020), data.iloc[:, 1])

# Calculating the mean of the data.
print('The average amount of green house gas emitted per year =', np.mean(data.iloc[:, 0]))
print('The average percentage of world population living in urban areas =', np.mean(data.iloc[:, 1]))

# Calculating the standard deviation of the data.
print('The standard deviation of the amount of green house gas emission data =', np.std(data.iloc[:, 0]))
print('The standard deviation of the percentage of world population living in urban areas data =', np.std(data.iloc[:, 1]))

# Calculating the correlation co-efficient of the two variables.
correlation_coefficient, pval = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])
print('The correlation co-efficient between the amount of green house gases emitted and the percentage of world population\n',
        'living in urban areas =', correlation_coefficient)