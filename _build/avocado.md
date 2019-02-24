---
interact_link: content/avocado.ipynb
kernel_name: python3
title: 'avocado.ipynb'
prev_page:
  url: /insurance
  title: 'insurance.ipynb'
next_page:
  url: 
  title: ''
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Avocado Classifier
This Jupyter Notebook contains the code that takes in a table with information about avocados (average price, total volumne, total bags) and classifies them as either `conventional` or `organic`. This classifier is a $k$-nearest neighbors classifier using the cartesian distance between the point in question and the points in the training set. The data set is from Kaggle (https://www.kaggle.com/neuromusic/avocado-prices).

## 1. Import datascience, numpy, and the table
The cell below imports the `datascience` and `numpy` libraries of Python, as well as opens the csv file as a `datascience` Table object.



{:.input_area}
```python
from datascience import *
import numpy as np

avocado = Table.read_table('avocado.csv')
avocado
```





<div markdown="0" class="output output_html">
<table border="1" class="dataframe">
    <thead>
        <tr>
            <th>Unnamed: 0</th> <th>Date</th> <th>AveragePrice</th> <th>Total Volume</th> <th>4046</th> <th>4225</th> <th>4770</th> <th>Total Bags</th> <th>Small Bags</th> <th>Large Bags</th> <th>XLarge Bags</th> <th>type</th> <th>year</th> <th>region</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0         </td> <td>2015-12-27</td> <td>1.33        </td> <td>64236.6     </td> <td>1036.74</td> <td>54454.8</td> <td>48.16</td> <td>8696.87   </td> <td>8603.62   </td> <td>93.25     </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
        <tr>
            <td>1         </td> <td>2015-12-20</td> <td>1.35        </td> <td>54877       </td> <td>674.28 </td> <td>44638.8</td> <td>58.33</td> <td>9505.56   </td> <td>9408.07   </td> <td>97.49     </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
        <tr>
            <td>2         </td> <td>2015-12-13</td> <td>0.93        </td> <td>118220      </td> <td>794.7  </td> <td>109150 </td> <td>130.5</td> <td>8145.35   </td> <td>8042.21   </td> <td>103.14    </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
        <tr>
            <td>3         </td> <td>2015-12-06</td> <td>1.08        </td> <td>78992.1     </td> <td>1132   </td> <td>71976.4</td> <td>72.58</td> <td>5811.16   </td> <td>5677.4    </td> <td>133.76    </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
        <tr>
            <td>4         </td> <td>2015-11-29</td> <td>1.28        </td> <td>51039.6     </td> <td>941.48 </td> <td>43838.4</td> <td>75.78</td> <td>6183.95   </td> <td>5986.26   </td> <td>197.69    </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
        <tr>
            <td>5         </td> <td>2015-11-22</td> <td>1.26        </td> <td>55979.8     </td> <td>1184.27</td> <td>48068  </td> <td>43.61</td> <td>6683.91   </td> <td>6556.47   </td> <td>127.44    </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
        <tr>
            <td>6         </td> <td>2015-11-15</td> <td>0.99        </td> <td>83453.8     </td> <td>1368.92</td> <td>73672.7</td> <td>93.26</td> <td>8318.86   </td> <td>8196.81   </td> <td>122.05    </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
        <tr>
            <td>7         </td> <td>2015-11-08</td> <td>0.98        </td> <td>109428      </td> <td>703.75 </td> <td>101815 </td> <td>80   </td> <td>6829.22   </td> <td>6266.85   </td> <td>562.37    </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
        <tr>
            <td>8         </td> <td>2015-11-01</td> <td>1.02        </td> <td>99811.4     </td> <td>1022.15</td> <td>87315.6</td> <td>85.34</td> <td>11388.4   </td> <td>11104.5   </td> <td>283.83    </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
        <tr>
            <td>9         </td> <td>2015-10-25</td> <td>1.07        </td> <td>74338.8     </td> <td>842.4  </td> <td>64757.4</td> <td>113  </td> <td>8625.92   </td> <td>8061.47   </td> <td>564.45    </td> <td>0          </td> <td>conventional</td> <td>2015</td> <td>Albany</td>
        </tr>
    </tbody>
</table>
<p>... (18239 rows omitted)</p>
</div>



## 2. Divide the Kaggle data set into the training and test sets
This cell selects the 4 columns we will use from the original table (three data point columns and the type column) and shuffles the rows of the csv file and separates them into a training set, to which the avocado to be classified will be compared, and a test set, to test the accuracy of the classifer once it is built. The test set will retain its `type` column so that we know what proportion of avocados the classifier gets correct. The training set has 18,000 rows and the test set has 249.



{:.input_area}
```python
av = avocado.select('AveragePrice', 'Total Volume', 'Total Bags', 'type')
av = av.sample(with_replacement=False)
av_train = av.take(np.arange(18000))
av_test = av.take(np.arange(18000, 18249))
av
```





<div markdown="0" class="output output_html">
<table border="1" class="dataframe">
    <thead>
        <tr>
            <th>AveragePrice</th> <th>Total Volume</th> <th>Total Bags</th> <th>type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1.06        </td> <td>15669.8     </td> <td>11247.1    </td> <td>organic     </td>
        </tr>
        <tr>
            <td>0.92        </td> <td>844690      </td> <td>409948     </td> <td>conventional</td>
        </tr>
        <tr>
            <td>1.11        </td> <td>128079      </td> <td>80410.8    </td> <td>organic     </td>
        </tr>
        <tr>
            <td>1.11        </td> <td>140981      </td> <td>36725.7    </td> <td>conventional</td>
        </tr>
        <tr>
            <td>1.01        </td> <td>3.03333e+06 </td> <td>1.15638e+06</td> <td>conventional</td>
        </tr>
        <tr>
            <td>0.96        </td> <td>128078      </td> <td>20393.6    </td> <td>conventional</td>
        </tr>
        <tr>
            <td>1.55        </td> <td>28095       </td> <td>6.18       </td> <td>organic     </td>
        </tr>
        <tr>
            <td>1.01        </td> <td>171880      </td> <td>34792.9    </td> <td>conventional</td>
        </tr>
        <tr>
            <td>1.37        </td> <td>178200      </td> <td>48896.9    </td> <td>conventional</td>
        </tr>
        <tr>
            <td>1.23        </td> <td>413224      </td> <td>127142     </td> <td>conventional</td>
        </tr>
    </tbody>
</table>
<p>... (18239 rows omitted)</p>
</div>



## 3. Define a function to find the cartesian distances
In this section, I will define a function that finds the 3-dimensional cartesian distant between two points. This is an application of the Pythagorean Theorem. The distance between two points $(x_1, y_1, z_1)$ and $(x_2, y_2, z_2)$ is

$$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2 + (z_2-z_1)^2}$$

The function defined takes as arguments a table whose first 3 columns are data points and an array containing the corresponding values for the point that is being compared. It returns the table with a new column that has the distance between each row in the table and the point in the array.



{:.input_area}
```python
def dist(t, arr):
    '''Takes in a table where the 1st 3 columns are the numerical data
    and returns the cartesian distance from an array with coincident values'''
    dists = make_array()
    for i in np.arange(t.num_rows):
        dist = np.sqrt((t.column(0).item(i) - arr.item(0)) ** 2 + (t.column(1).item(i) - arr.item(1)) ** 2 + (t.column(2).item(i) - arr.item(2)) ** 2)
        dists = np.append(dists, dist)
    return t.with_column('distances', dists)
```




{:.input_area}
```python
dist(av_train, np.array(av_test.drop('type').row(0)))
```





<div markdown="0" class="output output_html">
<table border="1" class="dataframe">
    <thead>
        <tr>
            <th>AveragePrice</th> <th>Total Volume</th> <th>Total Bags</th> <th>type</th> <th>distances</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1.06        </td> <td>15669.8     </td> <td>11247.1    </td> <td>organic     </td> <td>21378.2    </td>
        </tr>
        <tr>
            <td>0.92        </td> <td>844690      </td> <td>409948     </td> <td>conventional</td> <td>898691     </td>
        </tr>
        <tr>
            <td>1.11        </td> <td>128079      </td> <td>80410.8    </td> <td>organic     </td> <td>111238     </td>
        </tr>
        <tr>
            <td>1.11        </td> <td>140981      </td> <td>36725.7    </td> <td>conventional</td> <td>106705     </td>
        </tr>
        <tr>
            <td>1.01        </td> <td>3.03333e+06 </td> <td>1.15638e+06</td> <td>conventional</td> <td>3.20627e+06</td>
        </tr>
        <tr>
            <td>0.96        </td> <td>128078      </td> <td>20393.6    </td> <td>conventional</td> <td>92197.2    </td>
        </tr>
        <tr>
            <td>1.55        </td> <td>28095       </td> <td>6.18       </td> <td>organic     </td> <td>19739.1    </td>
        </tr>
        <tr>
            <td>1.01        </td> <td>171880      </td> <td>34792.9    </td> <td>conventional</td> <td>136988     </td>
        </tr>
        <tr>
            <td>1.37        </td> <td>178200      </td> <td>48896.9    </td> <td>conventional</td> <td>145579     </td>
        </tr>
        <tr>
            <td>1.23        </td> <td>413224      </td> <td>127142     </td> <td>conventional</td> <td>392747     </td>
        </tr>
    </tbody>
</table>
<p>... (17990 rows omitted)</p>
</div>



## 4. Define a function to find the majority classification
$k$-NN classifiers work by determining what classification a majority of the $k$ points closest to a point in question have. The function `find_majority` defined below runs the `dist` function on a table and returns that output sorted by increasing distance. The function `knn` below that selects the top $k$ rows and returns the majority classification.



{:.input_area}
```python
def find_majority(t, t2, row_index):
    '''Takes in training table (t), test table (t2), and row index of test
    table value (row_index) and computes the cartesian distance then
    returns the training table sorted by incrasing distance'''
    test = np.array(t2.drop('type').row(row_index))
    d = dist(t, test)
    return d.sort('distances')

find_majority(av_train, av_test, 0)
```





<div markdown="0" class="output output_html">
<table border="1" class="dataframe">
    <thead>
        <tr>
            <th>AveragePrice</th> <th>Total Volume</th> <th>Total Bags</th> <th>type</th> <th>distances</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1.93        </td> <td>35248.5     </td> <td>18306.9   </td> <td>organic</td> <td>682.81   </td>
        </tr>
        <tr>
            <td>1.67        </td> <td>36469.4     </td> <td>17619.4   </td> <td>organic</td> <td>760.185  </td>
        </tr>
        <tr>
            <td>1.75        </td> <td>36529.1     </td> <td>17569.9   </td> <td>organic</td> <td>837.672  </td>
        </tr>
        <tr>
            <td>1.43        </td> <td>36082.4     </td> <td>17277.1   </td> <td>organic</td> <td>873.174  </td>
        </tr>
        <tr>
            <td>2.41        </td> <td>35543.1     </td> <td>18989.7   </td> <td>organic</td> <td>931.654  </td>
        </tr>
        <tr>
            <td>1.44        </td> <td>34888.7     </td> <td>18444.9   </td> <td>organic</td> <td>1066.79  </td>
        </tr>
        <tr>
            <td>1.51        </td> <td>36396.9     </td> <td>17144.6   </td> <td>organic</td> <td>1102.24  </td>
        </tr>
        <tr>
            <td>1.3         </td> <td>35447.5     </td> <td>16815.3   </td> <td>organic</td> <td>1395.94  </td>
        </tr>
        <tr>
            <td>0.98        </td> <td>34903.1     </td> <td>19193     </td> <td>organic</td> <td>1461.26  </td>
        </tr>
        <tr>
            <td>2.03        </td> <td>36228.4     </td> <td>16701.5   </td> <td>organic</td> <td>1466.61  </td>
        </tr>
    </tbody>
</table>
<p>... (17990 rows omitted)</p>
</div>





{:.input_area}
```python
def knn(t, t2, row, k):
    test = np.array(t2.drop('type').row(row))
    sort = find_majority(t, t2, row)
    tbl = sort.take(np.arange(k)).group('type').sort(1, descending=True)
    return tbl.column(0).item(0)
```




{:.input_area}
```python
knn(av_train, av_test, 0, 7)
```





{:.output .output_data_text}
```
'organic'
```



## 5. Test the accuracy of the 7-NN classifier
For an example, I will text how accurate the 7-nearest neighbors classifer is. The `test_accuracy` function defined below runs the classifier on all rows of the `av_test` table (the entire test set), and then returns the proportion of rows that were correctly classified.



{:.input_area}
```python
def test_accuracy(train, test, k):
    '''Returns proportion of correct classifications from avocado classifier'''
    classed = make_array()
    for i in np.arange(test.num_rows):
        cl = knn(train, test, i, k)
        classed = np.append(classed, cl)
    
    classed_test = test.with_column('k-NN Class', classed)
    return np.count_nonzero(classed_test.column('k-NN Class') == classed_test.column('type')) / classed_test.num_rows
```




{:.input_area}
```python
test_accuracy(av_train, av_test, 7)
```





{:.output .output_data_text}
```
0.9477911646586346
```



## 6. Determining the optimal value of $k$
In order to determine how many nearest neigbors would be best to run on a random avocado, this second determines the optimal value of $k$ based on the training set. It will run through the classifier for odd integer values 1 through 99, and return a table with the accuracy of each value.



{:.input_area}
```python
results = make_array()
for i in np.arange(1, 100, 2):
    result = test_accuracy(av_train, av_test, i)
    results = np.append(results, result)
    
optimal_k = Table().with_columns(
    'k', np.arange(1, 100, 2),
    'Accuracy', results
)
optimal_k.sort('Accuracy', descending=True)
```





<div markdown="0" class="output output_html">
<table border="1" class="dataframe">
    <thead>
        <tr>
            <th>k</th> <th>Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>49  </td> <td>0.947791</td>
        </tr>
        <tr>
            <td>13  </td> <td>0.947791</td>
        </tr>
        <tr>
            <td>7   </td> <td>0.947791</td>
        </tr>
        <tr>
            <td>93  </td> <td>0.943775</td>
        </tr>
        <tr>
            <td>89  </td> <td>0.943775</td>
        </tr>
        <tr>
            <td>87  </td> <td>0.943775</td>
        </tr>
        <tr>
            <td>69  </td> <td>0.943775</td>
        </tr>
        <tr>
            <td>67  </td> <td>0.943775</td>
        </tr>
        <tr>
            <td>65  </td> <td>0.943775</td>
        </tr>
        <tr>
            <td>63  </td> <td>0.943775</td>
        </tr>
    </tbody>
</table>
<p>... (40 rows omitted)</p>
</div>



Based on the table above, it seems that using 7, 13, or 49 for $k$ are all equally as accurate (with minor, neglible differences, presumably). 
