---
interact_link: content/notebooks/avocado/avocado.ipynb
kernel_name: ir
title: 'avocado.ipynb'
prev_page:
  url: /notebooks/insurance/insurance
  title: 'insurance.ipynb'
next_page:
  url: /notebooks/movies/movies
  title: 'movies.ipynb'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Avocado Classifier
This Jupyter Notebook contains the code that takes in a table with information about avocados (average price, total volumne, total bags) and classifies them as either `conventional` or `organic`. This classifier is a $k$-nearest neighbors classifier using the cartesian distance between the point in question and the points in the training set. The data set is from [Kaggle](https://www.kaggle.com/neuromusic/avocado-prices).



{:.input_area}
```R
library(tidyverse)

avocado <- read_csv('avocado.csv')
head(avocado)
```



<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>Date</th><th scope=col>AveragePrice</th><th scope=col>Total Volume</th><th scope=col>4046</th><th scope=col>4225</th><th scope=col>4770</th><th scope=col>Total Bags</th><th scope=col>Small Bags</th><th scope=col>Large Bags</th><th scope=col>XLarge Bags</th><th scope=col>type</th><th scope=col>year</th><th scope=col>region</th></tr></thead>
<tbody>
	<tr><td>2015-12-27  </td><td>1.33        </td><td> 64236.62   </td><td>1036.74     </td><td> 54454.85   </td><td> 48.16      </td><td>8696.87     </td><td>8603.62     </td><td> 93.25      </td><td>0           </td><td>conventional</td><td>2015        </td><td>Albany      </td></tr>
	<tr><td>2015-12-20  </td><td>1.35        </td><td> 54876.98   </td><td> 674.28     </td><td> 44638.81   </td><td> 58.33      </td><td>9505.56     </td><td>9408.07     </td><td> 97.49      </td><td>0           </td><td>conventional</td><td>2015        </td><td>Albany      </td></tr>
	<tr><td>2015-12-13  </td><td>0.93        </td><td>118220.22   </td><td> 794.70     </td><td>109149.67   </td><td>130.50      </td><td>8145.35     </td><td>8042.21     </td><td>103.14      </td><td>0           </td><td>conventional</td><td>2015        </td><td>Albany      </td></tr>
	<tr><td>2015-12-06  </td><td>1.08        </td><td> 78992.15   </td><td>1132.00     </td><td> 71976.41   </td><td> 72.58      </td><td>5811.16     </td><td>5677.40     </td><td>133.76      </td><td>0           </td><td>conventional</td><td>2015        </td><td>Albany      </td></tr>
	<tr><td>2015-11-29  </td><td>1.28        </td><td> 51039.60   </td><td> 941.48     </td><td> 43838.39   </td><td> 75.78      </td><td>6183.95     </td><td>5986.26     </td><td>197.69      </td><td>0           </td><td>conventional</td><td>2015        </td><td>Albany      </td></tr>
	<tr><td>2015-11-22  </td><td>1.26        </td><td> 55979.78   </td><td>1184.27     </td><td> 48067.99   </td><td> 43.61      </td><td>6683.91     </td><td>6556.47     </td><td>127.44      </td><td>0           </td><td>conventional</td><td>2015        </td><td>Albany      </td></tr>
</tbody>
</table>

</div>


## 2. Divide the Kaggle data set into the training and test sets
This cell selects the 4 columns we will use from the original table (three data point columns and the type column) and shuffles the rows of the csv file and separates them into a training set, to which the avocado to be classified will be compared, and a test set, to test the accuracy of the classifer once it is built. The test set will retain its `type` column so that we know what proportion of avocados the classifier gets correct. The training set has 18,000 rows and the test set has 249.



{:.input_area}
```R
av <- avocado %>%
    select(AveragePrice, 'Total Volume', 'Total Bags', type) %>%
    sample_frac(1)
av_train <- av[1:18000,]
av_test <- av[-(1:18000),]

# ensuring all rows are capture in av_test and av_train
dim(av)[1] == dim(av_train)[1] + dim(av_test)[1]
```



<div markdown="0" class="output output_html">
TRUE
</div>


## 3. Define a function to find the cartesian distances
In this section, I will define a function that finds the 3-dimensional cartesian distant between two points. This is an application of the Pythagorean Theorem. The distance between two points $(x_1, y_1, z_1)$ and $(x_2, y_2, z_2)$ is

$$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2 + (z_2-z_1)^2}$$

The function defined takes as arguments a table whose first 3 columns are data points and an array containing the corresponding values for the point that is being compared. It returns the table with a new column that has the distance between each row in the table and the point in the array.



{:.input_area}
```R
dist <- function (tbl, row) {
    vec <- row %>% slice(1) %>% c(recursive = TRUE) %>% unname()
    new_tbl <- as_tibble(tbl)
    distances <- sqrt((new_tbl[1] - vec[1])^2 + (new_tbl[2] - vec[2])^2 + (new_tbl[3] - vec[3])^2)
    new_tbl$distances = unlist(distances)
    new_tbl
}
```




{:.input_area}
```R
head(dist(av_train, av_test[1, 1:3]))
```



<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>AveragePrice</th><th scope=col>Total Volume</th><th scope=col>Total Bags</th><th scope=col>type</th><th scope=col>distances</th></tr></thead>
<tbody>
	<tr><td>1.82        </td><td>  31363.93  </td><td> 11088.64   </td><td>organic     </td><td>  60721.88  </td></tr>
	<tr><td>1.13        </td><td>   9466.36  </td><td>  4408.33   </td><td>organic     </td><td>  82821.97  </td></tr>
	<tr><td>1.73        </td><td>  10842.77  </td><td>  1953.67   </td><td>organic     </td><td>  81662.56  </td></tr>
	<tr><td>0.88        </td><td>1132112.24  </td><td>274879.88   </td><td>conventional</td><td>1073164.75  </td></tr>
	<tr><td>0.78        </td><td>4015563.02  </td><td>369784.70   </td><td>conventional</td><td>3939919.31  </td></tr>
	<tr><td>1.11        </td><td> 485736.10  </td><td> 97862.93   </td><td>conventional</td><td> 403282.36  </td></tr>
</tbody>
</table>

</div>


## 4. Define a function to find the majority classification
$k$-NN classifiers work by determining what classification a majority of the $k$ points closest to a point in question have. The function `find_majority` defined below runs the `dist` function on a table and returns that output sorted by increasing distance. The function `knn` below that selects the top $k$ rows and returns the majority classification.



{:.input_area}
```R
find_majority <- function (df, df2, row_index) {
    test <- df2[row_index, 1:3]
    d <- df %>%
        dist(test) %>%
        arrange(distances)
    d
}
```




{:.input_area}
```R
head(find_majority(av_train, av_test, 1))
```



<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>AveragePrice</th><th scope=col>Total Volume</th><th scope=col>Total Bags</th><th scope=col>type</th><th scope=col>distances</th></tr></thead>
<tbody>
	<tr><td>1.22        </td><td>91195.99    </td><td>11592.92    </td><td>conventional</td><td>1587.918    </td></tr>
	<tr><td>1.72        </td><td>92086.00    </td><td> 8451.03    </td><td>organic     </td><td>1823.039    </td></tr>
	<tr><td>1.98        </td><td>90125.42    </td><td>10797.14    </td><td>organic     </td><td>2023.701    </td></tr>
	<tr><td>1.55        </td><td>91728.18    </td><td>12674.57    </td><td>conventional</td><td>2426.205    </td></tr>
	<tr><td>1.65        </td><td>91949.26    </td><td>12857.87    </td><td>conventional</td><td>2587.133    </td></tr>
	<tr><td>1.99        </td><td>89435.60    </td><td> 9234.73    </td><td>organic     </td><td>2841.639    </td></tr>
</tbody>
</table>

</div>




{:.input_area}
```R
knn <- function (df, df2, row, k) {
    sort <- find_majority(df, df2, row)
    new_df <- sort[1:k,] %>%
        count(type) %>%
        arrange(desc(n))
    new_df[1, 1]
}
```




{:.input_area}
```R
knn(av_train, av_test, 1, 7)
```



<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>type</th></tr></thead>
<tbody>
	<tr><td>conventional</td></tr>
</tbody>
</table>

</div>


## 5. Test the accuracy of the 7-NN classifier
For an example, I will text how accurate the 7-nearest neighbors classifer is. The `test_accuracy` function defined below runs the classifier on all rows of the `av_test` table (the entire test set), and then returns the proportion of rows that were correctly classified.



{:.input_area}
```R
test_accuracy <- function (train, test, k) {
    indices <- 1:dim(test)[1]
    classed <- sapply(indices, (function(i){knn(train, test, i, k)}))
    classed_test <- as_tibble(test)
    classed_test$kNN.class <- classed
    sum(classed_test$kNN.class == classed_test$type) / dim(classed_test)[1] 
}
```




{:.input_area}
```R
test_accuracy(av_train, av_test, 7)
```



<div markdown="0" class="output output_html">
0.927710843373494
</div>


## 6. Determining the optimal value of $k$
In order to determine how many nearest neigbors would be best to run on a random avocado, this second determines the optimal value of $k$ based on the training set. It will run through the classifier for odd integer values 1 through 99, and return a table with the accuracy of each value.



{:.input_area}
```R
results <- tibble(.rows = 50)
results$k <- seq(1, 100, by = 2)
accuracy <- sapply(results$k, (function(k){test_accuracy(av_train, av_test, k)}))
results$accuracy = unlist(accuracy)
head(arrange(results, desc(accuracy)))
```



<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>k</th><th scope=col>accuracy</th></tr></thead>
<tbody>
	<tr><td>35       </td><td>0.9437751</td></tr>
	<tr><td>39       </td><td>0.9437751</td></tr>
	<tr><td>71       </td><td>0.9437751</td></tr>
	<tr><td>73       </td><td>0.9437751</td></tr>
	<tr><td>75       </td><td>0.9437751</td></tr>
	<tr><td>77       </td><td>0.9437751</td></tr>
</tbody>
</table>

</div>


Based on this test, it appears that the 35-NN classifier is the best classifier, given that it is the most efficient and is one of the $k$ values with the highest accuracy score.
