---
redirect_from:
  - "notebooks/movies/editing-dictionary"
interact_link: content/notebooks/movies/editing_dictionary.ipynb
kernel_name: python3
title: 'editing_dictionary.ipynb'
prev_page:
  url: /notebooks/movies/movies
  title: 'movies.ipynb'
next_page:
  url: 
  title: ''
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

## Editing `dictionary.csv`
In order to use this file as a dictionary in [`movies.ipynb`](./movies.ipynb), I had to do some editing. First, I loaded the file into a pandas DataFrame and took the unique values of the column of words:



{:.input_area}
```python
import pandas as pd
words = pd.read_csv('dictionary.csv')['A'].unique()
```


Then, to put all of the words into lowercase and remove any words with punctuation (mostly hyphens), I used the string library and the method `str.lower()` on each entry in the array `words` (created above). Then, I put the words into a dictionary, taking out all of the words replace with the empty string `''`.



{:.input_area}
```python
import string

dictionary = []
for word in words:
    word = word.lower()
    for c in word:
        if c not in string.ascii_letters:
            word = ''
    dictionary += [word]

dictionary = [word for word in dictionary if word != '']
```


Finally, I exported the list by putting it into a new DataFrame, renaming the column of words to `'dictionary'`, and using the `df.to_csv()` method to export it without the row indices.



{:.input_area}
```python
pd.DataFrame(dictionary).rename({0:'dictionary'}, axis=1).to_csv('dict.csv', index=False)
```

