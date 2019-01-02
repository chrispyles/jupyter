def standard_units(arr):
    return (arr - np.mean(arr)) / np.std(arr)

def correlation(x, y):
    return np.mean(standard_units(x) * standard_units(y))

def slope(x, y):
    r = correlation(x, y)
    return r * np.std(y) / np.std(x)

def intercept(x, y):
    m = slope(x, y)
    return np.mean(y) - m * np.mean(x)

a = 11