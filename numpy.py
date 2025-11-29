import numpy as np

numbers_list = [10, 20, 30, 40, 50]

numbers_array = np.array(numbers_list)

mean_value = np.mean(numbers_array)
max_value = np.max(numbers_array)
sum_value = np.sum(numbers_array)

print(f"Numbers: {numbers_array}")
print(f"Mean: {mean_value}")
print(f"Maximum Value: {max_value}")
print(f"Sum: {sum_value}")
