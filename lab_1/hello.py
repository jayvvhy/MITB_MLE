print('hello world!!')
def square(x):
    return x**2

# Prompt user for input
user_input = input("Enter a number: ")

# Convert input from string to a number (float or int)
num = float(user_input)

# Call the function
result = square(num)

# Print the result
print(f"The square of {num} is {result}")