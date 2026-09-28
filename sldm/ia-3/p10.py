import math

def expected_visits(age, condition):
    log_lambda = 2.5 - 0.03 * age + 0.5 * condition
    lambda_val = math.exp(log_lambda)
    return lambda_val

age = 60
visits_with_condition = expected_visits(age, condition=1)
visits_without_condition = expected_visits(age, condition=0)

print("Program 10 Output:")
print(f"Expected visits with chronic condition: {visits_with_condition:.2f}")
print(f"Expected visits without chronic condition: {visits_without_condition:.2f}")
print(f"Increase due to condition: {(visits_with_condition / visits_without_condition):.2f}x")
