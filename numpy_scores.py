#Task 1 — Generate and Inspect the Data
import numpy as np

# Fix the random seed for reproducibility
np.random.seed(42)

# Generate scores: 5 students × 4 subjects, values between 50 and 100
scores = np.random.randint(50, 101, size=(5, 4))
print("Scores:\n", scores)

print("3rd student, 2nd subject:", scores[2, 1])

print("Last 2 students:\n", scores[-2:, :])

print("First 3 students, subjects 2 & 3:\n", scores[:3, 1:3])



#Task 2 — Analyze with Broadcasting 
col_means = np.round(scores.mean(axis=0), 2)
print("Column-wise mean:", col_means)

curve = np.array([5, 3, 7, 2])
curved_scores = scores + curve
curved_scores = np.clip(curved_scores, 0, 100)  # ensure max 100
print("Curved scores:\n", curved_scores)

row_max = curved_scores.max(axis=1)
print("Row-wise max:", row_max)



#Task 3 — Normalize and Identify
row_min = curved_scores.min(axis=1, keepdims=True)
row_max = curved_scores.max(axis=1, keepdims=True)
normalized = (curved_scores - row_min) / (row_max - row_min)
print("Normalized scores:\n", normalized)

max_index = np.unravel_index(np.argmax(normalized), normalized.shape)
print("Highest normalized value at student index", max_index[0], "subject index", max_index[1])

above_90 = curved_scores[curved_scores > 90]
print("Scores above 90:", above_90)
