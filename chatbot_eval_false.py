import csv

# Read the CSV file
with open('chatbot_eval.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    lines = list(csv_reader)

# Find and print lines where the last column is "False"
print("Lines with 'False' in the last column:")
print("=" * 50)

for i, line in enumerate(lines[1:], start=2):  # Skip header, start from line 2
    if len(line) >= 4 and line[3].strip() == "False":
        print(f"Line {i}: {line}")
        print("-" * 30)