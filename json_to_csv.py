import json
import csv

IN = ".data/squad/dev-v1.1.json_preprocessed.json"

x = json.load(open(IN))

f = csv.writer(open(f"{IN}.csv", "w"))

# Write CSV Header, If you dont need that, remove this line
columns = ["id",
           "topic",
           "raw_paragraph_context",
           "paragraph_context",
           "paragraph_token_positions",
           "question",
           "a_start",
           "a_end",
           "a_extracted",
           "a_gt"]
f.writerow(columns)

for r in x:
    f.writerow([r[c] for c in columns])