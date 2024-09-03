# Open the file and read the lines
with open('text', 'r') as f:
    models = f.readlines()

# Remove newline characters, sort, and eliminate duplicates
models = sorted(set(line.strip() for line in models))

print(models)
print(len(models))
