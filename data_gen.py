from datasets import load_dataset

# Load the full E2E dataset
dataset = load_dataset("e2e_nlg")

# Access specific splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Example: Access the first entry
print(train_data[0]) 
# Returns: {'meaning_representation': '...', 'human_reference': '...'}