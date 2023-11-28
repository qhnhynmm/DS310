from load_data import load_data

def main():
    # Specify the path to your data file
    file_path = 'ner_dataset.csv'

    # Load and split the data
    train_dataset, dev_dataset, test_dataset = load_data(file_path)

    # Print some information about the datasets
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Optionally, you can iterate through the datasets to inspect the data
    # for inputs, labels in train_dataset:
    #     print(inputs, labels)

if __name__ == "__main__":
    main()
