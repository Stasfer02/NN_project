import pandas as pd
from tokenizerFile import Tokenizer

def main():
    
    # import the CSV file with data and transform it into a pandas dataframe
    CSV_data = pd.read_csv('cleaned_data.csv')
    #print(data.head())

    # tokenize data
    myTokenizer = Tokenizer()
    test_data = CSV_data.iloc[0:4]      # test data to tokanize with (first 5 rows)
    tokenized_data:pd.DataFrame = myTokenizer.tokenize_dataframe(test_data)
    tokenized_data.to_csv('tokenized_data.csv', index=False)

    # apply cross-validation to calculate train/test split

    


if __name__ == '__main__':
    main()