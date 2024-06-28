import pandas as pd
from tokenizerFile import Tokenizer
import matplotlib.pyplot as plt

def plot_article_proportions(df):
        label_counts = df['label'].value_counts()
        labels = ['Real', 'Fake']
        sizes = [label_counts[1], label_counts[0]]
        colors = ['#66b3ff', '#ff9999']

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Proportion of Real vs Fake Articles')
        plt.show()

def main():
    
    # import the CSV file with data and transform it into a pandas dataframe
    CSV_data = pd.read_csv('cleaned_data.csv')
    #print(data.head())

    plot_article_proportions(CSV_data)
    
    # tokenize data
    myTokenizer = Tokenizer()
    test_data = CSV_data.iloc[0:4]      # test data to tokanize with (first 5 rows)
    tokenized_data:pd.DataFrame = myTokenizer.tokenize_dataframe(CSV_data)
    tokenized_data.to_csv('tokenized_data.csv', index=False)


    # perform k-fold cross validation
    k = 10                              # NOTE number of folds; one of our hyperparameters



if __name__ == '__main__':
    main()