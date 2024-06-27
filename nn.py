import pandas as pd
import re
from collections import defaultdict
import string


from sklearn.model_selection import train_test_split

# Deze functie had ik gebruikt om te checken hoevaak "(Reuters)" oid voorkwam


def extract_parentheses_content_with_count(text_series):
    pattern = re.compile(r'\(([^)]+)\)')
    match_count = defaultdict(int)
    for text in text_series:
        if pd.notna(text):
            matches = pattern.findall(text)
            for match in matches:
                match_count[match] += 1

    return dict(match_count)


def remove_leading_spaces(df):
    df.iloc[:, 0] = df.iloc[:, 0].str.lstrip()
    return df

# check of er missing values zijn


def check_label_values(label_series):
    # Check for missing values in the label column
    return label_series.notna().all()

# Labels veranderen naar binary


def convert_labels(label_series):
    return label_series.map({'Real': 0, 'Fake': 1})

# Deze hoort bij de beschrijving van functie hieronder


def move_substring_to_title(df, substring):

    def process_row(row):
        text = row['Text']
        index = text.find(substring)
        if index != -1:
            row['title'] = text[:index].strip()
            row['Text'] = text[index + len(substring):].strip()
        return row

    df = df.apply(process_row, axis=1)
    return df

# Deze functie noemt de titel heel de string voor de gevonden substring, en de rest is de tekst
# En verwijdert de substring zelf


def move_sub_to_title(df):
    substring = "(Reuters)"
    df = move_substring_to_title(df, substring)
    substring = "(VIDEO)"
    df = move_substring_to_title(df, substring)
    substring = "(TWEETS)"
    df = move_substring_to_title(df, substring)
    substring = "(TWEET)"
    df = move_substring_to_title(df, substring)
    substring = "(AUDIO)"
    df = move_substring_to_title(df, substring)
    substring = "(DETAILS)"
    df = move_substring_to_title(df, substring)
    substring = "(IMAGES)"
    df = move_substring_to_title(df, substring)
    substring = "(IMAGE)"
    df = move_substring_to_title(df, substring)
    substring = "(SCREENSHOTS)"
    df = move_substring_to_title(df, substring)
    substring = "Reuters has not edited the statements or confirmed their accuracy."
    df = move_substring_to_title(df, substring)
    substring = "(VIDEO/TWEETS)"
    df = move_substring_to_title(df, substring)

    return df

# Dit split de tekst niet meer naar lower enz.


def to_normal_lower(text):
    # text = text.lower()
    # text = text.replace('”', ' " ')
    # text = text.replace('“', ' " ')
    # # hieronder verwijdert wel bezittelijke '
    # # text = text.replace('’', ' " ')

    # text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()
    return words

# tekst naar lowercase zonder punten etc


def to_normal_lower_2(text):
    text = ' '.join(text[:])
    text = text.lower()
    text = text.replace('–', ' " ')
    text = text.replace('”', ' " ')
    text = text.replace('“', ' " ')
    text = text.replace('‘', ' " ')
    text = text.replace('’', ' " ')

    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# tekst naar lowercase zonder punten etc


def to_normal_lower_3(text):
    text = text.lower()
    text = text.replace('–', ' " ')
    text = text.replace('”', ' " ')
    text = text.replace('“', ' " ')
    text = text.replace('‘', ' " ')
    text = text.replace('’', ' " ')
    # hieronder verwijdert wel bezittelijke '
    # text = text.replace('’', ' " ')

    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Dit is voor alle overige rows die nog geen title hebben. Van die rows had de title alles een hoofdletter
# dus ik heb 1 string voor t eerste woord beginnend met een kleine letter gestopt, en dat de titel gemaakt


def find_first_lower_text_index(row):
    if row['title'] == "Empty":
        text_list = row['Text']
        for i, text in enumerate(text_list):
            if text[0].isalpha() and text[0].islower() and i > 5:
                title = ' '.join(text_list[:i-1])
                title = title.lower()
                title = title.replace('”', ' " ')
                title = title.replace('“', ' " ')
                title = title.replace('‘', ' " ')
                title = title.replace('’', ' " ')
                title = title.translate(
                    str.maketrans('', '', string.punctuation))
                remaining_text = text_list[i-1:]
                return pd.Series({'title': title, 'Text': remaining_text})
    return pd.Series({'title': row['title'], 'Text': row['Text']})

# Dit haalt alle extra spaties weg ivm de apostrof weghalen


def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


# main
def main():
    # Read the CSV file
    df = pd.read_csv('fake_and_real_news.csv')
    # print(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    # print(df.duplicated().sum())
    # Remove leading spaces from the first column
    df = remove_leading_spaces(df)

    # print(df.head())

    # text_series = df.iloc[:, 0]
    # result = extract_parentheses_content_with_count(text_series)

    # sorted_result = sorted(
    #     result.items(), key=lambda item: item[1], reverse=True)

    # for content, count in sorted_result:
    #     if count > 5:
    #         print(f"'{content}': {count} times")

    # if check_label_values(df['label']):
    #     print("All rows have labels")
    # else:
    #     print("not all rows have labels")

    df['title'] = "Empty"
    df = df[['title', df.columns[0], 'label']]

    df['label'] = convert_labels(df['label'])
    print(df.columns)
    df = move_sub_to_title(df)

    # for text in empty_texts:
    #     print(text)

    df['Text'] = df['Text'].apply(to_normal_lower)
    # print(df['Text'][0][2])
    # df['title'] = df['title'].apply(to_normal_lower)

    df[['title', 'Text']] = df.apply(find_first_lower_text_index, axis=1)
    empty_texts = df[df['title'] == 'Empty']['Text']
    df['Text'] = df['Text'].apply(to_normal_lower_2)
    df['title'] = df['title'].apply(to_normal_lower_3)
    df['Text'] = df['Text'].apply(remove_extra_spaces)
    df['title'] = df['title'].apply(remove_extra_spaces)

    empty_texts.to_csv('text.txt', index=False)
    df.to_csv('cleaned_file.csv', index=False)


# Call the main function
if __name__ == "__main__":
    main()
