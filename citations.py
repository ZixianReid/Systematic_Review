import pandas as pd



def out_put_citations(df, type):
    df = df[df['Application_type'] == type].reset_index(drop=True)
    df.sort_values('Year', inplace=True)
    output_citations = []
    with open('output.txt', 'w', encoding='utf-8') as f:
        for item in df['cite']:
            item_name = item.split('{')[1].split(',')[0]
            output_citations.append(item_name)
            f.write("%s\n" % item)

    with open('output_citations.txt', 'w', encoding='utf-8') as f:
        f.write(", ".join(output_citations))

if __name__ == '__main__':
    file_path = './source/Primay_Study_new.xlsx'

    df = pd.read_excel(file_path)
    df = df[df['is_source_code'] != 'No'].reset_index(drop=True)
    out_put_citations(df, 'algorithm classification')