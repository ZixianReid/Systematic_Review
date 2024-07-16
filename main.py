from visualization import rq2_ml, rq1_applications, rq5_evaluation_metrics, rq4_dataset, rq3_code_represnetations, rq6_performance_comparsion
import pandas as pd

def get_demographics_publisher(df):
    print(df.groupby(['Journal_type', 'Publisher']).size())

print("es")
if __name__ == '__main__':
    file_path = './source/Primay_Study_new.xlsx'

    df = pd.read_excel(file_path)
    df = df[df['is_source_code'] != 'No'].reset_index(drop=True)

    # rq1_applications.plot_year(df)
    rq6_performance_comparsion.plot_clccd_performance(df)


    # rq3_code_represnetations.plot_code_representations_over_years(df)

    # rq6_performance_comparsion.plot_clccd_performance(df)

    # rq6_performance_comparsion.plot_ccd_performance_bcb(df)
    #
    # rq6_performance_comparsion.plot_ccd_performance_ojclone(df)
    #
    # rq6_performance_comparsion.plot_ccd_performance_gcj(df)
    # rq5_evaluation_metrics.table_evaluation_metrics(df)
    # rq4_dataset.datasets_over_applications(df)

    # rq1_applications.count_application_number(df)
    #
    # rq3_code_represnetations.plot_code_representation_name(df)
    #
    # rq3_code_represnetations.plot_code_representation_type(df)
    #
    # rq3_code_represnetations.plot_code_representations_over_years(df)