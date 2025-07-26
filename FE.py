import logging
import pandas as pd

def feature_eng(file_path: str):
   #melting to long format
    long_df = file_path.melt(
        file_path,
        id_vars=['Unnamed: 0', 'company'], # columns to keep as identifiers
        value_vars=['2021', '2022', '2023', '2024'], # columns to melt into rows (they were columns) so I get one row per metric per company per year
        var_name='year', # name for newly transformed feature with years
        value_name='value' # new column 'value'
    )

    # to prevent errors, converting into numerical values
    long_df['value'] = file_path.to_numeric(long_df['value'], errors='coerce')

    #pivoting to create columns for each metric
    pivot_df = long_df.pivot_table(
        index=['company', 'year'], # each column -> 1 year, one company
        columns=['Unnamed: 0'], # each metric - separate column
        values='value', #use financial data
    ).reset_index()

    #making sure all values are numeric
    numeric_col = pivot_df.columns[2:] # to skip 'company' and 'year'
    for c in numeric_col:
        pivot_df[c] = file_path.to_numeric(pivot_df[c], errors='coerce' ) #'coerce' prevents from errors


    pivot_df['year'] = file_path.to_numeric(pivot_df['year'], errors='coerce')
    pivot_df = pivot_df.dropna(subset=['year'])
    pivot_df['year'] = pivot_df['year'].astype(int)

    #for classification goal, creating a target variable
    pivot_df = pivot_df.sort_values(['company', 'year']) #important chronological order as this is financial data
    pivot_df['FCF_next_year'] = pivot_df.groupby('company')['Free Cash Flow'].shift(-1) #next year cash flow column
    pivot_df['OCF_next_year'] = pivot_df.groupby('company')['Operating Cash Flow'].shift(-1) #next year operating cash flow column
    pivot_df['burn_cash'] = (
        (pivot_df['FCF_next_year'] < 0) | (pivot_df['OCF_next_year'] < 0) # | = or; astype(int) converts boolean to binary 1/0
    ).astype(int)

    pivot_df = pivot_df.dropna(subset=['burn_cash']) # removes rows where there is no target (important for errors)
    pivot_df.to_csv(file_path, index=False) # saving combined pivoted dataset 

    for c in pivot_df['company'].unique():
        company_pivot = pivot_df[pivot_df['company'] == c]
        company_pivot.to_csv(f'{c}_with_target.csv', index=False)

        logging.info('Profitability and CF - metrics ratios...')

        pivot_df['FCF_margin'] = pivot_df['Free Cash Flow'] / pivot_df['End Cash'] # FCF / End Cash = FCFmargin
        pivot_df['OCF_margin'] = pivot_df['Operating Cash Flow'] / pivot_df['End Cash'] # OCF / ENd cash = OCF margin

        logging.info('Growth and Change ratios...')
        pivot_df['Net_Income_YoY'] = pivot_df.groupby('company')['Net Income'].pct_change() * 100
        print(f'YoY trend analysis of Net Income: ', pivot_df['Net_Income_YoY'])

        pivot_df['FCF_YoY'] = pivot_df.groupby('company')['Free Cash Flow'].pct_change() * 100
        print(f'YoY trend analysis of FCF: ', pivot_df['FCF_YoY'])

        pivot_df['OCF_YoY'] = pivot_df.groupby('company')['Operating Cash Flow'].pct_change() * 100
        print(f'YoY trend analysis of OCF: ', pivot_df['OCF_YoY'])

        #now comparing Cap Spending of each company

        pivot_df['capex_ratio'] = pivot_df['Capital Expenditures'] / pivot_df['End Cash'] # Capex ratio = capex / end cash
        pivot_df['debt_repay_ratio'] = pivot_df['Debt Repay.'] / pivot_df['End Cash'] # debt repay ration = debt repay / end cash


    burn_rate = pivot_df.groupby('company')['burn_cash'].mean().reset_index()
    burn_rate['burn_cash_percent'] = burn_rate['burn_cash'] * 100

    avg_margins = pivot_df.groupby('company')[['FCF_margin', 'OCF_margin']].mean().reset_index() #
    avg_margins = avg_margins.melt(
        id_vars='company',
        var_name='Margin Type',
        value_name='Value'
    ) #need to convert from wide to long format

    trend = pivot_df.groupby('year')['Free Cash Flow'].mean().reset_index()
    capex_ratio = pivot_df.groupby('company')['capex_ratio'].mean().reset_index()