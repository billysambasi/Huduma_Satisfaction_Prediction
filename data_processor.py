import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads the CSV file into a pandas DataFrame."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def overview(self):
        """Displays basic information about the dataset."""
        if self.df is not None:
            print("\n Head:")
            print(self.df.head())
            
            print("\n Info:")
            print(self.df.info())

            print(f"\n Shape: {self.df.shape}")

            print("\n Describe (Numeric):")
            print(self.df.describe())

            print("\n Describe (Categorical):")
            print(self.df.describe(include='object'))

            print("\n Missing Values:")
            print(self.df.isna().sum())
        else:
            print("Data not loaded. Run `load_data()` first.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def fill_agency_name(self):
        """Fills missing 'Agency Name' using mapping from 'Agency Acronym'."""
        agency_map = self.df.dropna(subset=['Agency Name']).drop_duplicates('Agency Acronym')[
            ['Agency Acronym', 'Agency Name']
        ]
        agency_dict = dict(zip(agency_map['Agency Acronym'], agency_map['Agency Name']))
        self.df['Agency Name'] = self.df['Agency Acronym'].map(agency_dict).fillna('Unknown Agency')
        print("Filled missing 'Agency Name'")

    def fill_general_placeholders(self):
        """Fills general columns with placeholders."""
        self.df['Descriptor'] = self.df['Descriptor'].fillna('Not Provided')
        self.df['Borough'] = self.df['Borough'].fillna('Unknown Borough')
        self.df['Resolution Description'] = self.df['Resolution Description'].fillna('No Description Provided')
        print("Filled general placeholders for Descriptor, Borough, and Resolution Description")

    def fill_dissatisfaction_reason(self):
        """Conditionally fills 'Dissatisfaction Reason'."""
        self.df['Dissatisfaction Reason'] = self.df.apply(
            lambda row: 'Not Applicable'
            if row['Satisfaction Response'] != 'Strongly Disagree' and pd.isnull(row['Dissatisfaction Reason'])
            else row['Dissatisfaction Reason'], axis=1
        )
        print("Conditionally filled 'Dissatisfaction Reason'")

    def investigate_missing_reasons(self, plot: bool = True):
        """Investigates rows with missing dissatisfaction reasons despite disagreement."""
        missing_reason_df = self.df[
            (self.df['Satisfaction Response'] == 'Strongly Disagree') &
            (self.df['Dissatisfaction Reason'].isnull())
        ]

        print("\n🔍 Missing 'Dissatisfaction Reason' (Strongly Disagree)")
        print(missing_reason_df['Agency Name'].value_counts())
        print(missing_reason_df['Complaint Type'].value_counts())

        if plot and not missing_reason_df.empty:
            plt.figure(figsize=(12, 5))
            sns.countplot(data=missing_reason_df, y='Complaint Type',
                          order=missing_reason_df['Complaint Type'].value_counts().index[:10])
            plt.title('Top Complaint Types with Missing Dissatisfaction Reason (Strongly Disagree)')
            plt.show()

    def final_imputation_and_export(self, output_path: str = "cleaned.csv"):
        """Final fill for dissatisfaction reason and export the cleaned dataset."""
        self.df['Dissatisfaction Reason'] = self.df['Dissatisfaction Reason'].fillna('Reason Not Provided')
        self.df.to_csv(output_path, index=False)
        print(f"Final cleaned data saved to: {output_path}")

    def get_cleaned_data(self):
        """Returns the cleaned dataframe."""
        return self.df
