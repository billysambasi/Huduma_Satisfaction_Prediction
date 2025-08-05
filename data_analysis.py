import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ComplaintDataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        
    def drop_initial_columns(self):
        cols_to_drop = ['Unique Key', 'Agency Acronym']
        self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        return self

    def handle_missing_values(self, drop_threshold=0.5):
        missing_fraction = self.df.isnull().mean()
        cols_to_drop = missing_fraction[missing_fraction > drop_threshold].index
        self.df.drop(columns=cols_to_drop, inplace=True)
        return self

    def fill_missing_values(self, fill_dict=None):
        if fill_dict:
            self.df.fillna(fill_dict, inplace=True)
        return self

    def get_cleaned_data(self):
        return self.df


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


class ComplaintEDA:
    def __init__(self, df):
        self.df = df

    def value_counts(self, column):
        print(self.df[column].value_counts())

    def plot_bar(self, column, top_n=10, title='Bar Plot'):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, y=column, order=self.df[column].value_counts().iloc[:top_n].index)
        plt.title(title)
        plt.xlabel("Count")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()

    def complaints_by_month(self):
        monthly_counts = self.df['Survey Month'].value_counts().sort_index()
        plt.figure(figsize=(10, 5))
        sns.barplot(x=monthly_counts.index, y=monthly_counts.values)
        plt.title('Number of Complaints by Month')
        plt.xlabel('Survey Month')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def wordcloud_column(self, column):
        text = ' '.join(str(val) for val in self.df[column] if pd.notnull(val))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {column}')
        plt.show()

    def plot_heatmap(self):
        numeric_df = self.df.select_dtypes(include='number')
        if numeric_df.empty:
            print("No numerical columns to display in heatmap.")
            return
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def plot_pairplot(self, hue_column=None):
        numeric_df = self.df.select_dtypes(include='number')
        if numeric_df.empty:
            print("No numerical columns to plot in pairplot.")
            return
        if hue_column and hue_column in self.df.columns:
            sns.pairplot(self.df, vars=numeric_df.columns, hue=hue_column, corner=True)
        else:
            sns.pairplot(numeric_df, corner=True)
        plt.suptitle("Pairplot of Numerical Features", y=1.02)
        plt.show()

    def satisfaction_by_borough(self):
        plt.figure(figsize=(10, 5))
        sns.countplot(data=self.df, x='Borough', hue='Satisfaction Response')
        plt.title('Satisfaction Response by Borough')
        plt.xlabel('Borough')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

class ComplaintPreprocessor:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()
        self.label_encoders = {}
        self.vectorizer = None
        self.pca = None
        self.tfidf_matrix = None

    def encode_categoricals(self):
        # Automatically detect categorical columns (object type) excluding text fields
        excluded = ['Justified Dissatisfaction', 'Dissatisfaction Reason']
        cat_cols = [col for col in self.df.select_dtypes(include='object').columns if col not in excluded]
        
        for col in cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        return self

    def vectorize_text_column(self, column='Justified Dissatisfaction', max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df[column].astype(str))
        return self

    def reduce_dimensions(self, n_components=2):
        if self.tfidf_matrix is None:
            raise ValueError("Text data must be vectorized before dimensionality reduction.")
        
        self.pca = PCA(n_components=n_components)
        reduced = self.pca.fit_transform(self.tfidf_matrix.toarray())

        self.df['PC1'] = reduced[:, 0]
        self.df['PC2'] = reduced[:, 1]
        return self

    def get_processed_data(self):
        return self.df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SatisfactionVisualizer:
    def __init__(self, df):
        self.df = df
        sns.set_theme(style="whitegrid")
    
    def plot_overall_satisfaction(self):
        counts = self.df['Satisfaction Response'].value_counts()
        counts.plot(kind='bar', color=['green', 'red'], figsize=(6, 4))
        plt.title("Overall Satisfaction Distribution")
        plt.xlabel("Response")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show(); 

    def plot_satisfaction_by_complaint_type(self, top_n=10):
        top_types = self.df['Complaint Type'].value_counts().head(top_n).index
        data = self.df[self.df['Complaint Type'].isin(top_types)]
        ct = pd.crosstab(data['Complaint Type'], data['Satisfaction Response'])
        ct.plot(kind='bar', stacked=True, figsize=(10, 5))
        plt.title("Satisfaction by Complaint Type (Top {})".format(top_n))
        plt.xlabel("Complaint Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show(); 

    def plot_satisfaction_by_agency(self, top_n=10):
        top_agencies = self.df['Agency Name'].value_counts().head(top_n).index
        data = self.df[self.df['Agency Name'].isin(top_agencies)]
        ct = pd.crosstab(data['Agency Name'], data['Satisfaction Response'])
        ct.plot(kind='bar', stacked=True, figsize=(10, 5))
        plt.title("Satisfaction by Agency (Top {})".format(top_n))
        plt.xlabel("Agency")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show(); 

    def plot_satisfaction_trend(self):
        time_group = self.df.groupby(['Survey Year', 'Survey Month'])['Satisfaction Response'].value_counts().unstack().fillna(0)
        time_group.index = pd.to_datetime([f"{y}-{m}-01" for y, m in time_group.index])
        time_group = time_group.sort_index()
        time_group.plot(figsize=(10, 5))
        plt.title("Satisfaction Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Response Count")
        plt.tight_layout()
        plt.show(); 