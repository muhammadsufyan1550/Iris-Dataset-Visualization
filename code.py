# Task : Exploring and Visualizing the Iris Dataset

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main function to execute the Iris dataset analysis
    """
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")

    print("="*60)
    print("TASK 1: EXPLORING AND VISUALIZING THE IRIS DATASET")
    print("="*60)

    # 1. Load the dataset
    print("\n1. LOADING THE DATASET")
    print("-" * 30)
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("✅ Dataset loaded successfully!")

    # 2. Dataset inspection
    inspect_dataset(df)
    
    # 3. Create visualizations
    create_visualizations(df)
    
    # 4. Generate insights
    generate_insights(df)
    
    print("\n" + "="*60)
    print("✅ TASK 1 COMPLETED SUCCESSFULLY!")
    print("="*60)

def inspect_dataset(df):
    """
    Perform basic dataset inspection
    """
    print("\n2. DATASET INSPECTION")
    print("-" * 30)
    
    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset Info:")
    print(df.info())
    print(f"\nSummary Statistics:")
    print(df.describe())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nSpecies distribution:")
    print(df['species_name'].value_counts())

def create_visualizations(df):
    """
    Create all required visualizations
    """
    print("\n3. DATA VISUALIZATION")
    print("-" * 30)
    
    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    # 3.1 Scatter plots
    create_scatter_plots(df, feature_columns)
    
    # 3.2 Histograms
    create_histograms(df, feature_columns)
    
    # 3.3 Box plots
    create_box_plots(df, feature_columns)
    
    # 3.4 Correlation heatmap
    create_correlation_heatmap(df, feature_columns)
    
    # 3.5 Pair plot
    create_pair_plot(df)

def create_scatter_plots(df, feature_columns):
    """
    Create scatter plots showing relationships between features
    """
    print("\n3.1 Creating scatter plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Scatter Plots: Relationships Between Features', fontsize=16, fontweight='bold')

    # Define scatter plot pairs
    plot_pairs = [
        ('sepal length (cm)', 'sepal width (cm)'),
        ('sepal length (cm)', 'petal length (cm)'),
        ('sepal length (cm)', 'petal width (cm)'),
        ('petal length (cm)', 'petal width (cm)'),
        ('sepal width (cm)', 'petal length (cm)'),
        ('sepal width (cm)', 'petal width (cm)')
    ]
    
    for i, (x_col, y_col) in enumerate(plot_pairs):
        row = i // 3
        col = i % 3
        sns.scatterplot(data=df, x=x_col, y=y_col, hue='species_name', 
                       s=100, alpha=0.8, ax=axes[row, col])
        axes[row, col].set_title(f'{x_col.title()} vs {y_col.title()}')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_histograms(df, feature_columns):
    """
    Create histograms showing value distributions
    """
    print("\n3.2 Creating histograms...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Histograms: Feature Value Distributions', fontsize=16, fontweight='bold')

    for i, col in enumerate(feature_columns):
        row = i // 2
        col_idx = i % 2
        
        sns.histplot(data=df, x=col, hue='species_name', alpha=0.7, 
                    bins=20, kde=True, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Distribution of {col}')
        axes[row, col_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_box_plots(df, feature_columns):
    """
    Create box plots for outlier detection
    """
    print("\n3.3 Creating box plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Box Plots: Outlier Detection by Species', fontsize=16, fontweight='bold')

    for i, col in enumerate(feature_columns):
        row = i // 2
        col_idx = i % 2
        
        sns.boxplot(data=df, x='species_name', y=col, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Box Plot: {col} by Species')
        axes[row, col_idx].grid(True, alpha=0.3)
        axes[row, col_idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def create_correlation_heatmap(df, feature_columns):
    """
    Create correlation heatmap
    """
    print("\n3.4 Creating correlation heatmap...")
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[feature_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_pair_plot(df):
    """
    Create comprehensive pair plot
    """
    print("\n3.5 Creating pair plot...")
    
    pair_plot = sns.pairplot(df, hue='species_name', diag_kind='hist', 
                            plot_kws={'alpha': 0.7}, diag_kws={'alpha': 0.7})
    pair_plot.fig.suptitle('Pair Plot: All Feature Relationships', 
                          fontsize=16, fontweight='bold', y=1.02)
    plt.show()

def generate_insights(df):
    """
    Generate and display key insights
    """
    print("\n4. DATA ANALYSIS AND INSIGHTS")
    print("-" * 40)
    
    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    print("📊 KEY FINDINGS:")
    print("1. Dataset contains 150 samples with 4 features")
    print("2. No missing values - clean dataset")
    print("3. Equal distribution: 50 samples per species")

    print("\n🔍 FEATURE INSIGHTS:")
    species_means = df.groupby('species_name')[feature_columns].mean()
    print("\nMean values by species:")
    print(species_means.round(2))

    print("\n📈 VISUALIZATION INSIGHTS:")
    print("1. SCATTER PLOTS: Petal length vs petal width shows clear species separation")
    print("2. HISTOGRAMS: Each feature shows different distribution patterns")
    print("3. BOX PLOTS: Few outliers detected, mainly in sepal width")
    print("4. CORRELATION: Strong correlation between petal length and petal width")

    # Outlier analysis
    print("\n⚠️  OUTLIER ANALYSIS:")
    for col in feature_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} potential outliers detected")
        else:
            print(f"{col}: No outliers detected")

if __name__ == "__main__":
    main()
