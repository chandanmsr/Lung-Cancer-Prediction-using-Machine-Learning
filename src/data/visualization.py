# src/data/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class DataVisualizer:
    """Handles all data visualization tasks"""
    
    @staticmethod
    def plot_distributions(data: pd.DataFrame, target_col: str) -> None:
        """Plot distributions of all features with smaller text"""
        num_cols = data.select_dtypes(include=['int64', 'float64']).columns
        num_cols = [col for col in num_cols if col != target_col]
        
        plt.figure(figsize=(10, 7.5))
        

        plt.rcParams.update({
            'axes.titlesize': 5,    # Subplot titles
            'axes.labelsize': 5,    # Axis labels
            'xtick.labelsize': 5,   # X-axis ticks
            'ytick.labelsize': 5,   # Y-axis ticks
            'legend.fontsize': 5,   # Legend text
            'figure.titlesize': 3  # Main title
        })
        
        for i, col in enumerate(num_cols, 1):
            plt.subplot(5, 4, i)
            sns.histplot(data=data, x=col, hue=target_col, kde=True, 
                        palette={'No Cancer':'dodgerblue', 'Cancer':'crimson'},
                        bins=20, alpha=0.6)
            plt.title(f'{col[:12]}...' if len(col) > 12 else col) 
            plt.xlabel('')
        
        plt.tight_layout()
        plt.suptitle('Feature Distributions by Cancer Status', y=1.02)
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

    @staticmethod
    def plot_boxplots(data: pd.DataFrame, target_col: str) -> None:
        """Plot boxplots with smaller text"""
        num_cols = data.select_dtypes(include=['int64', 'float64']).columns
        num_cols = [col for col in num_cols if col != target_col]

        plt.figure(figsize=(10, 5))
        

        plt.rcParams.update({
            'axes.titlesize': 8,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 2
        })
        
        for i, col in enumerate(num_cols, 1):
            plt.subplot(4, 5, i)
            sns.boxplot(data=data, y=col, x=target_col, hue=target_col,
                       palette={'Cancer':'crimson', 'No Cancer':'dodgerblue'},
                       linewidth=0.8,  # Thinner box lines
                       fliersize=2,    # Smaller outliers
                       legend=False)
            plt.title(f'{col[:10]}...' if len(col) > 10 else col)  # Truncate names
            plt.xlabel('Status', fontsize=6)  # Even smaller xlabel
        
        plt.tight_layout()
        plt.suptitle('Feature Distributions by Cancer Status', y=1.02, fontsize=9)
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault) 

    @staticmethod
    def plot_correlation_heatmap(data: pd.DataFrame, target_col: str) -> None:
        """Plot compact correlation matrix with small text"""
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        corr = numeric_data.corr()
    

        plt.rcParams.update({
            'axes.titlesize': 10,    
            'axes.labelsize': 8,     
            'xtick.labelsize': 7,    
            'ytick.labelsize': 7,    
            'font.size': 7      
        })

        # 1. Main Correlation Heatmap
        plt.figure(figsize=(8, 7)) 
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',  
                   cmap='coolwarm', center=0, square=True,
                   cbar_kws={'shrink': 0.5},  # colorbar
                   annot_kws={'size': 6})     # annotation text
        plt.title('Feature Correlation Matrix', pad=10)
        plt.tight_layout()
        plt.show()

        # 2. Target Correlations (compact)
        plt.figure(figsize=(5, 3.5)) 
        target_corr = corr[target_col].sort_values(ascending=False)
        target_corr = target_corr[target_corr.index != target_col]
    
        colors = ['crimson' if x > 0 else 'dodgerblue' for x in target_corr.values]
        sns.barplot(x=target_corr.values, y=target_corr.index, palette=colors)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
        plt.title('Correlation with Target', fontsize=9)
        plt.xlabel('Correlation', fontsize=7)
        plt.ylabel('')
        plt.tight_layout()
        plt.show()

        # 3. Top Correlations 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3)) 
    
        # Top positive
        top_positive = target_corr[1:6]
        sns.barplot(x=top_positive.values, y=top_positive.index, 
                   palette='Reds_r', ax=ax1)
        ax1.set_title('Top Positive', fontsize=9)
        ax1.set_xlabel('Correlation', fontsize=7)
    
        # Top negative
        top_negative = target_corr[-5:]
        sns.barplot(x=top_negative.values, y=top_negative.index, 
                   palette='Blues_r', ax=ax2)
        ax2.set_title('Top Negative', fontsize=9)
        ax2.set_xlabel('Correlation', fontsize=7)
    
        plt.tight_layout()
        plt.show()
    

        plt.rcParams.update(plt.rcParamsDefault)

    @staticmethod
    def plot_class_distribution(y: pd.Series) -> None:
        """Plot the distribution of target classes"""
        plt.figure(figsize=(6, 4))
        counts = y.value_counts()
        plt.bar(['No Cancer', 'Cancer'], counts, 
                color=['dodgerblue', 'crimson'])
        plt.title('Class Distribution')
        plt.ylabel('Count')
        for i, count in enumerate(counts):
            plt.text(i, count + 5, str(count), ha='center')
        plt.show()

    @staticmethod
    def plot_pca_visualization(X: np.ndarray, y: pd.Series) -> None:
        """Visualize data in 2D PCA space"""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, 
                       palette={0: 'dodgerblue', 1: 'crimson'},
                       alpha=0.7, s=100)
        plt.title('PCA Visualization (2 Components)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend(title='Cancer Status', labels=['No Cancer', 'Cancer'])
        plt.grid(alpha=0.3)
        plt.show()
        
plot_distributions = DataVisualizer.plot_distributions
