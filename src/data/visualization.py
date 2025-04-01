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
        """Plot distributions of all features"""
        num_cols = data.select_dtypes(include=['int64', 'float64']).columns
        num_cols = [col for col in num_cols if col != target_col]
        
        plt.figure(figsize=(20, 15))
        for i, col in enumerate(num_cols, 1):
            plt.subplot(5, 4, i)
            sns.histplot(data=data, x=col, hue=target_col, kde=True, 
                        palette={'No Cancer':'dodgerblue', 'Cancer':'crimson'},
                        bins=20, alpha=0.6)
            plt.title(f'{col} Distribution')
            plt.xlabel('')
        plt.tight_layout()
        plt.suptitle('Feature Distributions by Cancer Status', y=1.02)
        plt.show()

    @staticmethod
    def plot_boxplots(data: pd.DataFrame, target_col: str) -> None:
        num_cols = data.select_dtypes(include=['int64', 'float64']).columns
        num_cols = [col for col in num_cols if col != target_col]
    
        plt.figure(figsize=(20, 10))
        for i, col in enumerate(num_cols, 1):
            plt.subplot(4, 5, i)
            sns.boxplot(data=data, y=col, x=target_col, hue=target_col,
                   palette={'Cancer':'crimson', 'No Cancer':'dodgerblue'},
                   legend=False)
            plt.title(f'{col} Distribution')
            plt.xlabel('Cancer Status')
        plt.tight_layout()
        plt.suptitle('Feature Distributions by Cancer Status', y=1.02)
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(data: pd.DataFrame, target_col: str) -> None:
        """Plot correlation matrix with target with square boxes in heatmap"""
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        
        # Calculate figure size based on number of features
        n_features = len(numeric_data.columns)
        heatmap_size = max(10, n_features * 0.8)  # Dynamic sizing
        
        # Create figure with adjusted layout
        fig = plt.figure(figsize=(heatmap_size + 8, heatmap_size + 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[3, 1])
        corr = numeric_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        target_corr = corr[target_col].sort_values(ascending=False)
        colors = ['crimson' if (x > 0.3) else 'dodgerblue' if (x < -0.3) else 'gray' 
                 for x in target_corr]
        
        # Main correlation heatmap (square boxes)
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, cbar_kws={'shrink': 0.8},
                   annot_kws={'size': 10}, ax=ax1, square=True)
        ax1.set_title('Feature Correlation Matrix', pad=20, fontsize=14)
        
        # Feature correlation with target
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(x=target_corr.values, y=target_corr.index, hue=target_corr.index,
                   palette=colors, legend=False, dodge=False, ax=ax2)
        ax2.axvline(0, color='black', linestyle='--')
        ax2.set_title('Feature Correlation with Target', pad=15, fontsize=12)
        ax2.set_xlabel('Correlation Coefficient', fontsize=10)
        plt.setp(ax2.get_yticklabels(), fontsize=9)
        
        # Top positive correlations
        ax3 = fig.add_subplot(gs[1, 0])
        top_positive = target_corr[1:6]
        sns.barplot(x=top_positive.values, y=top_positive.index, hue=top_positive.index,
                   palette='Reds_r', legend=False, dodge=False, ax=ax3)
        ax3.set_title('Top 5 Positive Correlations', pad=15, fontsize=12)
        
        # Top negative correlations
        ax4 = fig.add_subplot(gs[1, 1])
        top_negative = target_corr[-5:]
        sns.barplot(x=top_negative.values, y=top_negative.index, hue=top_negative.index,
                   palette='Blues_r', legend=False, dodge=False, ax=ax4)
        ax4.set_title('Top 5 Negative Correlations', pad=15, fontsize=12)
        
        plt.tight_layout()
        plt.show()

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
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, 
                       palette={0: 'dodgerblue', 1: 'crimson'},
                       alpha=0.7, s=100)
        plt.title('PCA Visualization (2 Components)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend(title='Cancer Status', labels=['No Cancer', 'Cancer'])
        plt.grid(alpha=0.3)
        plt.show()