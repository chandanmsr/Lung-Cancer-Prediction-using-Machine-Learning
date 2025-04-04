# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = 'viridis'
FIGURE_FACECOLOR = 'white'
AXES_GRID = True
GRID_ALPHA = 0.3
DEFAULT_FIGSIZE = (6, 3)

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

def configure_environment():
    """Set up global configuration"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    plt.style.use(PLOT_STYLE)
    sns.set_theme(style="whitegrid", palette=COLOR_PALETTE)
    plt.rcParams['figure.facecolor'] = FIGURE_FACECOLOR
    plt.rcParams['axes.grid'] = AXES_GRID
    plt.rcParams['grid.alpha'] = GRID_ALPHA
    plt.rcParams['figure.figsize'] = DEFAULT_FIGSIZE
    pd.set_option('display.max_columns', None)