# LoanDiscrimination
Program to calculate fairness and the cost of being fair in credit loan domain
Here’s a README template for your project that explains the directory structure, how to download and set up the repository, and instructions on how to run the Jupyter Notebook and modules with attention to address paths.

---

# Project Name: Loan Discrimination Analysis

This project investigates potential discrimination in loan interest rates using a simulated dataset. It includes data processing, visualization, and fairness cost analysis modules.

## Project Structure

The project is organized as follows:

```
project-name/
├── data/                   # Contains datasets
│   ├── SimulatedDataset/   # Simulated dataset with Gender discrimination in bank loans
│   
├── docs/                   # Documentation files (e.g., additional explanations, images)
│   └── images/             # Images for documentation
├── notebooks/              # Jupyter Notebooks with analysis and exploration
│   └── Fairness_Main.ipynb
├── src/                    # Source code modules
│   ├── VisualizationPKG.py    # Visualization functions
│   ├── DiscriminationMetrics.py          # Functions for calculating metrics
│   └── CostOfFairness.py             # Functions for calculating the cost of fairness
├
├── .gitignore              # Specifies files and folders to ignore in version control
└── README.md               # Main project documentation (this file)
```

## Getting Started

### Prerequisites

To run this project locally, you need the following:

- Python 3.x
- Jupyter Notebook
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`

### Installation

1. **Clone the Repository**  
   First, clone this repository to your local machine:

   ```bash
   git clone https://github.com/CristinaBicharra/LoanDiscrimination.git
   cd project-name
   ```

2. **Install Required Libraries**  
   Install the necessary Python libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: If no `requirements.txt` file is provided, manually install libraries with:
   >
   > ```bash
   > pip install pandas numpy matplotlib seaborn
   > ```

3. **Check File Paths**  
   The project paths are structured relative to the root of the project folder. Ensure that the paths to the `data`, `notebooks`, and `src` folders are correct when you run the code.

   - **Dataset**: `data/processed/your_dataset.csv`
   - **Jupyter Notebooks**: `notebooks/analysis_notebook.ipynb`
   - **Modules**: `src/`

## Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open the `analysis_notebook.ipynb` file in the `notebooks` folder.
3. Follow the instructions in the notebook to execute cells and explore the analysis.

### Running the Project Modules

Each module in the `src` folder serves a specific purpose:

- **`VisualizationPKG.py`**: Contains functions for creating plots.  
- **`DiscriminationMetrics.py`**: Contains functions to calculate discrimination metrics.  
- **`CostOfFairness.py`**: Contains functions to calculate the cost of fairness adjustments.

You can import and use these modules directly in your Python scripts or in the Jupyter Notebook. Make sure the paths are correctly specified, especially if you're working outside the Jupyter Notebook.

#### Example Usage

```python
# Import necessary modules
import pandas as pd
from src.VisualizationPKG import plot_graph
from src.DiscriminationMetrics import calculate_weighted_disparity_index
from src.CostOfFairness import calculate_fairness_cost

# Load dataset
file_path = "data/processed/your_dataset.csv"
df = pd.read_csv(file_path)

# Run visualizations
plot_graph(df, 'InterestRate', '', 'histogram', 'Gender', ['Male', 'Female'], '', 'Interest Rate', 'Density', 'Histogram of Interest Rate')
```

## Notes on Paths

- **Dataset Path**: If you're running the notebook or modules from another directory, make sure to adjust the `file_path` variable to point to the correct dataset location.
- **Module Imports**: Ensure that the `src` folder is in the Python path when importing modules. You can add it dynamically in your notebook:

  ```python
  import sys
  sys.path.append("src")
  ```

## License

This project is licensed under the MIT License.

---

This README provides a clear guide on how to navigate the project, set up the environment, and run the code. Let me know if there’s any other detail you’d like to include!
