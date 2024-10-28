## Voting Analysis Project

This project performs analysis on voting data using Python and various scientific libraries. The analysis can be enhanced by integrating election results data from the [fivethirtyeight/election-results](https://github.com/fivethirtyeight/election-results) repository.

### Setting up the Election Results Submodule

To include the election results data as a submodule in this project:

1. **Clone this repository** (if you haven't already):

    ```bash
    git clone git@github.com:bwbellmath/voting.git
    cd voting
    ```

2. **Add the election-results repository as a submodule**:

    ```bash
    git submodule add git@github.com:fivethirtyeight/election-results.git path/to/submodule
    ```

    Replace `path/to/submodule` with the desired directory name.

3. **Initialize and update the submodule**:

    ```bash
    git submodule update --init --recursive
    ```

4. **Pull updates to the submodule** as needed:

    ```bash
    git submodule update --remote
    ```

5. **Commit the changes**:

    After setting up the submodule, commit the changes:

    ```bash
    git add .gitmodules path/to/submodule
    git commit -m "Added election-results as a submodule"
    ```

### Setting Up the Conda Environment

To set up the environment for running the project, follow these steps:

1. **Create the environment** from the provided `voting.yml` file:

    ```bash
    conda env create -f voting.yml
    ```

2. **Activate the environment**:

    ```bash
    conda activate voting
    ```

3. **Install any additional dependencies** using `pip` or `conda` as needed.

4. **Launch JupyterLab** (if you want to use it for development and analysis):

    ```bash
    jupyter lab
    ```

This will set up the necessary environment with all the dependencies required to run the analysis on voting data.

### Dependencies

The environment includes:
- **Python 3.10**: The core language used for analysis.
- **NumPy**: For numerical operations.
- **Pandas**: For handling data frames and CSVs.
- **Matplotlib** & **Seaborn**: For visualization.
- **JupyterLab**: For interactive data analysis and notebooks.
- **Scikit-learn**: For any machine learning tasks.
- **PyTorch**: Included for deep learning or tensor computations.
- **fivethirtyeight**: For the election data.

