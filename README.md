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


### Survey Data and Election Results in This Repository

This project uses several datasets, both internal to the project and from the **fivethirtyeight/election-results** submodule, to analyze voter behavior, party identification, and election results over time.

#### Internal Datasets:

1. **`data-1atsq.csv`**: 
   - **Description**: This dataset contains longitudinal survey data focused on **party identification**. Categories include strong Democrat, weak Democrat, independent, Republican, and other minor parties. It also tracks how these affiliations have shifted over time, providing a historical overview of voter self-identification.
   
2. **`data-qOKi4.csv`**: 
   - **Description**: This file provides more granular insights into voter preferences. It includes responses from independents, breaking down whether they tend to lean toward a major party or prefer a third-party option. This data is crucial for modeling voter behavior, especially among those who are not strongly aligned with either major party.

3. **`data-survey1.csv`**: 
   - **Description**: Contains raw survey data from a recent poll focused on voter preferences, including data on party affiliation, voter turnout likelihood, and preferences in hypothetical election matchups. This dataset is useful for understanding current voter sentiment.

4. **`data-survey2.csv`**: 
   - **Description**: A follow-up survey to `data-survey1.csv`, capturing additional variables such as voter issue priorities and how those issues impact their party affiliation. It provides deeper context for voters' shifting allegiances.

#### Datasets from the fivethirtyeight/election-results Submodule:

1. **`presidential_results.csv`**: 
   - **Description**: This file contains detailed data on **presidential election results** over time, including state-by-state breakdowns of vote totals, percentages, and winning candidates. This dataset allows for analysis of trends in presidential elections and how they correlate with changes in party identification and voter behavior.

2. **`senate_results.csv`**: 
   - **Description**: A dataset focused on **U.S. Senate election results**. It includes vote counts, percentages, and party affiliations of candidates across multiple election cycles. This is useful for comparing local/state trends with national voter behaviors.

3. **`house_results.csv`**: 
   - **Description**: Provides election results for **U.S. House races**, with district-level data on vote totals, percentages, and party affiliations. This dataset supports analysis of voting patterns in smaller, more localized races compared to national elections.

4. **`governor_results.csv`**: 
   - **Description**: This file contains election results for **gubernatorial races**. It includes state-level data on candidates, vote totals, and party affiliations, offering insights into how state-level executive races reflect or differ from broader national trends.

5. **`special_election_results.csv`**: 
   - **Description**: Tracks the results of **special elections** across various offices. This dataset can be particularly useful for analyzing how non-standard election cycles (such as mid-term vacancies) affect voter turnout and party support.

6. **`primary_results.csv`**: 
   - **Description**: Focuses on **primary election results**, breaking down how party candidates perform during the primary process. It includes vote totals and percentages for each primary candidate, which can help assess party dynamics and internal competition.

#### Usage:
These datasets are utilized to model voter behavior, simulate elections, and explore how shifts in party identification over time may influence election outcomes. By analyzing both survey data and actual election results, this project aims to provide a comprehensive understanding of voter dynamics.

### Instructions for Using the Datasets:
1. **Download the necessary submodules**:
   ```bash
   git submodule update --init --recursive
