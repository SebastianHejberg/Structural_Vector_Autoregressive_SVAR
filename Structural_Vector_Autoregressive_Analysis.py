from matplotlib.font_manager import FontProperties
from IPython.display import display, Math
from IPython.display import clear_output
import matplotlib.ticker as ticker
from sympy import Matrix, latex
import matplotlib.pyplot as plt
from tabulate import tabulate

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.tsatools import vech

import scipy.stats as stats
from scipy.optimize import least_squares
from scipy.optimize import fsolve
from scipy.linalg import sqrtm
from scipy.linalg import kron
from scipy.stats import chi2
from scipy.stats import f

from pathlib import Path
import seaborn as sns
import matlab.engine
import pandas as pd
import numpy as np
import inspect
import random
import os
import re

font_props = FontProperties(family='Times New Roman', size=10)
colors_navy = ["#a5c6e2", "#80afd6", "#5b97ca", "#3b7fb9", "#2f6694", "#234c6f", "#17334a", "#0b1925"]
color_navy = '#{:02x}{:02x}{:02x}'.format(85, 108, 131)
colors_pastel = sns.color_palette("pastel", 10)

class BaseClass:

    def __init__(self, LaTeX_path=str,Python_path=str,list_of_info_latex=list,list_of_info=list,path=str):
        """
        Generates LaTeX definitions and textual descriptions.

        ### Initial setup:
        - `list_of_info_latex` (str)  |  List of defintions from data (LaTeX symbols)
        - `list_of_info`       (str)  |  List of defintions from data (Text)
        - `path`               (str)  |  Path to the estiamted significans levels etc.
        - `LaTeX_path`         (str)  |  Path to LaTeX enviroment
        """
        self.Python_path = Python_path
        self.LaTeX_path = LaTeX_path
        self.list_of_info_latex = list_of_info_latex
        self.list_of_info = list_of_info
        self.path = path # f"{Path.cwd()}/Data & Estimates"
    
class Initial_Tools(BaseClass):

    def __init__(self, Base_initialize, date_column="Dates"):
        """
        Initializes and uses an existing instance of the BaseClass to initialize its attributes.

        ### Initial setup:
        - Base_initialize : (BaseClass) |  An instance of BaseClass containing data attributes such as LaTeX path, lists of definitions, and the path.
        - date_column     : (str)       |  The name of the date column, default is "Dates".
        """
        self.date_column = date_column
        self.LaTeX_path = Base_initialize.LaTeX_path
        self.Python_path = Base_initialize.Python_path
        self.list_of_info_latex = Base_initialize.list_of_info_latex
        self.list_of_info = Base_initialize.list_of_info
        self.path = Base_initialize.path
    
    def seasonal_dummies(self,nobs=int,freq=int):
        """
        Creates a matrix of seasonal dummy variables.

        Parameters:
            - nobs (int) : The number of observations (rows) in the output matrix.
            - freq (int) : The frequency of the seasonal data; 
                            set to 4 for quarterly data or 12 for monthly data.

        Returns:
            - numpy.ndarray: An (nobs x freq) matrix filled with 0's and 1's, 
                            representing seasonal dummies. The output will have
                            the following structure:
                            
                            For freq=4 (quarterly):
                            1 0 0 0
                            0 1 0 0
                            0 0 1 0
                            0 0 0 1
                            1 0 0 0
                            
                            For freq=12 (monthly):
                            1 0 0 0 0 0 0 0 0 0 0 0
                            0 1 0 0 0 0 0 0 0 0 0 0
                            ...
                            0 0 0 0 0 0 0 0 0 0 0 1
                            (Note: The pattern will repeat for each year.)
        """

        nobs_new = nobs + (freq - (nobs % freq))
        seas = np.zeros([nobs_new,freq])
        
        for i in range(1, nobs_new, freq):
            seas[i-1:i+freq-1,0:freq] = np.identity(freq)
        
        seas = seas[:nobs]
        return seas
    
    def setup_axis(self, ax, x_data, y_data, ylabel, font_props, colors_palette, linewidth_size=1.2, at_zero=False, mean_line=False,
                   special_data_format=[False,10],y_axis=[None,None]):
        """
        Plots `y_data` against `x_data` on the specified axis with customization options.

        Features:
        ---------
        - Plots data with customizable color and line width.
        - Optionally adds a mean line (`mean_line=True`).
        - Sets y-axis label (`ylabel`) and adjusts tick labels.
        - Hides top and right spines, optionally positions x-axis at zero (`at_zero=True`).
        - Adjusts y-axis limits with a 5% padding.
        - Applies font properties to tick labels.

        Example:
        --------
        ```
        plot = Initial_Tools(ax, x_data, y_data, "Y-axis label", font_props, 'blue', mean_line=True)
        plot.plot_with_axis()
        ```
        """

        ax.plot(x_data, y_data, color=colors_palette, linewidth=linewidth_size)

        if mean_line:
            ax.axhline(np.mean(y_data), color='red', linestyle=":", linewidth=1)

        ax.set_ylabel(ylabel, fontname='Times New Roman')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if at_zero:
            ax.spines['bottom'].set_position('zero')
        
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.margins(x=0,y=0)

        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_fontproperties(font_props)
        
        ax.set_xlim(min(x_data), max(x_data))

        if at_zero:
            ymin, ymax = ax.get_ylim()
            if ymin > 0:
                ax.set_ylim(0, ymax)
            elif ymax < 0:
                ax.set_ylim(ymin, 0)

        if special_data_format[0] == True:
            tick_indices = np.linspace(10, len(x_data) - 1, special_data_format[1]).astype(int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(x_data[tick_indices], rotation=45, ha='right')
            ax.set_xlim(0, len(x_data) - 1)
    
    def format_p_value(self, p=float):
        """
        Formats p-values with significance stars.
        If p <= 0.01 when it will return ***
        
        Parameters
        ----------
        - p: (float/int) | The numercial p-value

        Returns
        ----------
        - String *, **, ***, or ""
 
        
        Example:
        --------
        ```
        - If p <= 0.01 when it will return "***"
        - If p <= 0.05 when it will return "**"
        - If p <= 0.1 when it will return "*"
        - If p > 0.1 when it will return ""
        ```
        """
        if p <= 0.01:
            return "***"
        elif p <= 0.05:
            return "**"
        elif p <= 0.1:
            return "*"
        else:
            return ""
        
    def Descriptive_Statistics(self, result, display_data=True):
        # Clean the dataframe by dropping the first column and missing values
        #date_col = getattr(self, 'date_column', 'Dates')  # Fallback to 'Dates' if attribute is missing
        
        df_clean = result.drop(columns=[self.date_column]).dropna()

        # Generate descriptive statistics and add median
        desc_stats = df_clean.describe().transpose()
        desc_stats['median'] = df_clean.median()

        # Check if all variables have the same sample size
        if df_clean.count().nunique() == 1:
            print(f"All variables have the same sample size: {int(df_clean.count().mean())}")
        else:
            raise KeyError("The sample size varies across variables.")
        
        # Select and rename columns for final table
        desc_stats = desc_stats[['mean', 'median', 'min', 'max', 'std']]
        desc_stats.columns = ['Mean', 'Median', 'Min', 'Max', 'Std. Dev.']
        
        # Set index to a custom list of info
        desc_stats.index = self.list_of_info[1:]

        if display_data == True:
            display(desc_stats)

        # Convert to LaTeX format
        latex_table = desc_stats.to_latex(index=True,
                                          column_format=f'l{(len(desc_stats.columns)) * "c"}', 
                                          header=True, 
                                          float_format="%.2f", 
                                          multirow=True)

        if self.LaTeX_path != None:
            latex_name = f"{self.LaTeX_path}/Table/descriptive_statistics.tex"
            os.makedirs(os.path.dirname(latex_name), exist_ok=True)
            with open(latex_name, "w") as f:
                f.write(latex_table)
            print("Saved to LaTeX Path","\n")

        return latex_table

    def Data_plot(self,result,Save=True):
        font_props = FontProperties(family='Times New Roman', size=9)
        
        k = len(result.drop(columns=[self.date_column]).columns)

        num_pairs = k // 2  # Number of 1x2 subplots
        remaining_plot = k % 2  # 1 if odd, 0 if even
        
        # Iterate over pairs of subplots (2 plots per row)
        for i in range(num_pairs):
            fig, axs = plt.subplots(1, 2, figsize=(10, 2.5))  # 1 row, 2 columns
            self.setup_axis(axs[0], result['Dates'], result[result.columns[2*i+1]], 
                    f"{self.list_of_info[2*i+1]}  ({self.list_of_info_latex[2*i]})", 
                    font_props, colors_navy[4], mean_line=True)
            self.setup_axis(axs[1], result['Dates'], result[result.columns[2*i+2]], 
                    f"{self.list_of_info[2*i+2]}  ({self.list_of_info_latex[2*i+1]})", 
                    font_props, colors_navy[4], mean_line=True)
            
            plt.tight_layout()
            if Save == True:
                if self.Python_path != None:
                    os.makedirs(self.Python_path, exist_ok=True)
                    fig.savefig(f"{self.Python_path}/Plots/Q1_{2*i+1}.pdf", bbox_inches='tight', transparent=True)
                if self.LaTeX_path != None:
                    os.makedirs(self.LaTeX_path, exist_ok=True)
                    fig.savefig(f"{self.LaTeX_path}/Grafisk/Q1_{2*i+1}.pdf", bbox_inches='tight', transparent=True)
            plt.show()
        
        # If k is odd, add one more plot for the last element
        if remaining_plot == 1:
            fig, ax = plt.subplots(figsize=(5, 2.5))  # 1 row, 1 column
            self.setup_axis(ax, result['Dates'], result[result.columns[k]], 
                    f"{self.list_of_info[k]}  ({self.list_of_info_latex[k-1]})", 
                    font_props, colors_navy[4], mean_line=True)
            
            plt.tight_layout()
            if Save == True:
                if self.Python_path != None:
                    os.makedirs(self.Python_path, exist_ok=True)
                    fig.savefig(f"{self.Python_path}/Plots/Q1_{k}.pdf", bbox_inches='tight', transparent=True)
                if self.LaTeX_path != None:
                    os.makedirs(self.LaTeX_path, exist_ok=True)
                    fig.savefig(f"{self.LaTeX_path}/Grafisk/Q1_{k}.pdf", bbox_inches='tight', transparent=True)
            plt.show()
        
        if Save == True:
            if self.Python_path == None:
                print("No path specified for saving Python output. Please provide `Python_path` or set `Save=False`")
            if self.LaTeX_path == None:
                print("No path specified for saving LaTeX output. Please provide `LaTeX_path` or set `Save=False`")

    def clean_data(self, result,display_data=True):
        """
        Cleans the DataFrame by removing the specified column.

        Parameters
        ----------
        - result       : DataFrame    | The rare timeseries data.
        - date_column  : string       | Index of the column to drop (default is 0).
        - display_data : bool         | Whether to display the cleaned DataFrame (default is True).

        Returns
        ----------
        - Table : pd.DataFrame | Cleaned DataFrame without the specified column.
        """
        
        y_clean_data = result.drop(columns=[self.date_column])
        
        if display_data==True:
            display(y_clean_data)

        return y_clean_data

    def To_Matrix(self, matrix,short=True):
        latex_code = f"\\begin{{pmatrix}}\n"
        for row in matrix:
            latex_code += " & ".join(f"{val:.5f}" for val in row) + " \\\\\n"
        latex_code += r"\end{pmatrix}"
        
        if short==True:
            return latex_code.replace("\\\\\n\\end{pmatrix}",r"\end{pmatrix}").replace("\n"," ")
        else:
            return latex_code

class Lag_Order_Determination(BaseClass):

    def __init__(self, Base_initialize, y_dataframe):
        """
        Parameters:
            - y_dataframe      : (pd.DataFrame) | A (T, K) matrix / dataframe
            - Base_initialize  : (BaseClass)    |  An instance of BaseClass containing data attributes such as LaTeX path, lists of definitions, and the path.
        """
        self.y_dataframe = y_dataframe
        self.LaTeX_path = Base_initialize.LaTeX_path
        self.list_of_info_latex = Base_initialize.list_of_info_latex
        self.list_of_info = Base_initialize.list_of_info
        self.path = Base_initialize.path

    def Information_Criteria(self, maximum_lag_length=int, Trend=bool, Show=bool):
        """
        The Information_Criteria function calculates the optimal lag length for Vector Autoregressive (VAR) models 
        using three information criteria: AIC, Hannan-Quinn Criterion (HQ), and SIC.

        Parameters:
            - maximum_lag_length : (int)  | Maximum number of lags to evaluate (default: 3).
            - Trend              : (bool) | Include a trend term in the model (default: True).
            - display_data       : (bool) | Print results (default: True).

        Returns:
            - Table      : (pd.DataFrame) | A DataFrame with lag lengths and their corresponding SIC, HQ, and AIC values.
        """
        
        aiccrit = []
        hqccrit = []
        siccrit = []

        t, K = self.y_dataframe.shape
        XMAX = np.ones((1, t - maximum_lag_length))

        if Trend:
            XMAX = np.vstack([XMAX, np.arange(1, t - maximum_lag_length + 1).reshape(1, -1)]) 

        for i in range(1, maximum_lag_length + 1):
            XMAX = np.vstack([XMAX, self.y_dataframe.iloc[maximum_lag_length - i:t - i, :].T])   

        Y = self.y_dataframe.iloc[maximum_lag_length:t, :].T

        for i in range(0, maximum_lag_length + 1):  # Iterating over every second value
            X = XMAX[:1 * i * K + 1, :]  

            B = Y @ X.T @ np.linalg.inv(X @ X.T)

            SIGMA = (np.array(Y) - np.array(B @ X)) @ (np.array(Y) - np.array(B @ X)).T / (t - maximum_lag_length)

            ############### Information Criterion ###############

            ###### Model fit term 
            fit_term = np.log(np.linalg.det(SIGMA))
            
            ###### \varphi(m) term - (Deterministic regressors including an intercept)
            regressors_and_constant = (i * K**2 + K)

            ##### The three information criterion 
            aiccrit.append(fit_term + 2 / (t - maximum_lag_length) * regressors_and_constant)
            hqccrit.append(fit_term + 2 * np.log(np.log(t - maximum_lag_length)) / (t - maximum_lag_length) * regressors_and_constant)
            siccrit.append(fit_term + np.log(t - maximum_lag_length) / (t - maximum_lag_length) * regressors_and_constant)
        
        # Create a DataFrame to display results
        lags = np.arange(maximum_lag_length + 1)
        
        result_table = pd.DataFrame({'Lag Length': lags, 'SIC': siccrit, 'HQ': hqccrit, 'AIC': aiccrit})

        if Show==True:
            display(result_table.rename(columns={"Lag Length":"Lag"}).style.hide(axis="index"))

        return result_table
    
    def Top_Down_Sequence(self, maximum_lag_length=int, Trend = bool, Show=bool):
        """
        Top-Down Sequential test to determine the optimal lag length for VAR models.

        This method tests a sequence of null hypotheses regarding the coefficients at different lag lengths, 
        starting from the maximum lag length and proceeding downward. The testing continues until a null hypothesis 
        is rejected or all hypotheses are tested, concluding whether additional lags are needed or if the optimal 
        lag length is zero.

        Parameters:
            - maximum_lag_length : (int)            | Maximum number of lags to consider.
            - Trend              : (bool, optional) | Include a trend term in the model. Defaults to True.
            - display_data       : (bool, optional) | Display the results. Defaults to True.

        Returns:
            - Table              : (pd.DataFrame)   | A DataFrame containing the lag order, log likelihood, LR test statistics, and p-values.
        """

        t, K = self.y_dataframe.shape
        
        # Construct regressor matrix and dependent variable
        XMAX = np.ones((1, t - maximum_lag_length))
        
        if Trend == True:
            XMAX = np.vstack([XMAX, np.arange(1, t - maximum_lag_length + 1).reshape(1, -1)]) 

        for i in range(1, maximum_lag_length + 1):
            XMAX = np.vstack([XMAX, self.y_dataframe.iloc[maximum_lag_length - i:t - i, :].T])

        
        Y = self.y_dataframe.iloc[maximum_lag_length:t, :].T

        # Initialize Log Likelihood storage
        Log_Likelihood = []
        
        # Evaluate models for lag lengths from pmax down to 0
        for i in range(maximum_lag_length, -1, -1):
            
            T = t - maximum_lag_length
            
            X = XMAX[:i * K + 1, :]

            B = np.array(Y) @ np.array(X).T @ np.linalg.inv(X @ X.T)
            
            SIGMA = (np.array(Y) - np.array(B @ X)) @ (np.array(Y) - np.array(B @ X)).T / T

            Log_Likelihood.append(T * np.log(np.linalg.det(SIGMA)))

        Log_Likelihood.reverse()

        lr_stat = [""]  
        p_value = [""]  
        lags_value = [0] 

        # Calculate LR test statistics and p-values
        for i in range(0,maximum_lag_length):
            lr_stat.append(Log_Likelihood[i] - Log_Likelihood[i+1])
            lags_value.append(i+1)
            p_value.append(1 - chi2.cdf((Log_Likelihood[i] - Log_Likelihood[i+1]), K ** 2))
                
        # Convert results to a table
        table_lr = pd.DataFrame({"Lag": lags_value, "Log. Likelihood": Log_Likelihood, "LR test": lr_stat, "p-value": p_value})

        if Show == True:
            display(table_lr.style.hide(axis="index"))

        return table_lr

    def Combined_Lag_Selection(self, maximum_lag_length=int, Trend = bool, LaTeX=False):
        
        # Run Information Criteria and Top Down Testing
        result_table_IC = self.Information_Criteria(maximum_lag_length=maximum_lag_length, Trend=Trend, Show=False)
        result_table_TD = self.Top_Down_Sequence(maximum_lag_length=maximum_lag_length, Trend=Trend, Show=False)

        formatted_LR_test = []
        for i in range(len(result_table_TD)):
            if result_table_TD["LR test"][i] != "":
                formatted_LR_test.append(np.round(result_table_TD["LR test"][i], 3).astype(str) + Initial_Tools.format_p_value(self,result_table_TD["p-value"][i]))
            else:
                formatted_LR_test.append("")
        formatted_LR_test = pd.DataFrame(formatted_LR_test, columns=['LR Test'])
        merged_df = pd.concat([result_table_IC, formatted_LR_test], axis=1)
        merged_df = merged_df.rename(columns={"Lag Length": "Lag"})
        
        TDT_formatted = []

        for i in range(len(result_table_TD)):
            if result_table_TD["LR test"][i] != "":
                TDT_formatted.append(fr'$\underset[[({round(result_table_TD["LR test"][i], 1)})]][[{round(result_table_TD["Log. Likelihood"][i], 1)}^[[{Initial_Tools.format_p_value(self,result_table_TD["p-value"][i])}]]]]$')
            else:
                TDT_formatted.append("")

        result_table_IC["Top down"] = TDT_formatted

        filtered_df = result_table_TD[result_table_TD['p-value'] != ""]
        max_lag = filtered_df['Lag'].max() if not filtered_df.empty else None

        # Calculate optimal lags
        SIC_ast = int(np.argmin(result_table_IC["SIC"]))
        HQ_ast = int(np.argmin(result_table_IC["HQ"]))
        AIC_ast = int(np.argmin(result_table_IC["AIC"]))

        # Add new rows to the result table
        new_rows = pd.DataFrame([{'Lag Length': "-", 'SIC': "-", 'HQ': "-", 'AIC': "-", 'Top down': "-"},
                                 {'Lag Length': "Optimal Lag:", 'SIC': SIC_ast, 'HQ': HQ_ast, 'AIC': AIC_ast, 'Top down': "-"}])
        
        result_table_TD = pd.concat([result_table_IC, new_rows], ignore_index=True)
        result_table_df = pd.concat([merged_df, pd.DataFrame([{'Lag': "Optimal", 'SIC': int(SIC_ast), 'HQ': int(HQ_ast), 'AIC': int(AIC_ast), 'LR Test': ""}])], ignore_index=True)

        def format_result_table(df):
            df = df.copy()
            for col in ['SIC', 'HQ', 'AIC']:
                df[col] = df[col].astype(object)  # Undgå dtype-advarsler
                df.loc[df.index[:-1], col] = df.loc[df.index[:-1], col].map('{:.3f}'.format)
                df.loc[df.index[-1], col] = int(df.loc[df.index[-1], col])
            return df
        
        result_table_df = format_result_table(result_table_df)

        # Convert to LaTeX
        for k in range(len(result_table_TD.columns)-1):
            if k >= len(filtered_df):
                for j in range(maximum_lag_length+1):
                    result_table_TD[result_table_TD.columns[k]][j] = f"\\raisebox[[-0.75ex]][[${np.round(result_table_TD[result_table_TD.columns[k]][j],4)}$]]"
            else:
                for j in range(maximum_lag_length+1):
                    result_table_TD[result_table_TD.columns[k]][j] = f"${np.round(result_table_TD[result_table_TD.columns[k]][j],4)}$"
            
        
        latex_code = result_table_TD.to_latex(index=False, escape=False,
                                              column_format=f'{len(result_table_TD.columns) * "c"}',
                                              header=True, float_format="%.4f", multirow=True)

        # Modify LaTeX formatting
        latex_code = latex_code.replace("\\\\\n- & - & - & - & - \\\\\n",f"\\\\ \midrule\midrule \n").replace("[[", "{").replace("]]", "}")
        latex_code = latex_code.replace("\n\\toprule\nLag Length & SIC & HQ & AIC & Top down \\\\\n\\midrule", "\n\\toprule\n \\footnotesize  Lag  & \\footnotesize Schwarz Information & \\footnotesize Hannan-Quinn & \\footnotesize Akaike Information  & \\footnotesize Top down \\\\\n \\footnotesize Length & \\footnotesize Criterion& \\footnotesize Information Criterion & \\footnotesize Criterion& \\footnotesize Sequence\\\\\n\\midrule")
        latex_code = latex_code.replace("Optimal Lag: &","\\footnotesize Suggestion &")
        latex_code = latex_code.replace("\end{tabular}","\end{tabular}\n\captionsetup{width=12.5cm} \n\captionsetup{font=scriptsize}  \n\caption*{The Top Down Sequence shows the log-likelihood values, with likelihood ratio in parentheses. \nThe corresponding $p$-values are represented as follows: $p < 0.01$ is denoted by $^{***}$, $p < 0.05\\text{ by }^{**}$, and $p < 0.1\\text{ by }^{*}$.\n\end{table}")

        # Ensure the directory exists and save to file
        if self.LaTeX_path != None:
            latex_name = f"{self.LaTeX_path}/Table/lag_length.tex"
            os.makedirs(os.path.dirname(latex_name), exist_ok=True)
            with open(latex_name, "w") as f:
                f.write(latex_code)
            print("Saved to LaTeX Path","\n")
        
        if LaTeX == True:
            return print(latex_code,"\n\n Saved to LaTeX Path")
        else:
            return result_table_df.style.hide(axis="index")

class Diagnostic_Testing(BaseClass):

    def __init__(self, Base_initialize, y_dataframe):
        """
        Parameters:
            - y_dataframe      : (pd.DataFrame) | A (T, K) matrix / dataframe
            - Base_initialize  : (BaseClass)    |  An instance of BaseClass containing data attributes such as LaTeX path, lists of definitions, and the path.
        """
        self.y_dataframe = y_dataframe
        self.LaTeX_path = Base_initialize.LaTeX_path
        self.list_of_info_latex = Base_initialize.list_of_info_latex
        self.list_of_info = Base_initialize.list_of_info
        self.path = Base_initialize.path

    def VAR_estimation_with_exogenous(self, lags=int,Constant=bool,Trend=bool,Exogenous=None,y=None):
        """
        Estimates a VAR model with optional exogenous variables.

        Parameters
        -----------
        - lags     : int                            | Number of lags to include in the model.
        - constant : bool, optional, default=False  | If True, a constant term is included in the model.
        - trend    : bool, optional, default=False  | If True, a linear trend term is included in the model.
        - exog     : np.ndarray, optional           | Optional exogenous variables matrix, shape (n_obs, n_vars).

        Returns
        -----------
        - Beta          : np.ndarray | Estimated coefficient matrix, shape (n_lags, n_vars).
        - CovBeta       : np.ndarray | Covariance matrix of the estimated coefficients.
        - tratioBeta    : np.ndarray | T-statistics for the estimated coefficients.
        - residuals     : np.ndarray | Residuals of the model, shape (n_obs, n_vars).
        - X             : np.ndarray | The independent variable matrix used in the model.
        - SIGMA         : np.ndarray | Covariance matrix of residuals.

        """
        if y is None:
            T, K = self.y_dataframe.shape
            if isinstance(self.y_dataframe, pd.DataFrame):
                Y = self.y_dataframe.iloc[lags:T, :]
            else:
                Y = self.y_dataframe[lags:T, :]
            X = self.lagmatrix(self.y_dataframe,lags)  
        else:
            T, K = y.shape
            if isinstance(y, pd.DataFrame):
                Y = y.iloc[lags:T, :]
            else:
                Y = y[lags:T, :]
            X = self.lagmatrix(y,lags) 

        X = X[lags:, :]

        if Constant:
            X = np.hstack([X, np.ones((len(X), 1))])
            
        if Trend:
            X = np.hstack([X, np.arange(1, len(X) + 1).reshape(-1, 1)])
            
        if isinstance(Exogenous, np.ndarray):
            X = np.hstack([X, Exogenous])
            
        T, Kp = X.shape

        # Beta estimation
        Beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        Beta = np.array(Beta)
            
        # Residuals
        residuals = np.array(Y) - X @ Beta
            
        # Covariance matrix of residuals
        SIGMA = (residuals.T @ residuals) / T
            
        # Covariance matrix of Beta
        CovBeta = np.kron(SIGMA, np.linalg.inv(X.T @ X))
        tratioBeta = Beta.reshape(-1, order='F') / np.sqrt(np.diag(CovBeta))
        tratioBeta = tratioBeta.reshape(-1, K, order='F')
            
        return Beta, CovBeta, tratioBeta, residuals, X, SIGMA

    def Companion_matrix(self,Beta,lags=int):
        """
        Constructs the companion matrix for the VAR model.

        Parameters
        ----------
        - Beta : (np.ndarray) | Estimated coefficients from the VAR model.
        
        Returns
        ----------
        - A : (np.ndarray)    |  The companion matrix constructed from the coefficients.
        """

        K, _ = Beta.T.shape
        
        A = np.vstack([Beta[0:K*lags, 0:K].T, np.hstack([np.eye((lags-1)*K), np.zeros(((lags-1)*K, K))])])
        
        return A

    def Stability(self, Companion_matrix=np.ndarray):
        """
        Computes the eigenvalues of the companion matrix.
        
        Parameters
        ----------
        - Companion_matrix : ndarray | The companion matrix.
        
        Returns
        ----------
        - Eigenvalue : list    | Sorted list of absolute eigenvalues.
        """

        Eigenvalue = sorted(abs(np.linalg.eig(Companion_matrix)[0]), reverse=True)
        return Eigenvalue

    def lagmatrix(self, y_dataframe, lags=None):
        """
        Creates a lagged data matrix from the time series.

        Parameters
        ----------
        - lags        : (int)                     | The number of lags to create. If None, uses the class's lags attribute.
        - y_dataframe : (ndarray or pd.Dataframe) | The matrix/dataframe that to be lagged.

        Returns
        ----------
        - lagged_data : ndarray | Matrix of lagged observations.

        """

        T, K = y_dataframe.shape
        
        Lagged_data = np.zeros((T, K * lags))
        
        for lag in range(1, lags + 1):  
            if isinstance(y_dataframe, pd.DataFrame):
                Lagged_data[lag:, (lag - 1) * K:lag * K] = y_dataframe.iloc[:T - lag, :]
            else:
                Lagged_data[lag:, (lag - 1) * K:lag * K] = y_dataframe[:T - lag, :]
        
        return Lagged_data

    def Eigenvalues(self, lags=int, Constant=bool,Trend=bool,Exogenous=None):
        Beta, _, _, _, _, _ = self.VAR_estimation_with_exogenous(lags=lags,Constant=Constant,Trend=Trend,Exogenous=Exogenous)
        
        Companion_matrix = self.Companion_matrix(Beta=Beta,lags=lags)
        
        Eigenvalues = self.Stability(Companion_matrix=Companion_matrix)
        
        return Eigenvalues

    def Eigenvalue_Long_Table(self, maximum_lag_length=int,LaTeX=False):
        """
        Creates a table of eigenvalues based on different combinations of lags, constantbools, and trends.

        Parameters:
            maximum_lag_length  : int | lags to test.

        Returns
            Dataframe and/or LaTeX table
        """
        
        _, K = self.y_dataframe.shape

        # Initialize a list to hold data for the DataFrame
        table_data = []

        # Loop through different combinations of lags, constant, and trend
        for lags in range(1,int(maximum_lag_length)+1):
            # Temporary list to hold rows for the specific lag
            lag_rows = []  
            
            for constant in [False, True]:
                for trend in [False, True]:
                    # Run estimation
                    Eigenvalues = self.Eigenvalues(lags=lags)
                    Eigenvalues = self.Eigenvalues(lags=lags, Constant=constant,Trend=trend,Exogenous=None)

                    # Transpose eigenvalues (resulting in a (lags, 5) matrix)
                    Eigenvalues = np.array(Eigenvalues)
                    Eigenvalues_T = Eigenvalues.T

                    num_rows = Eigenvalues_T.shape[0] // K  # Calculate the number of rows
                    if Eigenvalues_T.shape[0] % K != 0:
                        num_rows += 1  # If there are leftovers, add an extra row

                    # Reshape to (n, 5), where n is the number of rows
                    Eigenvalues_T = np.reshape(Eigenvalues_T[:num_rows * K], (num_rows, K))

                    # If there is only one lag, flatten Eigenvalues_T
                    if Eigenvalues_T.ndim == 1:
                        # Handle cases where there is only one row (one lag)
                        row = [lags] + Eigenvalues_T.tolist()[:K]  # Store the first five values
                        row.append(bool(trend))  # Trend value
                        row.append(bool(constant))  # Constant value
                        lag_rows.append(row)
                    else:
                        for i in range(Eigenvalues_T.shape[0]):
                            row = [fr"$\lambda_{{p={i+1}}}$"]  # First column is lags
                            row.extend(Eigenvalues_T[i, :K])  # The next five columns are eigenvalues
                            row.append(bool(trend))  # Trend value
                            row.append(bool(constant))  # Constant value
                            lag_rows.append(row)

                    if lags > 1:
                        lag_rows.append([''] * len(lag_rows[0]))  # Add an empty row with the same number of columns

            # Add lag index column
            for row in lag_rows:
                if row[3] != "":
                    row.insert(0, f"Lag {lags}")  # Insert lag value at the beginning of the row

            table_data.extend(lag_rows)  # Append rows for the specific lag to the overall table
            table_data.append(['.'] * len(table_data[0]))  # Add an empty row with the same number of columns

        # Create DataFrame with the desired columns
        columns = ["Lags", "Eigenvalues"] + self.list_of_info_latex + ["Trend", "Constant"]
        df = pd.DataFrame(table_data, columns=columns)
        def replace_duplicates_with_empty(series):
            return series.where(series.duplicated(keep='first') == False, '')

        df['Lags'] = replace_duplicates_with_empty(df['Lags'])

        # Generate LaTeX table
        latex_table = df.to_latex(index=False, escape=False, column_format='l' + 'c' * (df.shape[1] - 1), float_format="%.3f")
        latex_table = latex_table.replace("NaN", r"")
        latex_table = latex_table.replace("None", r"")
        latex_table = latex_table.replace(f"{'&  ' * (K + 2 + 1)}\\\\", rf"\addlinespace\cdashline{{2-{K + 4}}}\addlinespace")
        latex_table = latex_table.replace(f"{' & .' * (K + 3)} \\\\", r"\addlinespace\midrule\addlinespace")
        latex_table = latex_table.replace(f" \\addlinespace\\cdashline{{2-{K + 4}}}\\addlinespace\n\\addlinespace\\midrule\\addlinespace\n\\bottomrule\n", "\\bottomrule\n")
        latex_table = latex_table.replace(f" \\addlinespace\\cdashline{{2-{K + 4}}}\\addlinespace\n\\addlinespace\\midrule\\addlinespace", "\\addlinespace\\midrule\\addlinespace")
        latex_table = latex_table.replace(".\\addlinespace\\midrule\\addlinespace", "\\addlinespace\\midrule\\addlinespace")
        latex_table = latex_table.replace("Lags &", "&")

        for i in range(1, int(maximum_lag_length)):
            latex_table = latex_table.replace(f"Lag {i} &", fr"\multirow{{{i * 4}}}{{*}}{{Lag {i}}} &")

        # Save the LaTeX table to a file
        if self.LaTeX_path != None:
            latex_name = f"{self.LaTeX_path}/Table/stable_appendix.tex"
            os.makedirs(os.path.dirname(latex_name), exist_ok=True)
            with open(latex_name, "w") as f:
                f.write(latex_table)
            print("Saved to LaTeX Path","\n")
        elif LaTeX == True:
            print(latex_table)
        
        df = df.map(lambda x: x.replace("$", "") if isinstance(x, str) else x)
        df = df.map(lambda x: x.replace(r"\lambda", "λ") if isinstance(x, str) else x)
        df.columns = [col.replace("$", "") if isinstance(col, str) else col for col in df.columns]
        df = df[~(df == ".").any(axis=1)].dropna().style.hide(axis="index")

        return df

    def Eigenvalue_Short_Table(self,maximum_lag_length=7, LaTeX=False):
        """
        Computes a table of maximum eigenvalues from SVAR estimation
        across different lag lengths and deterministic term combinations.

        Parameters:
        - maximum_lag_length: The maximum number of lags to test (default=7)

        Returns:
        - eigen_df: A pandas DataFrame with maximum eigenvalues per specification
        """
        _, K = self.y_dataframe.shape  # Only used here if you need K elsewhere

        table_data = []
        column_names = ["Lag","No Constant, No Trend","Constant Only","Trend Only","Constant and Trend"]

        # Loop over lag lengths
        for lags in range(1, maximum_lag_length + 1):
            row = [lags]

            # All combinations of constant and trend inclusion
            combinations = [
                (False, False),  # No constant, no trend
                (True, False),   # Constant only
                (False, True),   # Trend only
                (True, True)     # Constant and trend
            ]

            # Loop over deterministic term combinations
            for constant, trend in combinations:
                # Compute eigenvalues from the SVAR estimator
                Eigenvalues = self.Eigenvalues(lags=lags, Constant=constant, Trend=trend, Exogenous=None)

                # Ensure it's a NumPy array and transpose it
                Eigenvalues = np.array(Eigenvalues).T

                # Append the maximum eigenvalue to the row
                row.append(np.max(Eigenvalues))

            # Add the completed row to the table
            table_data.append(row)

        # Create and return the DataFrame
        eigen_df = pd.DataFrame(table_data, columns=column_names)

        latex_table = eigen_df.to_latex(index=False,float_format="%.6f",column_format="ccccc",    caption="\\\\\\sc Stability Diagnostics: Eigenvalues Across Lag Structures")
        latex_table = latex_table.replace(r"\begin{table}","\\begin{table}[H]\n\\footnotesize\n\centering")
        latex_table = latex_table.replace(r"\end{tabular}","\end{tabular}\n\captionsetup{width=0.8\linewidth, font=scriptsize}\n\caption*{The table reports the maximum eigenvalue for each VAR specification across lags and deterministic term combinations.}")
        
        if self.LaTeX_path != None:
            latex_name = f"{self.LaTeX_path}/Table/stable_short_appendix.tex"
            os.makedirs(os.path.dirname(latex_name), exist_ok=True)
            with open(latex_name, "w") as f:
                f.write(latex_table)
            print("Saved to LaTeX Path","\n")
        elif LaTeX == True:
            print(latex_table)

        return eigen_df.style.hide(axis="index")

    def Autocorrelation_LM_tests(self, lags=int, Constant=bool, Trend=bool, Exogenous=None, lags_h=int, display_data=True):
        import math
        t, K = self.y_dataframe.shape

        Beta, _, _, residuals_u, _, SIGMA_u = self.VAR_estimation_with_exogenous(lags=lags, Constant=Constant, Trend=Trend, Exogenous=Exogenous)

        Companion_matrix = self.Companion_matrix(Beta=Beta, lags=lags)
        Eigenvalues = self.Stability(Companion_matrix=Companion_matrix)
        residuals_u_h_lagged = self.lagmatrix(y_dataframe=residuals_u,lags=lags_h)

        _, _, _, _, _, SIGMA_e = self.VAR_estimation_with_exogenous(lags=lags, Constant=Constant, Trend=Trend, Exogenous=residuals_u_h_lagged)
        
        LM_test = (t - lags) * (K - np.trace(np.linalg.inv(SIGMA_u) @ SIGMA_e))

        degrees_of_freedom = lags_h * K**2
        LMLpval = 1 - chi2.cdf(LM_test, degrees_of_freedom)

        m = K * lags_h
        q = 1/2 * K * m -1
        s = ((K**2 * m**2 - 4)/(K**2 + m**2-5))**(1/2)

        N = (t - lags) - K*lags-m-1/2*(K-m+1)

        #FLMh =  ((np.linalg.det(SIGMA_u) * np.linalg.det(SIGMA_e)**(-1))**(1/s)-1) * (N*s-q)/(K*m) #
        FLMh = ((N*s-q)/(K*m))*(pow((np.linalg.det(SIGMA_u)/np.linalg.det(SIGMA_e)),1/s)   -1)
        
        degrees_of_freedom_numerator    = lags_h * K ** 2
        degrees_of_freedom_denominator  = (N * s - q)
        FLMh_pval = 1 - f.cdf(FLMh, degrees_of_freedom_numerator, degrees_of_freedom_denominator)
        # Lutkepohl (2004) p. 129/44

        Results = [[LM_test, FLMh], [LMLpval, FLMh_pval], [int(lags_h), int(lags_h)]]

        Results_table = pd.DataFrame({'Measure': pd.Categorical(['Test statistic', 'p-value', 'Lag order (h)']),
                                'Breusch Godfrey': [row[0] for row in Results],
                                'Edgerton Shukur': [row[1] for row in Results]})
        if display_data == True:
            display(Results_table)

        return Results_table  

    def Autocorrelation_LM_Table(self, lags=int, Constant=bool, Trend=bool, Exogenous=None, lags_h=int, LaTeX=True):    
        
        for hspace in ["\hspace[[1.0cm]]",""]:
            lag = []
            Breusch_Godfrey = []
            Edgerton_Shukur = []
            
            for h in range(1, lags_h + 1):
                Results = self.Autocorrelation_LM_tests(lags=lags, Constant=Constant, Trend=Trend, Exogenous=Exogenous, lags_h=h, display_data=False)
                
                if hspace == "":
                    lag.append(f'{h}')
                    Breusch_Godfrey.append(f'{round(Results["Breusch Godfrey"][0], 3)}{Initial_Tools.format_p_value(self,Results["Breusch Godfrey"][1])}')
                    Edgerton_Shukur.append(f'{round(Results["Edgerton Shukur"][0], 3)}{Initial_Tools.format_p_value(self,Results["Edgerton Shukur"][1])}')
                else:
                    lag.append(f'{hspace}{h}{hspace}')
                    Breusch_Godfrey.append(f'{hspace}${round(Results["Breusch Godfrey"][0], 3)}^{{{Initial_Tools.format_p_value(self,Results["Breusch Godfrey"][1])}}}${hspace}')
                    Edgerton_Shukur.append(f'{hspace}${round(Results["Edgerton Shukur"][0], 3)}^{{{Initial_Tools.format_p_value(self,Results["Edgerton Shukur"][1])}}}${hspace}')

            if hspace == "":
                results_df_1 = pd.DataFrame({'Lags ($h$)': lag,'Breusch Godfrey': Breusch_Godfrey,'Edgerton Shukur': Edgerton_Shukur})
            else:
                results_df = pd.DataFrame({'Lags ($h$)': lag,'Breusch Godfrey': Breusch_Godfrey,'Edgerton Shukur': Edgerton_Shukur})

        
        # Caption text determination
        if Constant == 0 and Trend == 0:
            text_caption = "without a trend nor a constant"
        elif Constant == 1 and Trend == 0:
            text_caption = "with a constant"
        elif Constant == 0 and Trend == 1:
            text_caption = "with a trend"
        elif Constant == 1 and Trend == 1:
            text_caption = "with a trend and a constant"

        # Create LaTeX table
        latex_table = results_df.to_latex(index=False, escape=False, column_format='ccc')
        latex_table = latex_table.replace("[[", "{").replace("]]", "}")
        latex_table = latex_table.replace("\end{tabular}","\end{tabular}\n\captionsetup{width=9cm} \n\captionsetup{font=scriptsize}  \n\caption*{The test is based on a VAR model REPLACE_textcaption_REPLACE and with REPLACE_p_REPLACE lags.\nThe corresponding $p$-values are represented as follows: $p < 0.01$ is denoted by $^{***}$, $p < 0.05\\text{ by }^{**}$, and $p < 0.1\\text{ by }^{*}$.} \n\end{table}")
        latex_table = latex_table.replace("REPLACE_textcaption_REPLACE", text_caption)
        latex_table = latex_table.replace("REPLACE_p_REPLACE", f"{lags}")

        # Save to .tex file
        table_name = "Breusch_Edgerton_table"
        
        if self.LaTeX_path != None:
            latex_name = f"{self.LaTeX_path}/Table/{table_name}.tex"
            os.makedirs(os.path.dirname(latex_name), exist_ok=True)
            with open(latex_name, "w") as f:
                f.write(latex_table)
            print("Saved to LaTeX Path","\n")
        
        if LaTeX == True:
            print(latex_table)
        
        results_df_1 = results_df_1.rename(columns={"Lags ($h$)":"Lags (h)"})
        
        return results_df_1.style.hide(axis="index")

    def Multivariate_ARCH(self,residuals=np.ndarray, lags_q=int, display_data=False):
        """
        Test for Multivariate ARCH
        
        Parameters
        -------------
        - residual : (np.ndarray) | (T x K) residual matrix
        - lags_q   : (int)        | Number of lagged ARCH effects
        - K        : (int)        | Number of variables in the underlying VAR/VEC model
        
        Returns
        -------------
        - Function produces a table with results
        """

        T = residuals.shape[0]
        K = len(self.y_dataframe.columns)
        
        # Subtract the mean from each column
        uhat = residuals - residuals.mean(axis=0)
        
        # Initialize list to store UUT columns
        UUT_list = []
        
        # Construct UUT by stacking 'vech' of uhat[i,:]' * uhat[i,:]
        for i in range(T):
            utut = np.outer(uhat[i, :], uhat[i, :]) # K x K matrix
            tmp = []
            
            for j in range(K):
                tmp.extend(utut[j:, j]) # Collect lower triangular elements including diagonal
            
            UUT_list.append(tmp)
        
        # Convert list to numpy array and transpose
        UUT = np.array(UUT_list).T # Shape: (K*(K+1)//2, T)
        
        # Create matrices of regressors
        Y = UUT[:, lags_q:] # Shape: (K*(K+1)//2, T - lags)
        T_Y = Y.shape[1]
        Z_list = []
        
        for i in range(T_Y):
            temp = UUT[:, i:i + lags_q] # Shape: (K*(K+1)//2, lags)
            temp_vec = temp.flatten(order='F') # Vectorize in column-major order
            col = np.vstack(([1], temp_vec.reshape(-1, 1))) # Prepend a 1
            Z_list.append(col)
        
        # Stack columns horizontally to form Z
        Z = np.hstack(Z_list) # Shape: (1 + K*(K+1)//2 * lags, T_Y)
        
        # Compute omega
        A = Y @ Z.T @ np.linalg.inv(Z @ Z.T)
        residuals = Y - A @ Z
        omega = residuals @ residuals.T / T_Y
        
        # Compute omega0
        Y_mean = Y.mean(axis=1, keepdims=True)
        omega0 = (Y - Y_mean) @ (Y - Y_mean).T / T_Y
        
        # Compute R-squared value
        R2 = 1 - (2 / (K * (K + 1)) * np.trace(omega @ np.linalg.inv(omega0)))
        
        # Compute test statistic
        VARCHLM = 0.5 * T_Y * K * (K + 1) * R2
        
        # Degrees of freedom
        df = lags_q * K ** 2 * (K + 1) ** 2 / 4
        
        # Compute p-value
        pvalue = 1 - chi2.cdf(VARCHLM, df)
        
        # Collect test results
        test = np.array([VARCHLM, pvalue, df])
        
        # Create result table
        march_table = pd.DataFrame({'Test': ['Test statistic', 'P-value', 'Degrees of freedom'],'Doornik_Hendry': test})
        
        if display_data==True:
            display(march_table.style.hide(axis="index"))

        return march_table

    def Multivariate_Normality(self,residuals=np.ndarray,display_data=False):

        """
        Purpose
        --------------
        Multivariate test on normality. H0: Gaussian data generation process
        
        Parameters
        --------------
        - residual : (np.ndarray) | A (T x K) matrix of residuals
        
        Return
        --------------
        - norm : (np.ndarray) | A (7 x 2) NumPy array containing test statistics and p-values. Rows correspond to for each column [Doornik_Hansen, Lutkepohl].
            1. Joint test statistic, 
                - P-value, 
                - Degrees of freedom,

            2. Skewness only, 
                - P-value, 

            3. Kurtosis only, 
                - P-value 

        - multnorm_table: (pd.DataFrame) | A pandas DataFrame summarizing the results.
        
        References:
            - Lütkepohl (1993), Introduction to Multiple Time Series Analysis, 2nd ed., p. 150.
            - Doornik & Hansen (1994) Based on JMulTi Gauss procedure by Michael Bergman.
        """

        n, k = residuals.shape
        
        umat = residuals - np.mean(residuals, axis=0)
        Su = (1 / n) * (residuals.T @ residuals)
        
        lambda_, Pmat = np.linalg.eig(Su)
        lambda_diag = np.diag(lambda_)
        
        x = np.sqrt(np.diag(Pmat.T @ Pmat))
        
        rP, cP = Pmat.shape
        
        for i in range(cP):
            Pmat[:, i] = Pmat[:, i] / x[i]
        
        sqrt_lambda = np.sqrt(lambda_diag)
        Q = Pmat @ sqrt_lambda @ Pmat.T
        v1 = np.linalg.inv(Q) @ umat.T
        L = np.linalg.cholesky(Su).T
        v2 = np.linalg.inv(L.T) @ umat.T
        
        b21 = (np.sum(v1.T ** 4, axis=0) / n).T
        b11 = (np.sum(v1.T ** 3, axis=0) / n).T
        b22 = (np.sum(v2.T ** 4, axis=0) / n).T
        b12 = (np.sum(v2.T ** 3, axis=0) / n).T
        
        l11 = n * b11.T @ b11 / 6
        pskew1 = 1 - chi2.cdf(l11, df=k)
        
        l12 = n * b12.T @ b12 / 6
        pskew2 = 1 - chi2.cdf(l12, df=k)
        
        l21 = n * (b21 - 3).T @ (b21 - 3) / 24
        pkurt1 = 1 - chi2.cdf(l21, df=k)
        
        l22 = n * (b22 - 3).T @ (b22 - 3) / 24
        pkurt2 = 1 - chi2.cdf(l22, df=k)
        
        NormDf = 2 * k
        l31 = l11 + l21
        Normpv1 = 1 - chi2.cdf(l31, df=NormDf)
        l32 = l12 + l22
        Normpv2 = 1 - chi2.cdf(l32, df=NormDf)
        
        norm = np.array([[l31, l32],
                        [Normpv1, Normpv2],
                        [NormDf, NormDf],
                        [l11, l12],
                        [pskew1, pskew2],
                        [l21, l22],
                        [pkurt1, pkurt2]])
        
        tests = ['Joint test statistic:', 'P-value', 'Degrees of freedom',
                    'Skewness only', 'P-value', 'Kurtosis only', 'P-value']
        
        multnorm_table = pd.DataFrame({'Test': tests,
                                    'Doornik_Hansen': np.round(norm[:, 0], 5),
                                    'Lutkepohl': np.round(norm[:, 1], 5)})
        if display_data==True:
            display(multnorm_table.style.hide(axis="index"))

        return multnorm_table

    def Multivariate_Portmanteau(self,residuals=np.ndarray, lag_h=int, lags=int, display_data=False):
        """
        Computes Portmanteau test (Multivariate Ljung-Box test) for autocorrelation
        
        Parameters
        --------------------
        - res : np.array  |  Matrix (T x K) of estimated residuals.
        - h   : int       |  Maximum number of lags for Portmanteau test.
        - p   : int       |  Number of lags in estimated VAR(p), used for calculating degrees of freedom.
    
        Returns 
        --------------------
        - results : (pd.DataFrame) | A pandas DataFrame summarizing the results.

        """

        T, K = residuals.shape

        ################################################################################
        # Portmanteau test
        ################################################################################
        
        Q = []
        C_0 = (1/T) * (residuals.T @ residuals)

        for i in range(1, lag_h + 1):
            lagged_m = np.vstack([np.zeros((i, K)), residuals[:-i, :]]) 
            
            C_j = (1/T) * (residuals[i:, :].T @ lagged_m[i:, :])
            Q.append(np.trace(C_j.T @ np.linalg.inv(C_0) @ C_j @ np.linalg.inv(C_0)))

        ################################################################################
        # Modified Portmanteau test
        ################################################################################
        
        Q_modified = []
        C_0 = (1/T) * (residuals.T @ residuals)

        for i in range(1, lag_h + 1):
            lagged_m = np.vstack([np.zeros((i, K)), residuals[:-i, :]])
            
            C_j = (1/T) * (residuals[i:, :].T @ lagged_m[i:, :])
            Q_modified.append(1/(T - i) * np.trace(C_j.T @ np.linalg.inv(C_0) @ C_j @ np.linalg.inv(C_0)))

        ################################################################################
        ################################################################################
        
        degree_of_freedom = K**2 * (lag_h - lags)

        Q_h = T * np.sum(Q)
        p_Value = chi2.sf(Q_h, degree_of_freedom)
        
        Q_h_modified = T**2 * np.sum(Q_modified)
        p_Value_modified = chi2.sf(Q_h_modified, degree_of_freedom)


        results = pd.DataFrame({'' : ['Test Statistic', 'p-value', 'Degrees of Freedom'],
                                'Portmanteau': [Q_h, p_Value, degree_of_freedom],
                                'Modified Portmanteau': [Q_h_modified, p_Value_modified, degree_of_freedom]})

        if display_data==True:
            display(results.style.hide(axis="index"))

        return results

    def Diagnostic_Table(self, residuals=np.ndarray, Constant=bool, Trend=bool, lags=int, lags_h=int, lags_q=int, LaTeX=False):
        
        Results_ARCH = self.Multivariate_ARCH(residuals=residuals, lags_q=lags_q)
        Results_NORM = self.Multivariate_Normality(residuals=residuals)
        Results_AUTO = self.Multivariate_Portmanteau(residuals=residuals, lag_h=lags_h, lags=lags)

        def format_test_result(result, indices):
            return [fr'$\underset[[({round(result[indices[2]])})]][[{round(result[indices[0]],4)}]]^[[{Initial_Tools.format_p_value(self,result[indices[1]])}]]$',
                    fr'${round(result[indices[3]],4)}^[[{Initial_Tools.format_p_value(self,result[indices[4]])}]]$' if len(indices) > 4 else "",
                    fr'${round(result[indices[5]],4)}^[[{Initial_Tools.format_p_value(self,result[indices[6]])}]]$' if len(indices) > 6 else ""]
        
        def format_test_result_df(result, indices):
            return [fr'{round(result[indices[0]],3)}{Initial_Tools.format_p_value(self,result[indices[1]])}',
                    fr'{round(result[indices[3]],3)}{Initial_Tools.format_p_value(self,result[indices[4]])}' if len(indices) > 4 else "",
                    fr'{round(result[indices[5]],3)}{Initial_Tools.format_p_value(self,result[indices[6]])}' if len(indices) > 6 else ""]
        
        Lutkepohl = format_test_result(Results_NORM["Lutkepohl"], [0, 1, 2, 3, 4, 5, 6])
        portmanteau = format_test_result(Results_AUTO["Portmanteau"], [0, 1, 2])
        Doornik_Hansen = format_test_result(Results_NORM["Doornik_Hansen"], [0, 1, 2, 3, 4, 5, 6])
        Doornik_Hendry = format_test_result(Results_ARCH["Doornik_Hendry"], [0, 1, 2])
        portmanteau_modified = format_test_result(Results_AUTO["Modified Portmanteau"], [0, 1, 2])

        Lutkepohl_df = format_test_result_df(Results_NORM["Lutkepohl"], [0, 1, 2, 3, 4, 5, 6])
        portmanteau_df = format_test_result_df(Results_AUTO["Portmanteau"], [0, 1, 2])
        Doornik_Hansen_df = format_test_result_df(Results_NORM["Doornik_Hansen"], [0, 1, 2, 3, 4, 5, 6])
        Doornik_Hendry_df = format_test_result_df(Results_ARCH["Doornik_Hendry"], [0, 1, 2])
        portmanteau_modified_df = format_test_result_df(Results_AUTO["Modified Portmanteau"], [0, 1, 2])

        # Row names for the tests
        row_name = ["Joint test", r"Skewness test \hspace{0.15cm}", "Kurtosis test"]
        row_name_df = ["Joint test", "Skewness test", "Kurtosis test"]

        # Create DataFrame for results
        results = pd.DataFrame({'': row_name,'\hspace{0.25cm} Doornik Hendry \hspace{0.25cm}': Doornik_Hendry,'Doornik Hansen': Doornik_Hansen,'Lutkepohl': Lutkepohl,'Portmanteau': portmanteau,'Portmanteau$^\dagger$': portmanteau_modified})
        results_df = pd.DataFrame({'': row_name_df,'Doornik Hendry': Doornik_Hendry_df,'Doornik Hansen': Doornik_Hansen_df,'Lutkepohl': Lutkepohl_df,'Portmanteau': portmanteau_df,'Portmanteau Modified': portmanteau_modified_df})

        # Set text caption based on constant and trend values
        if Constant == False and Trend == False:
            text_caption = "without a trend nor a constant"
        elif Constant == True and Trend == False:
            text_caption = "with a constant"
        elif Constant == False and Trend == True:
            text_caption = "with a trend"
        elif Constant == True and Trend == True:
            text_caption = "with a trend and a constant"

        latex_table = results.to_latex(index=False, escape=False, column_format='lccccc')
        latex_table = latex_table.replace("[[", "{").replace("]]", "}")
        latex_table = latex_table.replace(f"\\\\\n{row_name[1]}", "\\\\\\addlinespace\\cdashline{1-6}\\addlinespace\nSkewness test")
        latex_table = latex_table.replace("\\\\\nKurtosis test", "\\\\\\addlinespace\\addlinespace\nKurtosis test")
        latex_table = latex_table.replace("Joint test", r"\raisebox{-0.4em}{Joint test}")
        latex_table = latex_table.replace("\end{tabular}", "\end{tabular}\n\\captionsetup{width=14cm} \n\\captionsetup{font=scriptsize}  \n\\caption*{The test is based on a VAR model REPLACE_textcaption_REPLACE and REPLACE_p_REPLACE lags. $p$-values are denoted as: $p < 0.01$ by $^{***}$, $p < 0.05$ by $^{**}$, and $p < 0.1$ by $^{*}$. Degrees of freedom are in parentheses. The Portmanteau test and the modified Portmanteau test, ($\dagger$), by \\texttt{Hosking1980} checks for autocorrelation in residuals at lag max_h_lags.} \n\\end{table}")
        latex_table = latex_table.replace("REPLACE_textcaption_REPLACE", f"{text_caption}")
        latex_table = latex_table.replace("REPLACE_p_REPLACE", f"{lags}")
        latex_table = latex_table.replace("max_h_lags", f"{lags_h}")
        latex_table = latex_table.replace("\\toprule\n", "\\toprule\n&\\multicolumn{1}{c}{\\sc Arch}  & \\multicolumn{2}{c}{\\sc Normality} & \\multicolumn{2}{c}{\\sc Autocorrelation}\\\\\n\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n")

        # Save the LaTeX table if a path is provided
        if self.LaTeX_path != None:
            latex_name = fr"{self.LaTeX_path}/Table/diagnostic_testing.tex"
            os.makedirs(os.path.dirname(latex_name), exist_ok=True)
            with open(latex_name, "w") as fh:
                fh.write(latex_table)
            print("Saved to LaTeX Path","\n")
        
        if LaTeX == True:
            print(latex_table)

        return results_df.style.hide(axis="index")

class Univariate_Diagnostic_Testing(BaseClass):
    
    def __init__(self, Base_initialize, y_dataframe):
        """
        Parameters:
            - y_dataframe      : (pd.DataFrame) | A (T, K) matrix / dataframe
            - Base_initialize  : (BaseClass)    |  An instance of BaseClass containing data attributes such as LaTeX path, lists of definitions, and the path.
        """
        self.y_dataframe = y_dataframe
        self.LaTeX_path = Base_initialize.LaTeX_path
        self.list_of_info_latex = Base_initialize.list_of_info_latex
        self.list_of_info = Base_initialize.list_of_info
        self.path = Base_initialize.path

    def Univariate_ARCH_Test(self,residuals=np.ndarray, lags=int):
        """
        Perform the ARCH test on the residuals.

        Parameters:
            residuals  : (T x K) numpy array of residuals.
            lags       : Number of lags for ARCH test.

        Returns:
        - DataFrame with results from the ARCH test.
        """
        K = residuals.shape[1]

        hypothesis_results = np.zeros(K)
        p_values_results = np.zeros(K)
        stat_results = np.zeros(K)
        crit_value_results = np.zeros(K)
        
        # Perform ARCH test for each variable
        for i in range(K):
            stat, pValue, fval, _ = het_arch(residuals[:, i], nlags=lags)
            hypothesis_results[i] = 1 if pValue < 0.05 else 0
            p_values_results[i] = pValue
            stat_results[i] = stat
            crit_value_results[i] = chi2.ppf(1 - 0.05, lags)

        Arch_table = pd.DataFrame({'Hypothesis': hypothesis_results, 'P-Value': p_values_results, 
                                   'Test Statistic': stat_results, 'Critical Value': crit_value_results}, index=self.list_of_info[1:])
        
        return Arch_table
    
    def Univariate_Normality_Test(self,residuals=np.ndarray, Combined=False):
        """
        Perform the Jarque-Bera test on the residuals.

        Parameters:
        - residuals:    T x K matrix of residuals.
        - columns_list: List of column names for the results.

        Returns:
        - DataFrame with results from the Jarque-Bera test.
        """

        K = residuals.shape[1]

        hypothesis_results = np.zeros(K)
        p_values_results = np.zeros(K)
        stat_results = np.zeros(K)
        crit_value_results = np.zeros(K)

        # Perform Jarque-Bera test for each variable
        for i in range(K):
            (JB,JBpv,skew,kurtosis) = jarque_bera(residuals[:, i])
            hypothesis_results[i] = 1 if JBpv < 0.05 else 0  # Hypothesis result based on p-value
            p_values_results[i] = JBpv
            stat_results[i] = JB
            crit_value_results[i] = chi2.ppf(1 - 0.05, 2)  # Degrees of freedom for JB test is 2

        Norm_table = pd.DataFrame({'Hypothesis': hypothesis_results, 'P-Value': p_values_results,
                                 'Test Statistic': stat_results, 'Critical Value': crit_value_results}, index=self.list_of_info_latex)
        
        Norm_table_df = pd.DataFrame({'Hypothesis': hypothesis_results, 'P-Value': p_values_results,
                                 'Test Statistic': stat_results, 'Critical Value': crit_value_results}, index=self.list_of_info[1:])
        if Combined == True:
            return Norm_table
        else:
            return Norm_table_df

    def Univariate_Autocorrelation_Test(self,residuals=np.ndarray, lag_h=int, Combined=False):
        """
        Perform the Ljung-Box test on the residuals.

        Parameters:
        - residuals:    K x K matrix of residuals.
        - columns_list: List of column names for the results.
        - max_lag_length: Maximum lag length for the Ljung-Box test.

        Returns:
        - DataFrame with results from the Ljung-Box test.
        """
        
        K = residuals.shape[1]

        lags_range = range(1, lag_h + 1)
        p_values_array = np.zeros((len(lags_range), K))
        lb_stat_array = np.zeros((len(lags_range), K))

        # Perform Ljung-Box test for each residual series
        for i in range(K):
            df = acorr_ljungbox(residuals[:, i], lags=lags_range, return_df=True)
            p_values_array[:, i] = df['lb_pvalue'].values
            lb_stat_array[:, i] = df['lb_stat'].values
        
        acorr_ljungbox_test = []  # Initialize an empty list
        acorr_ljungbox_test_df = []
        for i in range(len(p_values_array)):  # Loop over the first dimension
            inner_list = []  # Create a new inner list
            inner_list_df = []
            
            for k in range(len(p_values_array[0])):  # Loop over the second dimension
                inner_list.append(f"${round(lb_stat_array[i][k], 4)}^[[{Initial_Tools.format_p_value(self,p_values_array[i][k])}]]$")  # Append the value
                inner_list_df.append(f"{round(lb_stat_array[i][k], 4)}{Initial_Tools.format_p_value(self,p_values_array[i][k])}")  # Append the value

            acorr_ljungbox_test.append((fr"\qquad Lag {i+1}\hspace{{0.7cm}}", inner_list))  # Append the inner list as a tuple
            acorr_ljungbox_test_df.append((fr"Lag {i+1}", inner_list_df))

        # Assuming acorr_ljungbox_test is already defined as in your code
        data = {lag: values for lag, values in acorr_ljungbox_test}
        data_df = {lag: values for lag, values in acorr_ljungbox_test_df}
        Auto_table = pd.DataFrame(data).T  # Transpose the DataFrame
        Auto_table_df = pd.DataFrame(data_df).T  # Transpose the DataFrame
        Auto_table.columns = self.list_of_info_latex # or formatted_index #
        Auto_table_df.columns = self.list_of_info[1:]
        
        if Combined == True:
            return Auto_table
        else:
            return Auto_table_df
            
    def Combined_Univariate_LaTeX(self, residuals=np.ndarray,lags=int, lag_h=int, LaTeX=False):
        """
        Generate LaTeX tables for Engle ARCH, Jarque-Bera tests, and autocorrelation results.

        Parameters:
        - het_table: DataFrame containing results of the Engle ARCH test.
        - jb_table: DataFrame containing results of the Jarque-Bera test.
        - autocorrelation_uni: DataFrame containing autocorrelation results.
        - lags_range: Range of lags used for the autocorrelation tests.

        Returns:
        - LaTeX tables for both ARCH, Jarque-Bera tests, and autocorrelation results as strings.
        """
        Arch_table = self.Univariate_ARCH_Test(residuals=residuals, lags=lags)
        Norm_table = self.Univariate_Normality_Test(residuals=residuals, Combined=True)
        Auto_table = self.Univariate_Autocorrelation_Test(residuals=residuals, lag_h=lag_h, Combined=True)

        # Initialize lists for tests
        jarque_bera_test = []
        engle_arch_test = []

        # Process Jarque-Bera and ARCH test results
        for i in range(0,len(Norm_table)):
            jarque_bera_test.append(fr"\hspace[[1cm]] $\underset[[({round(Norm_table['Critical Value'].iloc[i],2)})]][[{round(Norm_table['Test Statistic'].iloc[i],1)}^[[{Initial_Tools.format_p_value(self,Norm_table['P-Value'].iloc[i])}]]]]$")
            engle_arch_test.append(fr"$\underset[[({round(Arch_table['Critical Value'].iloc[i],2)})]][[{round(Arch_table['Test Statistic'].iloc[i],3)}^[[{Initial_Tools.format_p_value(self,Arch_table['P-Value'].iloc[i])}]]]]$")

        # Format the index for the LaTeX table
        formatted_index = [f'\\raisebox[[-0.3em]][[{self.list_of_info[1:][l]} ({self.list_of_info_latex[l]})]]' for l in range(len(self.list_of_info[1:]))]

        # Create the ARCH and Jarque-Bera table
        test_arch_norm = pd.DataFrame({
            r'\sc Engle {\scriptsize ARCH} Test ': engle_arch_test,
            r'\sc \hspace{1cm} Jarque-Bera Test': jarque_bera_test,
        }, index=formatted_index)

        # Generate LaTeX for ARCH and Jarque-Bera test
        latex_table_arch_jarque = test_arch_norm.to_latex(index=True, escape=False, column_format='lccc')
        latex_table_arch_jarque = latex_table_arch_jarque.replace("[[", "{").replace("]]", "}")
        latex_table_arch_jarque = latex_table_arch_jarque.replace("\\\\", "\\\\")
        latex_table_arch_jarque = latex_table_arch_jarque.replace(
            "\\end{tabular}",
            "\\end{tabular}\n\\captionsetup{width=12cm} \n\\captionsetup{font=scriptsize}  \n\\caption*{The corresponding $p$-values are represented as follows: $p < 0.01$ is denoted by $^{}$, $p < 0.05$ by $^{}$, and $p < 0.1$ by $^{}$.\n Note that the critical values are indicated in parentheses. \nAdditionally, these tests were conducted using the Python module \\codepy{\\scriptsize statsmodels.stats} for \\codepydefine{\\scriptsize jarque\\_bera} and \\codepydefine{\\scriptsize het\\_arch}}.\n\\end{table}"
        )
        latex_table_arch_jarque = latex_table_arch_jarque.replace(
            "\\raisebox{-0.3em}{Gross Domestic Product}",
            "\\raisebox{-0.3em}{Gross Domestic Product}\\quad"
        )

        # Save the ARCH and Jarque-Bera LaTeX table
        if self.LaTeX_path != None:
            table_name = "uni_norm_arch"
            latex_name = fr"{self.LaTeX_path}/Table/{table_name}.tex"
            os.makedirs(os.path.dirname(latex_name), exist_ok=True)
            with open(latex_name, "w") as fh:
                fh.write(latex_table_arch_jarque)
            print("Saved to LaTeX Path","\n")
        
        # Generate LaTeX for autocorrelation results
        latex_table_autocorrelation = Auto_table.to_latex(index=True, escape=False, column_format=f'l{"c" * len(self.list_of_info[1:])}')
        latex_table_autocorrelation = latex_table_autocorrelation.replace("[[", "{").replace("]]", "}")
        latex_table_autocorrelation = latex_table_autocorrelation.replace(
            " & Fed Funds Rate ($i^\\ast_t$) & Gross Domestic Product ($y_t$) & Inflation ($\\pi_t$) & Interest Rate ($i_t$) & Exchange Rate ($e_t$) \\\\",
            "& Fed Funds & Gross Domestic & Inflation & Interest & Exchange  \\\\\n& Rate ($i^\\ast_t$) & Product ($y_t$) & ($\\pi_t$) & Rate ($i_t$) & Rate ($e_t$) \\\\")

        # Save the autocorrelation LaTeX table
        if self.LaTeX_path != None:
            table_name_autocorrelation = "uni_autocorrelation"
            latex_name_autocorrelation = fr"{self.LaTeX_path}/Table/{table_name_autocorrelation}.tex"
            os.makedirs(os.path.dirname(latex_name_autocorrelation), exist_ok=True)
            with open(latex_name_autocorrelation, "w") as fh:
                fh.write(latex_table_autocorrelation)
            print("Saved to LaTeX Path","\n")

        if LaTeX == True:
            return print(latex_table_arch_jarque,"\n\n",latex_table_autocorrelation)
        else:
            return 

class Testing_Cointegration(BaseClass):

    def __init__(self, Base_initialize, y_dataframe):
        """
        Parameters:
            - y_dataframe      : (pd.DataFrame) | A (T, K) matrix / dataframe
            - Base_initialize  : (BaseClass)    |  An instance of BaseClass containing data attributes such as LaTeX path, lists of definitions, and the path.
        """
        self.y_dataframe = y_dataframe
        self.LaTeX_path = Base_initialize.LaTeX_path
        self.list_of_info_latex = Base_initialize.list_of_info_latex
        self.list_of_info = Base_initialize.list_of_info
        self.path = Base_initialize.path

        # Create an instance of Diagnostic_Testing and pass y_dataframe
        self.diagnostic_tester = Diagnostic_Testing(Base_initialize,y_dataframe)  # Pass y_dataframe to Diagnostic_Testing class

    def Identication_Scheme(self,rank=int):
        print(f"Since the rank r = {int(rank)}, there are {int(rank)} transitory/temporary shocks.")
        print(f"Thus, there are K-r = {self.y_dataframe.shape[1]-int(rank)} permanent shocks.")
        print()
        print(f"    To identify the permanent shocks, we need (K-r)(K-r-1)/2 = {int((self.y_dataframe.shape[1]-int(rank))*(self.y_dataframe.shape[1]-int(rank)-1)/2)} restrictions. (Upsilon)")
        print(f"    To identify the transitory shocks, we need r(r-1)/2 = {int(int(rank)*(int(rank)-1)/2)} restrictions. (B^{{-1}}_0)")

    def Vectorization(self,vector=np.ndarray):
        """
        **Based on Lutz Kilian's Matlab code**\n
        This function vectorizes an (a x b) matrix y. The resulting vector vecy has dimension (a*b x 1).
        -  Michael Bergman 2023
        """
        [row,column] = vector.shape
        vecy = vector.reshape(row*column,1)
        return vecy

    def Granger_causality_test(self, lags=int, Constant=bool, Trend=bool,Exogenous=np.ndarray):
        """
        **Performs Granger Causality (GC) tests** between two variables using Wald Chi-squared and F-tests.

        Parameters
        -------------
            lags       : (int)                  |  Number of lags used in the VAR model.
            Constant   : (bool)                 |  Whether to include a constant term in the VAR estimation.
            Trend      : (bool)                 |  Whether to include a trend term in the VAR estimation.
            Exogenous  : (np.ndarray or None) | Optional exogenous variables (NumPy array). If no exogenous variables, pass an integer (e.g., 0).

        Returns
        -------------
        - None: 
            - Prints the results of the Wald Chi-squared and F-tests for GC between the two variables.
        """

        T,K = self.y_dataframe.shape

        Beta, CovBeta, _, _, _, _ = self.diagnostic_tester.VAR_estimation_with_exogenous(lags=lags,Constant=Constant,Trend=Trend,Exogenous=Exogenous)

        varbeta = CovBeta
        bhat = self.Vectorization(Beta)

        # if isinstance(Exogenous, int):
        #     exog_col = 0
        # else:
        #     exog_len, exog_col = Exogenous.shape

        if isinstance(Exogenous, np.ndarray):  # Check if Exogenous is a NumPy array
            exog_len, exog_col = Exogenous.shape
        else:
            exog_col = 0  # Assign default values or handle accordingly

        R1 = np.zeros([lags,K*K*lags+K*Constant+K*Trend+K*exog_col])

        for i in range(0, lags):
            R1[i,i*(K*2)+K] = 1

        # GC from variable 1 on variable 2

        R2 = np.zeros([lags,K*K*lags+K*Constant+K*Trend+K*exog_col])

        for i in range(0, lags):
            R2[i,i*(K*2)+K - 1] = 1


        # Then we can compute the test statistics.
        # Note that the LR chi2 test reported above is not the same as
        # the Wald test.   

        Q1 = ((-R1 @ bhat).T @ np.linalg.inv((R1 @ varbeta) @ R1.T)) @ (-R1 @ bhat)
        pvalQ1 = 1 - chi2.cdf(Q1,lags)
        
        Q1F = Q1/lags
        dgf = (T - lags)-K*lags-1
        pvalQ1F = 1 - f.cdf(Q1F,lags,dgf)

        Q2 = ((-R2 @ bhat).T @ np.linalg.inv((R2 @ varbeta) @ R2.T)) @ (-R2 @ bhat)
        pvalQ2 = 1 - chi2.cdf(Q2,lags)
        Q2F = Q2/lags
        dgf = (T - lags)-K*lags-1
        pvalQ2F = 1 - f.cdf(Q2F,lags,dgf)

        print('Results own GC tests\n')
        print('   Testing GC from variable 2 on variable 1')
        print('      Wald Chi2 test',round(Q1[0][0],3),'with p-value',round(pvalQ1[0][0],3),'with',lags,'degrees of freedom')
        print('      Wald F test',round(Q1F[0][0],3),'with p-value',round(pvalQ1F[0][0],3),'with df_denom',dgf,', df_num',lags,'\n')
        print('   Testing GC from variable 1 on variable 2')
        print('      Wald Chi2 test',round(Q2[0][0],3),'with p-value',round(pvalQ2[0][0],3),'with',lags,'degrees of freedom')
        print('      Wald F test',round(Q2F[0][0],3),'with p-value',round(pvalQ2F[0][0],3),'with df_denom',dgf,', df_num',lags)

    def LSKnownBeta(self, lags=int, beta=np.ndarray, model=int,y=None):
        """
        % y = Data in levels must be an array T x K
        % p = #lags in underlying VAR
        % beta = cointegration vector, i.e., beta' but remember that it must include
                any deterministic components
        %
        % Note: function requires specification of the deterministic components. We use
        % the Matlab conventions to define models.
        %
        % model = 1 corresponds to model H2 ("n" in Python)
        % model = 2 corresponds to model H1* ("co" in Python)
        % model = 3 corresponds to model H1 ("ci" in Python)
        % model = 4 corresponds to model H* ("cili" in Python)
        % model = 5 corresponds to model H ("colo" in Python)
        %
        %
        % Output:
        % Beta: K x (p-1) parameters associated with first differences;
        %       K x r error correction terms
        %       K x 1 constant terms (if included in the model)
        %       K x 1 linear trend terms and (if included in the model)
        % 
        % Betavec: vectorized Beta (vector)      
        % SEBeta: Standard errors associated with each parameter (vector)
        % tratioBeta: t-ratios for all parameters (vector)
        % res: Residuals from VEC estimates
        % so: variance-covariance matrix
        % so_ml: ML estimate of variance-covariance matrix
        % 
        % Note: The cointegration vector beta used as input must include constant and/or trend
        %
        % Michael Bergman
        % 
        % Revised September 2024
        %
        % Verified using Stata
        """
        #y = pandas.DataFrame(y).to_numpy()
        if y is None:
            y = pd.DataFrame(self.y_dataframe).to_numpy()
        else:
            y = y
        p=lags
        [T, K]=y.shape
        
        dy = y[1:len(y)]-y[0:len(y)-1]
        dep = dy[p-1:len(dy),0:K]
        
        if p>1:
            nlags = np.arange(0,p-1)
            indep = self.lagmat(dy, lags=nlags)
            indep = indep[p-2:len(indep)-1]
            
            if model==1:
                cointvec = np.dot(y[p-1:len(y)-1,:],beta.T)
                indep = np.concatenate((indep,cointvec), axis=1)
            elif model==2:
                cointvec = np.dot( np.concatenate((y[p-1:len(y)-1,:],np.ones((len(indep),1))), axis=1),beta.T)
                indep = np.concatenate((indep,cointvec), axis=1)
            elif model==3:
                cointvec = np.dot( np.concatenate((y[p-1:len(y)-1,:],np.ones((len(indep),1))), axis=1),beta.T)
                indep =  np.concatenate((np.concatenate((indep,cointvec), axis=1),np.ones((len(indep),1))), axis=1)
            elif model==4:
                help1 = np.concatenate((np.concatenate( (y[p-1:len(y)-1,:],np.ones((len(indep),1)) ), axis=1),np.arange(1,len(indep)+1).reshape(len(indep),1)), axis=1)
                cointvec = np.dot(help1,beta.T)
                indep =  np.concatenate((np.concatenate((indep,cointvec), axis=1),np.ones((len(indep),1))), axis=1)    
            elif model==5:
                help1 = np.concatenate((np.concatenate( (y[p-1:len(y)-1,:],np.ones((len(indep),1))), axis=1),np.arange(1,len(indep)+1).reshape(len(indep),1)), axis=1)
                cointvec = np.dot(help1,beta.T)
                indep = np.concatenate((np.concatenate((np.concatenate( (indep,cointvec), axis=1 ),np.ones((len(indep),1))), axis=1),np.arange(1,len(indep)+1).reshape(len(indep),1)), axis=1)
        elif p==1:
            if model==1:
                cointvec = np.dot(y[p-1:len(y)-1,:],beta.T)
                indep = cointvec
            elif model==2:
                cointvec = np.dot( np.concatenate((y[p-1:len(y)-1,:],np.ones((len(dep),1))), axis=1),beta.T)
                indep = cointvec
            elif model==3:
                cointvec = np.dot( np.concatenate((y[p-1:len(y)-1,:],np.ones((len(dep),1))), axis=1),beta.T)
                indep =  np.concatenate( (cointvec,np.ones((len(dep),1))), axis=1)    
            elif model==4:
                help1 = np.concatenate((np.concatenate( (y[p-1:len(y)-1,:],np.ones((len(dep),1)) ), axis=1),np.arange(1,len(dep)+1).reshape(len(dep),1)), axis=1)
                cointvec = np.dot(help1,beta.T)
                indep =  np.concatenate((cointvec,np.ones((len(dep),1))), axis=1)    
            elif model==5:
                help1 = np.concatenate((np.concatenate( (y[p-1:len(y)-1,:],np.ones((len(dep),1)) ), axis=1),np.arange(1,len(dep)+1).reshape(len(dep),1)), axis=1)
                cointvec = np.dot(help1,beta.T)
                help1 = np.concatenate((np.ones((len(dep),1)),np.arange(1,len(dep)+1).reshape(len(dep),1)), axis=1)
                indep =  np.concatenate((cointvec,help1), axis=1)    

        
        # =============================================================================
        [T2,Kp2]=indep.shape
        Beta = np.dot(np.linalg.inv(np.dot(indep.T,indep)),np.dot(indep.T,dep))
        res = dep-np.dot(indep,Beta)
        so = np.dot(res.T,res)/(T-1)
        so_ml = np.dot(res.T,res)/(T-1)
        # Compute t-ratios. Remember to sort Beta to match order of the diagonal of SEBeta
        SEBeta = np.sqrt(np.diag(np.kron(so,np.linalg.inv(np.dot(indep.T,indep))))).reshape(Kp2*K,1)
        Betavec = np.reshape(np.ravel(Beta, order='F'), [1,Kp2*K]).T
        tratioBeta = np.divide( Betavec, SEBeta)

        return Beta, Betavec, SEBeta, tratioBeta, res, so, so_ml

    def Initilize_beta_vecm(self,lags=int,model=int, beta_theory=np.ndarray, rank=int):# These are the unresticted estimates of beta and alpha
        rank = int(rank)
        beta = beta_theory
        _, K = self.y_dataframe.shape

        _, _, _, _, _, beta_est, alpha, _, c0, c1, _, d0, d1, _, _ = self.jcitest(lags-1,model=model)
        clear_output(wait=False)
        
        betaorg = beta[:, 0:rank]
        alphaorg = alpha[:, 0:rank]

        # Need to make sure that c0, c1, d0 and d1 are all arrays
        #print(c0)
        if len(c0)>1:
            c0 = np.reshape(np.array(c0), (K,1))
        else:
            c0 = c0.reshape(-1,1)
            
        if len(c1)>1:
            c1 = np.reshape(np.array(c1), (K,1))
        else:
            c1 = c1.reshape(-1,1)

        if len(d0)>1:
            d0 = np.reshape(np.array(d0), (K,1))
        else:
            d0 = d0.reshape(-1,1)

        if len(d1)>1:
            d1 = np.reshape(np.array(d1), (K,1))
        else:
            d1 = d1.reshape(-1,1)

        # Then we construct cointegration vector(s) depending on model and rank r
        alpha = alpha[:,0:rank]
        beta = beta[:,0:rank]

        if model==1:
            beta = beta
        elif model==2:
            beta = np.hstack((beta.T,c0[0:rank])).T
        elif model==3:
            beta = np.hstack((beta.T,c0[0:rank])).T
        elif model==4:
            beta = np.hstack((np.hstack((beta.T,c0[0:rank])),d0[0:rank])).T
        elif model==5:
            beta = np.hstack((np.hstack((beta.T,c0[0:rank])),d0[0:rank])).T

        return beta

    def lagmat(self,A: np.array, lags: list, orient: str = 'col') -> np.array:
        """
        Create array with time-lagged copies of the features/variables

        Parameters
        --------------
        - A      : (np.ndarray)          | Dataset. One column for each features/variables, and one row for each example/observation at a certain time step.
        - lags   : (ist)                 | Definition what time lags the copies of A should have.
        - orient : (str, Default: 'col') | Information if time series in A are in column-oriented or row-oriented

        Assumptions
        --------------
        - It's a time-homogenous time series (that's why there is no time index)
        - Features/Variables are ordered from the oldest example/observation (1st row) to the latest example (last row)
        - Any Missing Value treatment have been done previously.
        ***  
        > Copyright 2021 Ulf Hamster         
            
        """
        # detect negative lags
        if min(lags) < 0:
            raise Exception((
                "Negative lags are not allowed. Only provided integers "
                "greater equal 0 as list/tuple elements"))
        # None result for missing lags
        if len(lags) == 0:
            return None
        # enforce floating subtype
        if not np.issubdtype(A.dtype, np.floating):
            A = np.array(A, np.float32)

        if orient in ('row', 'rows'):
            # row-oriented time series
            if len(A.shape) == 1:
                A = A.reshape(1, -1)
            A = np.array(A, order='C')
            return self.lagmat_rows(A, lags)

        elif orient in ('col', 'cols', 'columns'):
            # column-oriented time series
            if len(A.shape) == 1:
                A = A.reshape(-1, 1)
            A = np.array(A, order='F')
            return self.lagmat_cols(A, lags)
        else:
            return None

    def jcitest(self,lags=int,model=str):
        """
        Test a multivariate time series for cointegration using the default values of the Johansen cointegration test.
        
        Input
        --------------
        - self.y_dataframe  | is a T x K data array
        - lags              | Number of lags in VEC model
        
        Models
        --------------
        - H2  (no deterministic terms):                                                    model = 1   deterministic="n"
        - H1* (constant outside the cointegration relation):                               model = 2   deterministic="co"
        - H1  (Constant within the cointegration relation):                                model = 3   deterministic="ci"
        - H*  (constant and linear trend in cointegration relation, no quadratic trend):   model = 4   deterministic="cili"
        - H:  (constant and linear trend in cointegration relation, quadratic trend)       model = 5   deterministic="colo"
        *(Here we use the same as in Matlab)*

        Return
        --------------
        - lr1     |   Trace test statistic
        - cval5   |   Critical value 5%
        - cval10  |   Critical value 10%
        - pval    |   p-value
        - l       |   Eigenvalue
        - beta    |   Cointegration vector
        - alpha   |   Speed of adjustment parameter
        - c0      |   Constant in cointegration vector
        - c1      |   Constant in first differences
        - d0      |   Linear trend in cointegration vector
        - d1      |   Linear trend in first differences

        """

        if model not in [1, 2, 3, 4, 5]:
            #print(self.jcitest.__doc__)  # Access the method's docstring directly
            raise ValueError("The parameter 'model' must be equal to 1, 2, 3, 4, or 5.")

        # Number of lags in underlying VAR
        p = lags + 1

        # First we need to load critical values
        JCV = np.load(f"{self.path}/JCV.npy")
        PSSCV = np.load(f"{self.path}/PSSCV.npy")
  
        if isinstance(self.y_dataframe, pd.DataFrame):
            x = self.y_dataframe.to_numpy()
        else:
            x = self.y_dataframe
        
        #x = pd.DataFrame(self.y_dataframe).to_numpy()
        
        [T, K] = x.shape
        #print(K)
        # Setting up first difference of x
        
        dx = x - self.lagmat(x, [1], 'col')
        dx = dx[1:len(dx)]
        Z0t = dx[p-1:len(dx)]
        
        lagsdx = np.arange(1,p)
        lagsdx = lagsdx.tolist()
        
        if p<=1:
            dxlags = []
            Z1t = []
            if model==3:
                Z1t = np.ones([len(Z0t),1])
            elif model==4:
                Z1t = np.ones([len(Z0t),1])
            elif model==5:
                Z1t = np.concatenate((np.ones([len(Z0t),1]),np.array([np.arange(1, len(Z0t)+1, 1)]).T), axis=1)
            
                
            #print('This is Z1t',Z1t)
            
        elif p>1:
            dxlags = self.lagmat(dx,lagsdx, 'col')
            dxlags = dxlags[p-1:len(x)]
            Z1t = dxlags
            [nrows, ncols] = dxlags.shape
            # Now we need to add deterministic components depending on model
            if model==3:
                Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
            elif model==4:
                Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
            elif model==5:
                Z1t = np.concatenate((dxlags,np.concatenate((np.ones([nrows,1]),np.array([np.arange(1, nrows+1, 1)]).T), axis=1)  ), axis=1)
            
        
        
        # Setting up lagged level
        Zkt = self.lagmat(x, [1], 'col')
        Zkt = Zkt[p:len(Zkt)]
                            
        
        # Add deterministic components to lagged level depending on model
        if model==2:
            Zkt = np.concatenate((Zkt,np.ones([len(Zkt),1])), axis=1)
        elif model==4:
            Zkt = np.concatenate((Zkt,np.array([np.arange(1, len(Zkt)+1, 1)]).T), axis=1)
        elif model==5:
            Zkt = Zkt
        
        # Ready to run the regressions and to compute the residuals
        # This is done in two steps using OLS
        
        if p>1:
            Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
            R0t = Z0t-np.dot(Z1t,Beta)
            Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
            R1t = Zkt-np.dot(Z1t,Beta)
        elif p<=1:
            R0t = Z0t
            R1t = Zkt
            if model>2:
                Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
                R0t = Z0t-np.dot(Z1t,Beta)
                Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
                R1t = Zkt-np.dot(Z1t,Beta)
                
            
        # Compute sum of squares
        S01 = np.dot(R0t.T,R1t)/len(Zkt)
        S10 = S01.T
        S00 = np.dot(R0t.T,R0t)/len(Zkt)
        S00I = np.linalg.inv(S00)
        S11 = np.dot(R1t.T,R1t)/len(Zkt)
        S11I = np.linalg.inv(S11)
        G = np.linalg.inv(sqrtm(S11))
        A = np.dot(np.dot(np.dot(np.dot(G,S10),S00I),S01),G.T)  
        
        # Compute eigenvalues and eigenvectors

        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # ordering eigenvalues and eigenvectors
        index = np.argsort(eigenvalues)
        index = np.flipud(index)
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:,index]

        # Compute cointegration vector beta and adjustment coefficients alpha
        beta = np.dot(G,eigenvectors[:,0:K-0])
        alpha = np.dot(np.dot(S01,beta),np.linalg.inv(np.dot(np.dot(beta.T,S11),beta)))

        if model==1:
            Pi = np.dot(alpha,beta.T)
            c0 = np.empty(1, dtype=object)
            c1 = np.empty(1, dtype=object)
            d0 = np.empty(1, dtype=object)
            d1 = np.empty(1, dtype=object)
            c = np.empty(1, dtype=object)
            d = np.empty(1, dtype=object)
        elif model==2:
            c0 = beta[K,:]
            c1 = np.empty(1, dtype=object)
            d0 = np.empty(1, dtype=object)
            d1 = np.empty(1, dtype=object)
            c = np.empty(1, dtype=object)
            d = np.empty(1, dtype=object)
            beta = beta[0:K,:]
            Pi = np.dot(alpha,beta.T)
        if model==3:
            W = Z0t - np.dot(np.dot(Zkt,beta),alpha.T)
            P,h1,h2,h3 = np.linalg.lstsq(Z1t, W, rcond=None)
            P = P.T
            c = P[:,len(P.T)-1]
            c0,h1,h2,h3 = np.linalg.lstsq(alpha, c, rcond=None)
            c1 = c - np.dot(alpha,c0)
            d0 = np.empty(1, dtype=object)
            d1 = np.empty(1, dtype=object)
            d = np.empty(1, dtype=object)
            Pi = np.dot(alpha,beta.T)
        elif model==4:
            d0 = beta[K,:].T
            W = Z0t - np.dot(np.dot(Zkt,beta),alpha.T)
            P,h1,h2,h3 = np.linalg.lstsq(Z1t, W, rcond=None)
            P = P.T
            c = P[:,len(P.T)-1]
            #print(P[:,len(P.T)-1])
            c0,h1,h2,h3 = np.linalg.lstsq(alpha, c, rcond=None)
            c1 = c - np.dot(alpha,c0)
            d1 = np.empty(1, dtype=object)
            d = np.empty(1, dtype=object)
            beta = beta[0:K,:]
            Pi = np.dot(alpha,beta.T)
        elif model==5:
            W = Z0t - np.dot(np.dot(Zkt,beta),alpha.T)
            P,h1,h2,h3 = np.linalg.lstsq(Z1t, W, rcond=None)
            P = P.T
            c = P[:,len(P.T)-2]
            d = P[:,len(P.T)-1]
            c0,h1,h2,h3 = np.linalg.lstsq(alpha, c, rcond=None)
            c1 = c - np.dot(alpha,c0)
            d0,h1,h2,h3 = np.linalg.lstsq(alpha, d, rcond=None)
            d1 = d - np.dot(alpha,d0)
            beta = beta[0:K,:]
            Pi = np.dot(alpha,beta.T)

        
        # Formulate eigenvalue problem
        
        l = np.linalg.eigvals(A).T
        l = np.sort(l)[::-1]
        l = l[0:K]
        
        # Compute Trace test lr1
        
        lr1 = np.zeros([len(l),1])
        iota = np.ones(len(l))
        
        for i in range(0, len(l)):
            tmp = self.trimr(np.log(iota - l), i , 0)
            lr1[i] = -len(Zkt) * np.sum(tmp, 0)     
        
        
        # Now we need to add critical values
        
        testStat = np.flip(lr1)
        
        cval5 = np.flip(JCV[0:K,10,model-1,0])
        cval10 = np.flip(JCV[0:K,20,model-1,0])
        
        # Compute p-value using linear interpolation
        
        # Define significance levels
        siglevels = pd.read_excel(f"{self.path}/siglevels.xlsx")
        siglevels = siglevels[['siglevels']]
        siglevels = pd.DataFrame.to_numpy(siglevels).T
        
        # Then extract relevant critical values
        
        CVTable = JCV[:,:,model-1,0]
        xp = np.flip(CVTable[0:K,:])
    
        # Finally compute p-values using linear interpolation
        pval = np.zeros([K,1])
        
        for j in range(0, K):
            if lr1[j,0] >= xp[j,len(xp.T)-1]:
                pval[j,0] = siglevels[0,0]
            elif lr1[j,0] <= xp[j,0]:
                pval[j,0] = siglevels[0,len(siglevels)]
            else:          
                idx = self.bisection(xp[j,:],lr1[j,0])
                x1 = xp[j,idx]
                x2 = xp[j,idx+1]
                y1 = siglevels[0,idx]
                y2 = siglevels[0,idx+1]
                pval[j,0] = 1-(y1 + (lr1[j,0]-x1)*(y2-y1)/(x2-x1))
        
        # Print table with trace test results
        
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        
        #print('\n***************************')
        info_model_1='Johansen Cointegration test'
        info_model_2=f'Sample size: $N={len(Z1t)}$'
        info_model_3=f'Lags in VAR: $p={p}$'
        info_model_4=f'Lags in Vec: $p={p-1}$'
        info_model_5='Statistic:','Trace'

        # print(info_model_1)
        # print(info_model_2)
        # print(info_model_3)
        # print(info_model_4)
        # print(info_model_5)

        if model==1:
            info_model = 'Model: H2 ("n") [no deterministic terms]'
            #print(info_model)
        elif model==2:
            info_model = 'Model: H1* ("co") [constant in coint vec, no linear trend in levels]'
            #print(info_model)
        elif model==3:
            info_model = 'Model: H1 ("ci") [constant in coint vec and linear trend in levels]'
            #print(info_model)
        elif model==4:
            info_model = 'Model: H* ("cili") [constant and linear trend in coint vec, linear trend in levels]'
            #print(info_model)
        elif model==5:
            info_model = 'Model: H ("colo") [constant and linear trend in coint vec, quadratic trend in levels]'
            #print(info_model)
        
        diagres = np.zeros([1,6])
        # =============================================================================
        for j in range(0, K):
                data = [[ j , np.round(float(lr1[j]), 3), np.round(cval5[j], 3), np.round(cval10[j], 4), np.round(pval[j,0], 4), np.round(l[j], 4)]]
                #data = np.ones([1,5])
                diagres = np.concatenate((diagres,data), axis=0)
        
        diagres = diagres[1:K+1,:]
        # print('========================================================')
        # print (tabulate(diagres, headers=["r", "stat", "cVal5%", "cVal10%", "p-value", "EigVal"], numalign="right"))
        # print('========================================================')
        # print('')

        table_x = pd.DataFrame(diagres, columns=["r", "stat", "cVal5%", "cVal10%", "p-value", "EigVal"])
        infomation = [info_model_1,info_model_2,info_model_3,info_model_4,info_model_5,info_model]


        # Then we print ML estimates of beta, alpha and the deterministic terms
        # print('\n***************************')
        # print('ML estimates of beta.T, the cointegration vector')
        # print(tabulate(beta.T, numalign="right"))
        # print('ML estimates of alpha, the adjustment coefficients')
        # print(tabulate(alpha, numalign="right"))
        # print('ML estimates of Pi = alpha x beta.T')
        # print(tabulate(Pi, numalign="right"))
        # print('\n***************************')
        # print('Deterministic terms')
        # print('ML estimates of c0\n',c0)
        # print('ML estimates of c1\n',c1)
        # print('ML estimates of d0\n',d0)
        # print('ML estimates of d1\n',d1)
        
        
        
        return lr1, cval5, cval10, pval, l, beta, alpha, c, c0, c1, d, d0, d1, table_x,infomation

    def lagmat_rows(self,A: np.array, lags: list):
        # number of colums and lags
        n_rows, n_cols = A.shape
        n_lags = len(lags)
        # allocate memory
        B = np.empty(shape=(n_rows * n_lags, n_cols), order='C', dtype=A.dtype)
        B[:] = np.nan
        # Copy lagged columns of A into B
        for i, l in enumerate(lags):
            # target rows of B
            j = i * n_rows
            k = j + n_rows  # (i+1) * n_rows
            # number cols of A
            nc = n_cols - l
            # Copy
            B[j:k, l:] = A[:, :nc]
        return B

    def lagmat_cols(self,A: np.array, lags: list):
        # number of colums and lags
        n_rows, n_cols = A.shape
        n_lags = len(lags)
        # allocate memory
        B = np.empty(shape=(n_rows, n_cols * n_lags), order='F', dtype=A.dtype)
        B[:] = np.nan
        # Copy lagged columns of A into B
        for i, l in enumerate(lags):
            # target columns of B
            j = i * n_cols
            k = j + n_cols  # (i+1) * ncols
            # number rows of A
            nl = n_rows - l
            # Copy
            B[l:, j:k] = A[:nl, :]
        return B
    
    def Additional_VEC_estimates(self,p=int,model=int):
        """
        This function computes additional output when estimating VEC for unknown coint vector(s).
    
        Input
        ----------------------
        - self.y_dataframe | pd.dataframe : is a T x K data array
        - p                | (int)        : Number of lags in VEC model
        
        - Model: Here we use the same as in Matlab
            - H2  (no deterministic terms):                                                    model = 1   deterministic="n"
            - H1* (constant outside the cointegration relation):                               model = 2   deterministic="co"
            - H1  (Constant within the cointegration relation):                                model = 3   deterministic="ci"
            - H*  (constant and linear trend in cointegration relation, no quadratic trend):   model = 4   deterministic="cili"
            - H:  (constant and linear trend in cointegration relation, quadratic trend)       model = 5   deterministic="colo"
        
  
        Output:
        -----------------
        - Various matrices used when computing normalized coint vector(s)

        """

        import pandas
        from tabulate import tabulate
        from scipy.linalg import sqrtm
        
        # Number of lags in underlying VAR
        p = p+1
        
        # First we need to load critical values
        JCV = np.load(f"{self.path}/JCV.npy")
        PSSCV = np.load(f"{self.path}/PSSCV.npy")
        x = pandas.DataFrame(self.y_dataframe).to_numpy()
        
        [T, K] = x.shape
        
        # Setting up first difference of x
        
        dx = x - self.lagmat(x, [1], 'col')
        dx = dx[1:len(dx)]
        Z0t = dx[p-1:len(dx)]
        
        lagsdx = np.arange(1,p)
        lagsdx = lagsdx.tolist()
        
        if p<=1:
            dxlags = []
            Z1t = []
            if model==3:
                Z1t = np.ones([len(Z0t),1])
            elif model==4:
                Z1t = np.ones([len(Z0t),1])
            elif model==5:
                Z1t = np.concatenate((np.ones([len(Z0t),1]),np.array([np.arange(1, len(Z0t)+1, 1)]).T), axis=1)
            
                
            
        elif p>1:
            dxlags = self.lagmat(dx,lagsdx, 'col')
            dxlags = dxlags[p-1:len(x)]
            Z1t = dxlags
            [nrows, ncols] = dxlags.shape
            # Now we need to add deterministic components depending on model
            if model==3:
                Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
            elif model==4:
                Z1t = np.concatenate( (dxlags,np.ones([nrows,1])), axis=1)
            elif model==5:
                Z1t = np.concatenate((dxlags,np.concatenate((np.ones([nrows,1]),np.array([np.arange(1, nrows+1, 1)]).T), axis=1)  ), axis=1)
            
        
        
        # Setting up lagged level
        Zkt = self.lagmat(x, [1], 'col')
        Zkt = Zkt[p:len(Zkt)]
                            
        
        # Add deterministic components to lagged level depending on model
        if model==2:
            Zkt = np.concatenate((Zkt,np.ones([len(Zkt),1])), axis=1)
        elif model==4:
            Zkt = np.concatenate((Zkt,np.array([np.arange(1, len(Zkt)+1, 1)]).T), axis=1)
        elif model==5:
            Zkt = Zkt
        
        # Ready to run the regressions and to compute the residuals
        # This is done in two steps using OLS
        
        if p>1:
            Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
            R0t = Z0t-np.dot(Z1t,Beta)
            Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
            R1t = Zkt-np.dot(Z1t,Beta)
        elif p<=1:
            R0t = Z0t
            R1t = Zkt
            if model>2:
                Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Z0t))
                R0t = Z0t-np.dot(Z1t,Beta)
                Beta = np.dot(np.linalg.inv(np.dot(Z1t.T,Z1t)),np.dot(Z1t.T,Zkt))
                R1t = Zkt-np.dot(Z1t,Beta)
                
            
        # Compute sum of squares
        S01 = np.dot(R0t.T,R1t)/len(Zkt)
        S10 = S01.T
        S00 = np.dot(R0t.T,R0t)/len(Zkt)
        S00I = np.linalg.inv(S00)
        S11 = np.dot(R1t.T,R1t)/len(Zkt)
        S11I = np.linalg.inv(S11)
        G = np.linalg.inv(sqrtm(S11))
        A = np.dot(np.dot(np.dot(np.dot(G,S10),S00I),S01),G.T)  
        
        return S00,S01,S10,S11
 
    def Phillips_normalisation(self,K=int,p=int,T=int,model=int,r=int,beta=np.ndarray,S00=np.ndarray,S10=np.ndarray,S01=np.ndarray,S11=np.ndarray):
        """
        This function normalizes estimated cointegration vector using Phillips normalization and computes standard errors of the normalized cointegration vector as well as for the implied estimate of alpha.
        In addition, the function also provides an estimate of Pi=alpha*beta' including standard errors.
        
        Input
        -----------
        - data = DataFrame
        - K     : (int) | number of variables in the VAR model
        - p     : (int) | number of lags in underlying VAR model
        - model : (int) | type of model. We follow Matlab convention.
            - model = 1 corresponds to model H2
            - model = 2 corresponds to model H1*
            - model = 3 corresponds to model H1
            - model = 4 corresponds to model H*
            - model = 5 corresponds to model H
        - r         : (int) |number of cointegration vectors
        - S01       : Sum of squares from ML estimates of VEC model for unknown coint vector computed in VECMLHelp function.
        - S11       : Sum of squares from ML estimates of VEC model for unknown coint vector computed in VECMLHelp function.
        - Gammasum  : Sum of autoregressive parameters in estimated VEC model
        
        Output:
        -----------
        - beta_norm | PHillips normalized cointegration vector
        - alpha_hat | the implied estimate of alpha
        - V_beta    | standard error of beta_nrom
        - V_alpha   | standard error of alpha_hat
        - Pi        | alpha_hat*beta_norm.T
        - V_Pi      | standard error of Pi
        
        ***
        > Author: Michael Bergman
        > Verified using STATA September 2024 
        """
        # Normalize coint vector using Phillips normalization
        import pandas
        import pandas
        data = pandas.DataFrame(self.y_dataframe).to_numpy()
        
        
        # First we normalize beta
        
        if r==1:
            beta_hat = beta/beta[0,0]
        elif r>1:
            beta1 = beta[0:r,0:r]
            beta2 = beta[r:len(beta),0:r]
            beta_hat =  np.vstack((np.identity(r),np.dot(beta2,np.linalg.inv(beta1))))
            
        
        beta = beta.T  # Need to take transpose of coint vec


        # Compute number of parameters depending on the model 

        if model == 1:
            m1 = K
            m2 = K*(p-1)
            nparams = K*r+K*r+K*(K*(p-1))
        elif model == 2:
            m1 = K+1
            m2 = K*(p-1)
            nparams = K*r+(K+1)*r+K*(K*(p-1))
        elif model == 3:
            m1 = K
            m2 = K*(p-1)+1
            nparams = K*r+K*r+K*(K*(p-1)+1)
        elif model == 4:
            m1 = K+1
            m2 = K*(p-1)+1
            nparams = K*r+(K+1)*r+K*(K*(p-1)+1)
        elif model == 5:
            m1 = K
            m2 = K*(p-1)+2
            nparams = K*r+K*r+K*(K*(p-1)+2)


        # We now use some output from VECMLHelp function

        # Compute alpha_hat given normalized coint vector

        if model==1:
            alpha_hat = S01 @ beta_hat @ np.linalg.inv(beta_hat.T @ S11 @ beta_hat)
        elif model==2:
            alpha_hat = S01 @ beta_hat @ np.linalg.inv(beta_hat.T @ S11 @ beta_hat)
        elif model==3:
            alpha_hat = S01 @ beta_hat[0:K,0:r] @ np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
        elif model==4:
            beta_hat = np.vstack((np.vstack((beta_hat[0:K,0:r],beta_hat[K+1,0:r])),beta_hat[K,0:r] ))
            alpha_hat = S01 @ beta_hat[0:K+1,0:r] @ np.linalg.inv(beta_hat[0:K+1,0:r].T @ S11 @ beta_hat[0:K+1,0:r])
        elif model==5:
            alpha_hat = S01 @ beta_hat[0:K,0:r] @ np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
            
            
            

        beta_norm = beta_hat[0:K,0:r]       # This is the normalized coint vector

        # Compute some matrices used to construct standard errors of alpha and beta

        if model==1:
            SigmaB = np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
            Omega_hat = S00 - alpha_hat @ beta_hat[0:K,0:r].T @ S10



        if model==2:
            SigmaB = np.linalg.inv(beta_hat[0:K+1,0:r].T @ S11 @ beta_hat[0:K+1,0:r])
            Omega_hat = S00 - alpha_hat @ beta_hat[0:K+1,0:r].T @ S10



        if model==3:
            SigmaB = np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
            Omega_hat = S00 - alpha_hat @ beta_hat[0:K,0:r].T @ S10


        if model==4:
            SigmaB = np.linalg.inv(beta_hat[0:K+1,0:r].T @ S11 @ beta_hat[0:K+1,0:r])
            Omega_hat = S00 - alpha_hat @ beta_hat[0:K+1,0:r].T @ S10


        if model==5:
            SigmaB = np.linalg.inv(beta_hat[0:K,0:r].T @ S11 @ beta_hat[0:K,0:r])
            Omega_hat = S00 - alpha_hat @ beta_hat[0:K,0:r].T @ S10




        Hj = np.hstack((np.zeros((r,m1-r)).T,np.identity(m1-r))).T
        d = int((nparams-r*r)/K)

        # Compute standard erorr of coint vec
        a1 = np.array((1/(T-d)))
        a2 = np.kron(np.identity(r),Hj)
        a3 = np.linalg.inv(np.kron( alpha_hat.T @ np.linalg.inv(Omega_hat) @ alpha_hat, Hj.T @ S11 @ Hj))
        a4 = np.kron(np.identity(r),Hj).T
        # Note that we take the standard error here so V_beta is the estimated standard error
        V_beta =  np.sqrt(np.diagonal(a1 * (a2 @ a3 @ a4)))
        # Reshape V_beta to match number of parameters in the cointegration vector
        if model == 1:
            V_beta = np.reshape(V_beta,(r,K)).T
        elif model==2:
            V_beta = np.reshape(V_beta,(r,K+1)).T
        elif model==3:
            V_beta = np.vstack((np.reshape(V_beta,(r,K)).T,np.zeros((1,r))))
        elif model==4:
            V_beta = np.vstack((np.reshape(V_beta, (r,K+1)).T,np.zeros((1,r))))
        elif model==5:
            V_beta = np.vstack((np.reshape(V_beta,(r,K)).T,np.zeros((2,r))))



        # Compute standard error of alpha given by normalized coint vector 
        V_alpha = np.kron((1/(T-d))*Omega_hat,SigmaB)

        # Compute standard error of Pi=alpha*beta'

        V_Pi = (1/(T-d))*np.kron(Omega_hat,(beta_norm @ SigmaB @ beta_norm.T))

        print('\n')
        print('Phillips normalized cointegration vector(s)\n')
        print("Normalized cointegration vector")
        print(beta_hat)
        print("Standard errors beta_normalized")
        if model==1:
            print(V_beta)
        elif model==2:
            print(V_beta)
        elif model==3:
            print(V_beta)
        elif model==4:
            print(V_beta)
        elif model==5:
            print(V_beta)


        print("Alpha based on normalized beta")
        print(alpha_hat)
        print("standard errors alpha");
        print(np.reshape(np.sqrt(np.diag(V_alpha)),(K,r)))
        print("Estimate of Pi and standard errors")
        print( np.hstack((  np.reshape((alpha_hat @ beta_norm.T), (K*K,1)), np.reshape(( np.sqrt(np.diag(V_Pi))),(K*K,1))  ))   )   

        # Prepare output
        V_alpha_hat = np.reshape(np.sqrt(np.diag(V_alpha)),(K,r))
        Pi = np.reshape((alpha_hat @ beta_norm.T), (K*K,1))
        V_Pi = np.reshape(( np.sqrt(np.diag(V_Pi))),(K*K,1))
        
        return beta_hat,alpha_hat,V_beta,V_alpha,Pi,V_Pi

    def vectovar(self, Gamma=np.ndarray, Pi=np.ndarray):
        """
        This function converts VEC estimates into companion matrix for VAR in levels

        Michael Bergman
        Checked October 2018

        Input: Gamma = K x K(p-1) coefficient matrix
            Pi = alpha * beta' a K x K matrix
        Output: Kp x Kp Companion matrix B
        """
        import numpy as np

        K, Kp = Gamma.shape
        p = Kp // K  # Ensure p is an integer

        # Initialize B with eye(K) + Pi + first K columns of Gamma
        B = np.eye(K) + Pi + Gamma[:, :K]

        # Augment Gamma with K zero columns
        Gamma = np.hstack((Gamma, np.zeros((K, K))))

        i = 0
        j = K

        while j <= Kp:
            # Append new columns to B
            B = np.hstack((B, Gamma[:, K + i:K + j] - Gamma[:, i:j]))
            i += K
            j += K

        # Append rows to B to form the companion matrix
        B = np.vstack((B, np.hstack((np.eye(p * K), np.zeros((p * K, K))))))

        return B
    
    def trimr(self,x, front, end):

        if end > 0:

            return x[front:-end]

        else:

            return x[front:]

    def bisection(self,array,value):
        '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
        and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
        to indicate that ``value`` is out of range below and above respectively.'''
        n = len(array)
        if (value < array[0]):
            return -1
        elif (value > array[n-1]):
            return n
        jl = 0# Initialize lower
        ju = n-1# and upper limits.
        while (ju-jl > 1):# If we are not yet done,
            jm=(ju+jl) >> 1# compute a midpoint with a bitshift
            if (value >= array[jm]):
                jl=jm# and replace either the lower limit
            else:
                ju=jm# or the upper limit, as appropriate.
            # Repeat until the test condition is satisfied.
        if (value == array[0]):# edge cases at bottom
            return 0
        elif (value == array[n-1]):# and top
            return n-1
        else:
            return jl
        
    def johcontest(self,r=float,test=[str],cons=np.ndarray,model=int,lags=int,alpha=float, print_output=False):
        # Now run jcontest! Note that I have added nargout=5 as input to function.
        # This ensures that result now contains all output that we normally obtain
        # from Matlab.
        # Result will be a tuple containing [h,pValue,stat,cValue,mles] where
        # mles is a structure
        # Start MATLAB engine
        # Convert Python types to MATLAB types
        """
        R: rank (r) where p-value <= threshold. Returns 0.0 if no ranks meet this.

        """
        r = float(r)
        
        eng = matlab.engine.start_matlab()
        
        if test[0] == 'acon':
            constrtype = '\u03B1'
        elif test[0] == 'bcon':
            constrtype = '\u03B2'
        elif test[0] == 'bvec':
            constrtype = '\u03B2'
        elif test[0] == 'avec':
            constrtype = '\u03B1'
        
        # data = matlab.double(Y.to_numpy())
        
        if isinstance(self.y_dataframe, pd.DataFrame):
            data = matlab.double(self.y_dataframe.to_numpy())
        else:
            data = matlab.double(self.y_dataframe)
        
        result = eng.jcontest(data, r, test, cons, 'Model', model, 'Lags', lags, 'Alpha', alpha,nargout=5)
        
        # We now have all the results from jcontest in the tuple result
        # First we extract the LR test statistic and the p-value directly from
        # result
        
        LRtest = result[2]
        pval = result[1]
        
        # Degrees of freedom is hidden in the dictionary in result (it's the
        # 4 element in result).
        # Extract the mles structure from result
        
        res = result[4]
        dof = res['dof']
        
        # Now we print the results
        
        
        if print_output == True:
            print('=======================================================================')
            print('Testing restrictions on',constrtype,'in VEC model with cointegration rank = ',int(r),'\n')
            print('The constraint on',constrtype,'tested is\n',cons.T,'\n')
            print('\nLR test of restriction with',int(dof),'degrees of freedom')
            print('Lr test statistic =',np.round(LRtest, 4))
            print('with p-value = ',np.round(pval, 4),'\n')
            
        # Then we need to print the restricted estimates of alpha and beta.
        # First we need to extract the appropriate dictionary from results,
        # the dictionary we extracted above to obtain degrees of freedom.
        # From this dictionary we extract the appropriate lists, define
        # the arrays we are interested in and then we print these arrays.
        
        mylist = list(res.values())[1]
        Arest = np.array(mylist['A'][:])
        Brest = np.array(mylist['B'][:])
        
        diagres = np.round(np.concatenate((Arest,Brest), axis=1), 4)
        tt1 = ["\u03B1"+str(i+1) for i in range(int(r))]
        tt2 = ["\u03B2"+str(i+1) for i in range(int(r))]
        header = tt1+tt2
        
        if print_output == True:
            print('Restricted estimates of alpha and beta')
            print(tabulate(diagres, headers=header, numalign="right"))
            
        # Restricted estimates of deterministic components
        
        if model=='H2':
            c0 = [[np.nan]]
            c1 = [[np.nan]]
            d0 = [[np.nan]]
            d1 = [[np.nan]]
        elif model=='H1*':
            c1 = [[np.nan]]
            d0 = [[np.nan]]
            d1 = [[np.nan]]
            if r == 1:
                c0 = np.reshape(np.asarray(mylist['c0']), (-1,1))
            else:
                c0 = np.asarray(mylist['c0'][:])
        
        elif model=='H1':
            if r == 1:
                c0 = np.reshape(np.asarray(mylist['c0']), (-1,1))
            else:
                c0 = np.asarray(mylist['c0'][:]) 
                
            c1 = np.array(mylist['c1'][:])
            d0 = [[np.nan]]
            d1 = [[np.nan]]
        elif model=='H*':
            d1 = [[np.nan]]
            if r == 1:
                c0 = np.reshape(np.asarray(mylist['c0']), (-1,1))
                d0 = np.reshape(np.asarray(mylist['d0']), (-1,1))
            else:        
                c0 = np.asarray(mylist['c0'][:])
                d0 = np.asarray(mylist['d0'][:])
                
            c1 = np.array(mylist['c1'][:])
        elif model=='H':
            if r == 1:
                c0 = np.reshape(np.asarray(mylist['c0']), (-1,1))
                d0 = np.reshape(np.asarray(mylist['d0']), (-1.1))
            else:        
                c0 = np.asarray(mylist['c0'][:])
                d0 = np.asarray(mylist['d0'][:])
                
            c1 = np.array(mylist['c1'][:])
            d1 = np.array(mylist['d1'][:])
        
        
        if print_output == True:
            print('\n Restricted estimates of deterministic terms \n')
            print(tabulate(c0, headers=["c0"],numalign="right"))
            print("")
            print(tabulate(c1, headers=["c1"],numalign="right"))
            print("")
            print(tabulate(d0, headers=["d0"],numalign="right"))
            print("")
            print(tabulate(d1, headers=["d1"],numalign="right"))
            print('=======================================================================')
            
        # This is all we need so we can now close the Matlab engine
        #eng.quit()
        
        return LRtest, pval, dof, Arest, Brest, c0, c1, d0, d1

    def Trace_Test(self,lags=int, p_value_threshold=0.05, LaTeX=False):
        X = {}

        T,K = self.y_dataframe.shape

        info = []
        highest_rank = []
   
        for model in range(5):
            X[model] = []
            _, _, _, _, _, _, _, _, _, _, _, _, _, table_jci, info_jci = self.jcitest(lags-1,model=model+1)

            temp = (info_jci[-1].split(":")[1].split('")')[0] + '').replace(" (","/").replace(" ","")
            info.append(f"\\texttt{{{model+1}/{temp}}}")
            highest_rank.append(int(self.Pantula_Principle(table=table_jci, p_value_threshold=p_value_threshold, display_data = False)))

            for k in range(K):
                X[model].append(f'$\\underset{{({table_jci["EigVal"][k]})}}{{{table_jci["stat"][k]}^{{{Initial_Tools.format_p_value(self,table_jci["p-value"][k])}}}}}$')

        df = pd.DataFrame({1: X[0],2: X[1],3: X[2],4: X[3],5: X[4]}, index=range(0, K))
        df.loc["Pantula"] = highest_rank
        df.columns = info

        latex_table = df.to_latex(index=True, escape=False,column_format='cccccc')
        latex_table = latex_table.replace('\\toprule\n &','\\toprule\n Rank ($r$) &')
        latex_table = latex_table.replace('"','').replace("\nPantula","\midrule\nPantula")
        latex_table = latex_table.replace('\end{tabular}',f'\end{{tabular}}\n\captionsetup{{width=0.8\linewidth}}\n\caption*{{Sample: {T-lags}. Lag length: $p={lags-1}$. Test statistic with Eigenvalue in parentheses and the significance level as a superscript (10\%, 5\% and 1\%). Pantula principle at a {int(p_value_threshold*100)}\%.}}')

        if self.LaTeX_path != None:
            latex_name = os.path.join(self.LaTeX_path, "Table", f"full_johansen_test_model.tex")
            os.makedirs(os.path.dirname(latex_name), exist_ok=True)
            with open(latex_name, "w") as fh:
                fh.write(latex_table)
            print("Saved to LaTeX Path","\n")
        
        if LaTeX == True:
            print(latex_table)
    

        def parse_latex_cell(cell):
            if not isinstance(cell, str):
                return cell 
            match = re.search(r"\((.*?)\).*?\{(.*?)(\^\{(.*?)\})?\}", cell)
            if match:
                eigen = match.group(1)
                stat = match.group(2)
                stars = match.group(4) if match.group(4) else ""
                return f"{stat}{stars} (λ={eigen})".strip()
            else:
                return cell  

        df_readable = df.copy()
        for row in df_readable.index:
            for col in df_readable.columns:
                df_readable.loc[row, col] = parse_latex_cell(df_readable.loc[row, col])

        df_readable.columns = df_readable.columns.to_series().apply(    lambda x: re.sub(r'\\texttt\{(.*?)\}', r'\1', x).replace('"', ''))
        
        display(df_readable)
        print(f'Sample: {T-lags}. Lag length: VAR: p={lags} and VEC: p={lags-1}. Test statistic with Eigenvalue\nin parentheses and the significance level as a superscript (10%, 5% and 1%).\nPantula principle at a {int(p_value_threshold*100)}%')
        
        return df_readable

    def Pantula_Principle(self,table=None, p_value_threshold=0.05, display_data=False):
        """
        Calculate the largest rank based on the Pantula principle.

        Determines the largest rank where the p-value is less than or 
        equal to the specified threshold.

        Parameters:
        ----------
        table : pandas.DataFrame, optional
            DataFrame with a 'p-value' column. Returns 0.0 if not provided.
        threshold : float, optional
            P-value threshold (default is 0.05).

        Returns:
        -------
        float
            Largest rank (r) where p-value <= threshold. Returns 0.0 if no ranks meet this.

        Example:
        --------
        >>> largest_r = self.pantula_principle(table_x)
        """

        largest_r = 0.0

        # Pantula principle: Find the largest rank where p-value is below the threshold
        if not table[table['p-value'] < p_value_threshold].empty:
            for i in range(len(table)):
                if table['p-value'].iloc[i] <= p_value_threshold:
                    largest_r = largest_r + 1.0
                else:
                    break
        
        if display_data == True:
            print("Highest rank",largest_r)

        return largest_r

    def Cointegration_Vectors(self,preferred_model=int, lags=int, p_value_threshold=0.05, normalise_on_element=None, rank=None, LaTeX=True):
        K = self.y_dataframe.shape[1]
        if normalise_on_element is not None and len(normalise_on_element) == 1:
            normalise_on_element = normalise_on_element * K  

        results = []
        highest_rank = {}
        cointegration_vectors = []

        for model in range(0, 5):
            lr1, cval5, cval10, pval, l, beta, alpha, c, c0, c1, d, d0, d1, table, information = self.jcitest(lags-1, model=model+1)
            results.append({"model": model,"lr1": lr1, "cval5": cval5, "cval10": cval10, "pval": pval, "l": l,"beta": beta, "alpha": alpha, "c": c, "c0": c0, "c1": c1,"d": d, "d0": d0, "d1": d1, "table": table, "information": information})
            highest_rank[model] = self.Pantula_Principle(table=table, p_value_threshold=p_value_threshold)
        
        preferred_model -= 1 
        print("Model:", preferred_model+1," | ",results[preferred_model]["information"][-1],"\n")

        if rank is None:
            rank = int(highest_rank[preferred_model])
            print("  Rank determined automatically using the Pantula principle")
        else:
            print("\n******* Cointegration rank set manually (not determined via Pantula principle) *******\n")

        if int(rank) > 0:
            for j in range(min(int(rank) + 1, K)):
                print("     Rank:", j, 
                    "LR:", results[preferred_model]["table"]["stat"].iloc[j],
                    "p-value:", results[preferred_model]["table"]["p-value"].iloc[j],
                    "(",Initial_Tools.format_p_value(self,results[preferred_model]["table"]["p-value"].iloc[j]),")",
                    "and λ =", results[preferred_model]["table"]["EigVal"].iloc[j])
            
            for j in range(int(rank)):
                vec = results[preferred_model]["beta"].T[j]

                if normalise_on_element is not None:
                    cointegration_vectors.append(vec / vec[normalise_on_element[j]] )
                else:
                    cointegration_vectors.append(vec)

            if LaTeX == True:
                for vec in cointegration_vectors:
                    print("\n",Initial_Tools.To_Matrix(self,[vec]).replace("}", "} "))
            
            return cointegration_vectors, int(rank)
        else:
            print(f"\nThe Johansen test indicates no cointegrating relationships (rank = 0)\nat the {p_value_threshold*100:.2f}% significance level.")
            return None, None

    def null(self, a, rtol=1e-5):
        u, s, v = np.linalg.svd(a)
        rank = (s > rtol*s[0]).sum()
        return rank, v[rank:].T.copy()
    
    def zero_array(self,K=int):
        """
        Generates a list of transposed unit vectors of size K.
        
        Parameters:
            K : (int) | The number of dimensions for the unit vectors.
        
        Returns:
            cons : (list) | A list containing transposed unit vectors.
        """

        cons = []
        for i in range(K):
            vec = np.zeros((1, K), dtype=np.float64)  # Create a 1xK zero vector
            vec[0, i] = 1  # Set the i-th position to 1
            cons.append(np.transpose(vec))  # Transpose and add to the list
        return np.array(cons)

    def Stationary_Exclusion_Exogeneity(self, rank=float, lags=int, model_johcontest="H1",LaTeX=True):
        """
        Perform cointegration hypothesis tests and generate a LaTeX table of the results.

        Parameters:
        - y_clean_data: The cleaned data for cointegration analysis.
        - rank: The rank to be used in the tests.
        - model_johcontest: The model specification for the Johansen test.
        - p: Number of lags in the model. not that the Johansen test takes float(p-1)

        Returns:
        - merged_df: A DataFrame containing the results for display.
        """
        rank = float(rank)
        K = len(self.list_of_info_latex)  # Assuming K is determined by the length of list_of_info_latex

        # Initialize containers for results
        Stationarity_test, Exclusion_test, Weak_exogeneity_test = [], [], []
        Stationarity_test_pval, Exclusion_test_pval, Weak_exogeneity_test_pval = [], [], []

        # Specify the constant array based on model
        cons = self.zero_array(K)  
        cons_1 = self.zero_array(K + 1)
        
        ####################################################################
        ########### Stationarity tests
        ####################################################################
        for i in range(K):
            if model_johcontest in ["H1*", "H*"]:  # H1* for trend in the cointegration vector and H* for no trend
                LRtest, pval, _, _, _, _, _, _, _ = \
                    self.johcontest(r=rank, test=['bvec'], cons=cons_1[i], 
                                    model=model_johcontest, lags=float(lags-1), alpha=0.05)
            else:
                LRtest, pval, _, _, _, _, _, _, _ = \
                    self.johcontest(r=rank, test=['bvec'], cons=cons[i], 
                                    model=model_johcontest, lags=float(lags-1), alpha=0.05)

            beta_i = f'{self.list_of_info_latex[i]}'
            temp = fr"${round(LRtest, 3)}^{{{Initial_Tools.format_p_value(self,pval)}}}$"
            Stationarity_test_pval.append(pval)
            Stationarity_test.append((beta_i, temp))

        Stationarity_df = pd.DataFrame(Stationarity_test, columns=[r'', r'\sc Stationarity'])
        
        ####################################################################
        ########### Exclusion tests
        ####################################################################
        for i in range(K):
            if model_johcontest in ["H1*", "H*"]:  # H1* H* for trend in the cointegration vector
                LRtest, pval, _, _, _, _, _, _, _ = \
                    self.johcontest(r=rank, test=['bcon'], cons=cons_1[i], 
                                    model=model_johcontest, lags=float(lags-1), alpha=0.05)
            else:
                LRtest, pval, _, _, _, _, _, _, _ = \
                    self.johcontest(r=rank, test=['bcon'], cons=cons[i], 
                                    model=model_johcontest, lags=float(lags-1), alpha=0.05)

            beta_i = f'{self.list_of_info_latex[i]}'
            temp = fr"${round(LRtest, 3)}^{{{Initial_Tools.format_p_value(self,pval)}}}$"
            Exclusion_test_pval.append(pval)
            Exclusion_test.append((beta_i, temp))

        Exclusion_df = pd.DataFrame(Exclusion_test, columns=[r'$\beta$', r'\sc Exclusion'])

        ####################################################################
        ########### Weak exogeneity tests
        ####################################################################
        for i in range(K):
            LRtest, pval, _, _, _, _, _, _, _ = \
                self.johcontest(r=rank, test=['acon'], cons=cons[i], 
                                model=model_johcontest, lags=float(lags-1), alpha=0.05)

            alpha_i = f'{self.list_of_info_latex[i]}'
            temp = fr"${round(LRtest, 3)}^{{{Initial_Tools.format_p_value(self,pval)}}}$"
            Weak_exogeneity_test_pval.append(pval)
            Weak_exogeneity_test.append((alpha_i, temp))

        Weak_exogeneity_df = pd.DataFrame(Weak_exogeneity_test, columns=[r'$\alpha$', r'\sc Weak Exogeneity'])


        empty_column = pd.DataFrame([''] * len(Stationarity_df), columns=[''])
        
        # Merge results into a final DataFrame for LaTeX output
        merged_df = pd.concat([Stationarity_df[""], empty_column, 
                                Stationarity_df["\sc Stationarity"], empty_column, 
                                Exclusion_df["\sc Exclusion"], empty_column, 
                                Weak_exogeneity_df["\sc Weak Exogeneity"]], axis=1)

        def clean(s):
            if isinstance(s, str):
                s = re.sub(r'\\sc\s*', '', s)           # Remove \sc og spaces
                s = re.sub(r'\$+', '', s)               # Remove $...$
                s = re.sub(r'\^\{(.*?)\}', r'\1', s)    # Remove ^{...}
                s = re.sub(r'\\([a-zA-Z]+)', r'\1', s)  # Remove e.g. \pi
            return s

        df_cleaned = merged_df.copy()
        df_cleaned[df_cleaned.columns[0]] = df_cleaned[df_cleaned.columns[0]].replace(dict(zip(self.list_of_info_latex, self.list_of_info[1:])))
        df_cleaned = df_cleaned.map(clean)
        df_cleaned.columns = [clean(col) for col in df_cleaned.columns]

        # Convert DataFrame to LaTeX and save
        Latex_table = merged_df.to_latex(index=False, column_format='lcccccccc')
        
        table_name = "Stationarity_Exclusion_Exogeneity"
        
        if self.LaTeX_path != None:
            latex_name = fr"{self.LaTeX_path}/Table/{table_name}.tex"
            os.makedirs(os.path.dirname(latex_name), exist_ok=True)
            with open(latex_name, "w") as fh:
                fh.write(Latex_table)
            print("Saved to LaTeX Path","\n")

        if LaTeX == True:
            display(Latex_table)
            formatted_pvals = [[f"{x:.7f}" for x in test] for test in  [Stationarity_test_pval, Exclusion_test_pval, Weak_exogeneity_test_pval]]
            print("\np-values:\n")
            print("   Stationarity", " ".join(formatted_pvals[0]))        
            print("      Exclusion", " ".join(formatted_pvals[1]))
            print("Weak_exogeneity", " ".join(formatted_pvals[2]))

        return df_cleaned.style.hide(axis="index")
    
class Identification_Restrictions_Cointegrated_SVAR(BaseClass):

    def __init__(self,Base_initialize, y_dataframe, horizon=int, lags=int, Constant=bool, Trend=bool, Exogenous=None):        
        """
        - y_dataframe (Y)     | dataframe a (T, K) dataframe
        - Base_initialize     | An instance of BaseClass containing data attributes such as LaTeX path, lists of definitions, and the path.
        - lags (p)            | Number of lags
        - Constant            | Constant bool (True or False)
        - Trend               | Trend bool (True or False)
        - Exogenous           | Exogenous regressors np.array or None
        """
        self.y_dataframe = y_dataframe
        self.horizon = horizon
        self.lags = lags
        self.Constant = Constant
        self.Trend = Trend
        self.Exogenous = Exogenous
        self.Python_path = Base_initialize.Python_path
        self.LaTeX_path = Base_initialize.LaTeX_path
        self.list_of_info_latex = Base_initialize.list_of_info_latex
        self.list_of_info = Base_initialize.list_of_info
        self.path = Base_initialize.path

        # Create an instance of Diagnostic_Testing and use y_dataframe as above
        self.diagnostic_tester = Diagnostic_Testing(Base_initialize,y_dataframe)  
        self.cointegration_tester = Testing_Cointegration(Base_initialize,y_dataframe)  

        self.K = len(y_dataframe.columns)
            
        Eigenvalues_est = np.max(np.abs(self.diagnostic_tester.Eigenvalues(lags=self.lags,Constant=self.Constant,Trend=self.Trend)))

        if Eigenvalues_est >= 1:
            print(f"The estimated VAR model may be unstable due to largest absolute eigenvalue is greater than or equal to 1. \nReview the model specification for potential adjustments.\nMax Eigenvalue = {round(Eigenvalues_est,3)}\n")        
            #raise NotImplementedError()
        else:
            print("The estimated VAR model is stable. The largest absolute eigenvalue is:", round(Eigenvalues_est,4))

    def Jmatrix(self):
        """
        Compute J matrix to use when computing IRF's
        Michael Bergman 2023
        """
        K = self.K 

        if self.lags == 1:
            Jmat = np.identity(K, dtype=int)
        else:   
            size = (K, (self.lags-1)*K)
            Jmat = np.concatenate((np.identity(K, dtype=int),np.zeros(size, dtype=int)), axis=1)
        
        return Jmat

    def MArep(self,A=np.ndarray):
        """
        Process to invert VAR to VMA
        
        Note: A is the companion matrix
        K = number of variables
        p = number of lags
        horizon = horizon to compute MA representation

        Output: MA representation organized in the same way as IRF
        """    
        jmat = self.Jmatrix(self.K,self.lags)
        ma = np.dot(np.dot(jmat,np.linalg.matrix_power(self.K,0)),jmat.T)
        ma = np.reshape(ma, [1,self.K * self.K])
        i=1
        while i <= self.horizon:
            marep = np.dot(np.dot(jmat,np.linalg.matrix_power(A,i)),jmat.T)
            ma = np.concatenate((ma,np.reshape(marep, [1,self.K * self.K ])), axis=0)
            i=i+1
        
        return ma

    def VECMknown(self, beta=np.ndarray,y=None,t_ratios=False, Micheal_bergman=False):
        
        if y is None:
            y = self.y_dataframe.to_numpy()
        else:
            y = y

        t, K = y.shape
        ydif = np.diff(y, axis=0)

        y = y.T
        ydif = ydif.T

        dy = ydif[:, self.lags-1:t]

        X = np.ones((t - self.lags, 1))  
        
        #X = np.hstack([X, np.arange(1, len(X) + 1).reshape(-1, 1)])
        
        for i in range(1, self.lags):            
            X = np.hstack([X, ydif[:,self.lags-i-1:t - i-1].T])

        y = y[:, self.lags-1:t-1] 
        R0t = dy - np.dot(np.dot(dy, X), np.linalg.inv(np.dot(X.T, X))) @ X.T # Equation relates to (3.2.8)
        
        
        R1t = y - (y @  X @ np.linalg.inv(np.dot(X.T, X))) @ X.T

        S00 = R0t @ R0t.T / (t-self.lags)
        S11 = R1t @ R1t.T / (t-self.lags)
        S01 = R0t @ R1t.T / (t-self.lags)

        alpha = S01 @ beta @ np.linalg.inv(beta.T @ S11 @ beta)  # KL (3.2.10)

        Gamma = (dy - alpha @ beta.T @ y) @ X @ np.linalg.inv(X.T @ X)  # KL (3.2.7)

        

        u = dy - alpha @ beta.T @ y - Gamma @ X.T
        SigmaML = u @ u.T / (t-self.lags)

        # if t_ratios == True:
        #     _,rank = alpha.shape
        #     X = X.T

        #     temp_1 = beta.T @ y @ y.T @ beta
        #     temp_2 = beta.T @ y @ X.T
        #     temp_3 = X @ y.T @ beta
        #     temp_4 = X @ X.T

        #     top_row = np.hstack((temp_1,temp_2))
        #     bottom_row = np.hstack((temp_3,temp_4))

        #     helper_matrix = np.linalg.inv(np.vstack((top_row, bottom_row)))

        #     Kp = len(helper_matrix)
        #     print(Kp)
        #     SEAlphaGamma = kron(SigmaML, helper_matrix)

        #     SEBeta = np.sqrt(np.diag(np.kron(SigmaML,helper_matrix))).reshape(Kp*K,1)
        #     #Betavec = np.reshape(np.ravel(beta, order='F'), [1,Kp*K]).T
        #     #tratioBeta = np.divide(Betavec, SEBeta)

        #     # if Micheal_bergman == True:
        #     #     concatenated = np.concatenate((alpha[:, :], Gamma[:, :1], Gamma[:, 1:]), axis=1).reshape(Kp * K, 1)
        #     #     tratioBeta = concatenated.reshape(Kp * K, 1) / np.sqrt(np.diag(SEAlphaGamma)[:,None])
        #     #     #reshaped_tratioBeta = tratioBeta.reshape(Gamma[:, 0].shape[0], -1)
        #     #     #tratioBeta = np.reshape(np.ravel(tratioBeta, order='F'), [1,Kp*K]).T

        #     # else:
        #     #     concatenated = np.vstack((alpha.flatten(order='F').flatten().reshape(-1, 1), Gamma[:, :1],Gamma[:, 1:].flatten(order='F').reshape(-1, 1)))
        #     #     tratioBeta = concatenated.reshape(Kp * K, 1) / np.sqrt(np.diag(SEAlphaGamma)[:,None])
        #     #     reshaped_tratioBeta = tratioBeta.reshape(np.shape(np.hstack((Gamma[:, :1], alpha[:, :], Gamma[:, 1:])).T)).T
        #     #     #reshaped_tratioBeta.T[:rank].T


        #     return alpha, Gamma, SigmaML, u, SEBeta,SEAlphaGamma #reshaped_tratioBeta[:,0:rank]
        # else:

        return alpha, Gamma, SigmaML, u

    def vectovar(self, beta=np.ndarray, alpha=np.ndarray,Gamma=None):
        """
        This function converts VEC estimates into companion matrix for VAR in levels.
        
        Michael Bergman
        Checked October 2023
        
        Input: 
        - Gamma  =  K x Kp coefficient matrix
        - Pi     =  alpha*beta' a K x K matrix
        
        Output: Kp x Kp Companion matrix A
        """
        if Gamma is None:
            _, Gamma, _, _ = self.VECMknown(beta=beta)
            V = Gamma[:,0];                           # Constant
            Gamma = Gamma[:,1:self.K*(self.lags-1)+1] # first diff lags
        else:
            Gamma = Gamma

        # _, _, Gamma = self.Implement_identication(model = model, rank=rank, beta = beta)
        # vectovar(self, model=int,rank=int, beta=np.ndarray, alpha=np.ndarray):

        Pi = alpha @ beta.T

        if len(Gamma.T) == 0:
            (K,K) = Pi.shape
            A = np.identity(K) + Pi
        else:
            (K,Kp)=Gamma.shape
            p = Kp/K
            A = np.identity(K) + Pi + Gamma[:,0:K]
            Gamma = np.hstack([Gamma,np.zeros([K,K],dtype=np.float64)])
            
            i=0
            j=K
            while j <= Kp:
                A = np.hstack([A,(Gamma[:,i+K:j+K]-Gamma[:,i:j])])
                i = i+K
                j=j+K
            
            A = np.vstack([A,np.hstack([np.identity(K*int(p)),np.zeros([K*int(p),K])])])
        
        return A

    def IRF(self,B0inv=None,VEC=bool,VAR=bool, beta=np.ndarray, alpha=np.ndarray, A=None):
        """
        Process to invert VAR to VMA
        
        Note: A is the companion matrix
        K = number of variables
        p = number of lags
        horizon = horizon to compute MA representation
        B0inv = Identifying matrix

        Output: IRF's are organized such that the first K columns contain
        the effects of all shocks on the first variable, next K columns contain the
        effects of all shocks on the second variable and so on
        """
        
        if A is None:
            Beta, _, _, _, _, SIGMA = self.diagnostic_tester.VAR_estimation_with_exogenous(lags=self.lags,Constant=self.Constant,Trend=self.Trend,Exogenous=self.Exogenous)
            
            if VEC == True:
                A = self.vectovar(beta=beta, alpha=alpha)
            elif VAR == True:
                A = self.diagnostic_tester.Companion_matrix(Beta=Beta,lags=self.lags)
            else:
                print("Specifi to use the VAR or the VEC estimate for the Companion matrix A")

            if B0inv is None:  
                B0inv = np.linalg.cholesky(SIGMA)

        jmat = self.Jmatrix()
        irf = np.dot(np.dot(np.dot(jmat,np.linalg.matrix_power(A,0)),jmat.T),B0inv)
        irf = np.reshape(irf,[1, self.K  * self.K])
        
        i = 1
        while i <= self.horizon:
            help = np.dot(np.dot(np.dot(jmat,np.linalg.matrix_power(A,i)),jmat.T),B0inv)
            irf = np.concatenate((irf,np.reshape(help,[1, self.K * self.K])), axis=0)
            i = i + 1
        
        return irf

    def IRF_estimation(self, IRF_var=np.ndarray, list_of_info=list, normalise=None):
        result = {}

        list_in_use = list_of_info[1:] if "Dates" in list_of_info[0] else list_of_info

        for i in range(self.K):
            df_horizon = pd.DataFrame({'Horizon': range(self.horizon+1)})
            
            for j in range(self.K):
                df_horizon[list_in_use[j]] = IRF_var[:, i * self.K + j] 
            
            result[list_in_use[i]] = df_horizon.drop(columns=["Horizon"])

        if normalise is not None:  
            norm = abs(result[normalise][normalise][0])

        if normalise is not None:  
            for i in range(self.K):
                result[list_in_use[i]] = result[list_in_use[i]]/norm
        
        return result

    def FEVD(self,B0inv,VEC=bool,VAR=bool,beta=np.ndarray,alpha=np.ndarray, A=None):
        """
        Structural forecast error variance Decomposition
        using B0inv for a K dimensional VAR and horizon h 
        Input:
        A: companion matrix
        B0inv: Identifying matrix
        K: number of variables
        p: number of lags
        h: horizon 

        Output: fevd organized as
        Each row is for horizon 1, 2, ..., h
        First K columns contain effects of shock 1,
        next K columns contain effects of shock 2,
        ....
        last K columns contain effect of shock K
        Michael Bergman December 2018
        """

        if A is None:
            Beta, _, _, _, _, SIGMA = self.diagnostic_tester.VAR_estimation_with_exogenous(lags=self.lags,Constant=self.Constant,Trend=self.Trend,Exogenous=self.Exogenous)
            
            if VEC == True:
                A = self.vectovar(beta=beta, alpha=alpha)
            elif VAR == True:
                A = self.diagnostic_tester.Companion_matrix(Beta=Beta,lags=self.lags)
            else:
                print("Specifi to use the VAR or the VEC estimate for the Companion matrix A")
        
        if B0inv is None:  
            B0inv = np.linalg.cholesky(SIGMA)

        jmat = self.Jmatrix()
        Theta = np.dot(np.dot(np.dot(jmat,np.identity(self.K * self.lags, dtype=int)),jmat.T),B0inv) 
        
        # Need to take transpose such that rows=variables and cols=shocks   
        Theta = Theta.T
        # compute mse(1)
        Theta2 = np.multiply(Theta,Theta)
        Theta_sum = np.sum(Theta2, axis=0)
        VC = np.zeros([self.K ,self.K])
        i = 0
        while i<=self.K-1:
            VC[i,:]=Theta2[i,:] / Theta_sum
            i = i+1

        VC = np.reshape(VC,[1, self.K * self.K ] )
        # Then add the remaining horizons
        Thetah = Theta2
        j = 0
        while j <= self.horizon-1:
            j = j+1 
            Theta1 = np.dot(np.dot(   np.dot(jmat,np.linalg.matrix_power(A,j)) ,jmat.T),B0inv) 
            Theta1 = Theta1.T
            Theta2 = np.multiply(Theta1,Theta1)
            Thetah = np.array(Thetah)+np.array(Theta2)
            Thetah_sum = np.sum(Thetah, axis=0)
            VC1 = np.zeros([self.K , self.K])
            i = 0
            
            while i<=self.K-1:
                VC1[i,:]=Thetah[i,:] / Thetah_sum
                i = i+1

            VC1 = np.reshape(VC1,[1, self.K * self.K] )
            VC = np.concatenate((VC,np.reshape(VC1, [1,self.K *self.K])), axis=0)

        return VC

    def FEVD_IRF(self,irf):
        """
        Structural Forecast Error Variance Decomposition
        computed using Impulse Responses for dimension K and horizon h 
        Input:
        irf: Impulse Response Function
        K: number of variables
        h: horizon 

        Output: fevd organized as
        Each row is for horizon 1, 2, ..., h
        First K columns contain effects of shock 1,
        next K columns contain effects of shock 2,
        ....
        last K columns contain effect of shock K
        Michael Bergman December 2018
        """
        K = self.K
        Theta = np.reshape(irf[0,:], [K,K])
        Theta = Theta.T
        # compute mse(1)
        Theta2 = np.multiply(Theta,Theta)
        Theta_sum = np.sum(Theta2, axis=0)
        VC = np.zeros([K,K])
        i = 0
        
        while i <= K - 1:
            VC[i,:]=Theta2[i,:] / Theta_sum
            i = i + 1

        VC = np.reshape(VC,[1, K*K] )
        VChelp = np.zeros([K,K])
        
        # Then add the remaining horizons
        Thetah = Theta2
        j = 0

        while j <= self.horizon -1:
            j = j + 1 
            Theta1 = np.reshape(irf[j,:], [K,K]) 
            Theta1 = Theta1.T
            Theta2 = np.multiply(Theta1,Theta1)
            Thetah = np.array(Thetah)+np.array(Theta2)
            Thetah_sum = np.sum(Thetah, axis=0)
            i = 0
            
            while i <= K - 1:
                VChelp[i,:]=Thetah[i,:] / Thetah_sum
                i = i + 1

            VC1 = np.reshape(VChelp,[1, K*K] )
            VC = np.concatenate((VC,np.reshape(VC1, [ 1, K*K  ])), axis=0)
        #print("Use this function when the FEVD is based on the IRF from the VEC model.")

        return VC

    def FEVD_estimation(self, FEVD_var=np.ndarray, list_of_info=list, normalise=None):
        result = {}

        list_in_use = list_of_info[1:] if "Dates" in list_of_info[0] else list_of_info

        for i in range(self.K):
            df_horizon = pd.DataFrame({'Horizon': range(self.horizon+1)})
            
            for j in range(self.K):
                df_horizon[list_in_use[j]] = FEVD_var[:, i * self.K + j] 
            
            result[list_in_use[i]] = df_horizon.drop(columns=["Horizon"])

        # if normalise is not None:  
        #     for i in range(self.K):
        #         result[list_in_use[i]] = result[list_in_use[i]]/result[normalise][normalise][0]
        
        # print("FEVD_result[Shock][Variable of Interest]:\n")
        # print(" - The first specification indicates which shock will be shown, e.g., 'Interest Rate'.")
        # print(" - The second specification indicates which variable is impacted by the shock, e.g., 'Output (GDP)'.\n")
        # print("Thus, FEVD_result[Shock][Variable of Interest] refers to FEVD_result['Interest Rate']['Output (GDP)'].")
        # print("This shows the Forecast Error Variance Decomposition of how GDP reacts to an interest rate shock.")

        return result 

    def Implement_identication(self, model = int, rank=int, beta = np.ndarray, ML=True):
        
        if rank >= 1:
            beta_orginal = beta
            _, _, _, _, _, _, alpha, _, c0, _, _, d0, _, _, _ = self.cointegration_tester.jcitest(self.lags-1,model=model)
            clear_output(wait=False)
            rank = int(rank)

            betaorg  =  beta[:, 0:rank]
            alphaorg = alpha[:, 0:rank]

            if model == 1:
                beta = beta
            elif model == 2:
                beta = np.hstack((beta.T,c0[0:rank].reshape(-1, 1))).T
            elif model == 3:
                #beta = np.vstack((beta.T, c0[0:rank].reshape(1, -1)))
                beta = np.hstack((beta.T,c0[0:rank].reshape(-1, 1))).T
            elif model == 4:
                beta = np.hstack((np.hstack((beta.T,c0[0:rank].reshape(-1, 1))),d0[0:rank].reshape(-1, 1))).T
            elif model == 5:
                beta = np.hstack((np.hstack((beta.T,c0[0:rank].reshape(-1, 1))),d0[0:rank].reshape(-1, 1))).T

            Beta, _, _, _, _, SIGMA, _ = self.cointegration_tester.LSKnownBeta(lags=self.lags, beta=beta.T, model=model)

            if ML == True:
                _, _, SIGMA, _ = self.VECMknown(beta=beta_orginal)
            
            Beta = Beta.T

            GAMMA = Beta[:,0:self.K * (self.lags-1)]                # first diff lags
            alpha = Beta[:,self.K * (self.lags-1):len(Beta.T)-1]    # speed of adjustment


            Gamma = GAMMA[0:self.K ,0:self.K]

            if self.lags > 2:
                for i in range(1,self.lags - 1):
                    Gamma = Gamma+GAMMA[0:self.K ,i * self.K :i * self.K + self.K ]

            beta_perp = self.cointegration_tester.null(betaorg.T)
            beta_perp = beta_perp[1]
            alpha_perp = self.cointegration_tester.null(alpha.T)
            alpha_perp = alpha_perp[1]

            Xi = beta_perp @ np.linalg.inv(alpha_perp.T @ (np.identity(self.K )-Gamma) @ beta_perp) @ alpha_perp.T
        else:
        
            Beta, _, _, _, _, SIGMA = self.diagnostic_tester.VAR_estimation_with_exogenous(lags=self.lags,Constant=self.Constant,Trend=self.Trend,Exogenous=self.Exogenous)

            A = self.diagnostic_tester.Companion_matrix(Beta,lags=self.lags)

            Gamma = A[0:self.K, 0:self.K].copy()  
            
            for i in range(1, self.lags):
                Gamma = Gamma + A[0:self.K, i*self.K:(i+1)*self.K].copy()

            Xi = np.linalg.inv(np.eye(self.K) - Gamma)

        return Xi, SIGMA, Gamma

    def Extract_Elements(self,string_matrix=str):
        """
        Extracts the indices of non-star elements (`*?) from a string representation of a matrix.

        ### Parameters:
        - `string_matrix` (_str_) : A string representation of a lower triangular matrix where non-star elements (`*`) indicate positions to extract.

        ### Returns:
        - `elements` (_list of tuples_): A list of `(row, column)` indices for the non-star elements (`*`) in the given matrix.
        --------------
        #### Example:
        ```
        string_matrix = [ * 0 0 0
                          * * 0 0
                          * * 0 0
                          * * 0 0 ]

        >>> elements = [(0, 1), (0, 2), (0, 3),
                                (1, 2), (1, 3),
                                (2, 2), (2, 3),
                                (3, 2), (3, 3)]
        ```
        """
        # B= np.random.randn(self.K,self.K)
        
        elements = []
        rows = string_matrix.strip().splitlines()
        for i, row in enumerate(rows):
            values = re.split(r'\s+', row.strip(' []'))
            for j, value in enumerate(values):
                if value != '*':
                    elements.append((i, j))

        return elements

    def Restrictions(self, B0inv_flat=np.ndarray, B0inv_R=str, Upsilon_R=str, SIGMA=np.ndarray, Xi=np.ndarray, Beta=None):
        
        # Obtain positions of non-'*' elements from extract_elements
        positions_B = self.Extract_Elements(B0inv_R)
        positions_U = self.Extract_Elements(Upsilon_R)

        B0inv = B0inv_flat.reshape((self.K, self.K))
        
        # Compute Upsilon
        Upsilon = Xi @ B0inv

        if Beta is not None:
            F = vech(B0inv @ B0inv.T - SIGMA[:self.K, :self.K])
            G = vech(np.linalg.inv(B0inv) @ SIGMA @ np.linalg.inv(B0inv).T - np.eye(self.K))
            H = vech(Beta @ Xi @ B0inv)
            
            # Long run and short run restrictions using dynamic positions
            q = np.concatenate((F,G,H, *[[B0inv[i, j]] for i, j in positions_B],*[[Upsilon[i, j]] for i, j in positions_U]))
        else:
            F = vech(B0inv @ B0inv.T - SIGMA[:self.K, :self.K])
            
            # Long run and short run restrictions using dynamic positions
            q = np.concatenate((F, *[[B0inv[i, j]] for i, j in positions_B],*[[Upsilon[i, j]] for i, j in positions_U]))

        return q
    
    def VAR_Bootstrap(self,n_trials=int, Bootstrap = True, Bootstrap_type = "Wild", 
                        Orthogonal = True, iterations=1000, B0inv_R=str, Upsilon_R=str, initial_guess=None):

        t,K = self.y_dataframe.shape

        Beta, _, _, residuals, _, _ = self.diagnostic_tester.VAR_estimation_with_exogenous(lags=self.lags,Constant=self.Constant,Trend=self.Trend,Exogenous=self.Exogenous)
        u = residuals.T
        
        if self.Constant == True and self.Trend == True:
            V = Beta[len(Beta)-1-1:len(Beta),:].T
        elif self.Constant == True and self.Trend == False:
            V = Beta[len(Beta)-1-0:len(Beta),:].T
        elif self.Constant == False and self.Trend == True:
            V = Beta[len(Beta)-0-1:len(Beta),:].T
        else:
            V = Beta[len(Beta):len(Beta),:].T
            print("No constant nor trend")

        A = self.diagnostic_tester.Companion_matrix(Beta,lags=self.lags)
        a = A[0:K,:]

        IRFmat = np.zeros((n_trials,(K*K)*(self.horizon+1)), dtype=np.float64)
        VCmat = np.zeros((n_trials,(K*K)*(self.horizon+1)), dtype=np.float64)
        trial = fail = total_trials = failed_to_estimate = 0

        y = self.y_dataframe.to_numpy()

        while trial <= n_trials:

            if Bootstrap == False:
                y0p = y[:self.lags-1, :self.K-1]  # in Python, indexing starts from 0
            else:
                pos = random.randint(self.lags+1, t - 1)  # in Python, randint's upper limit is inclusive
                y0p = y[(pos-self.lags+1):(pos+1), :]  # add 1 to pos for Python's exclusive upper limit in slicing
                
            # Draw with replacement from uHat
            indexur = np.random.randint(0, t-self.lags, size=t-self.lags)  # Using numpy.random.randint! Generate Tbig
            u_r = u[:, indexur]
            u_r = np.hstack((np.zeros((K, self.lags)), u_r))  # Add residuals for initial conditions

            # Handle deterministic components: Only constant and linear trend allowed
            if self.Constant == True:
                determ = np.ones((t, 1))
            else:
                determ = np.array([]).reshape(0, 1)  # define an empty column vector
            
            if self.Trend == True:
                added_array = np.arange(-self.lags+1, t-self.lags+1).reshape(-1, 1)
                determ = np.column_stack((determ, added_array))
            
            i = self.lags
            j = 0
            y_r = np.zeros((self.K, t-self.lags))  # simulated y for t=1:Tbig
            y_r = np.hstack((y0p.T, y_r))  # Add initial values

            while i < t:   # Python uses 0-based indexing
                index = np.flip(np.arange(j, j+self.lags))
                ylags = self.cointegration_tester.Vectorization(y_r[:, index])
                ylags = np.reshape(ylags, (K,self.lags))
                ylags = self.cointegration_tester.Vectorization(ylags.T).reshape(-1,1)
                
                if Bootstrap_type == "Wild":
                    try:
                        if self.Constant == False and self.Trend == False:
                            y_r[:, i] = (a @ ylags + np.random.randn() * u_r[:, i].reshape(-1,1)).ravel()   # np.random.randn() generates a random number from a standard normal distribution
                        else:
                            y_r[:, i] = (V @ determ[i, :].reshape(-1, 1) + a @ ylags + np.random.randn() * u_r[:, i].reshape(-1,1)).ravel()   # np.random.randn() generates a random number from a standard normal distribution
                    except:
                        print(y_r.shape)
                        assert False
                else:
                    if self.Constant == False and self.Trend == False:
                        y_r[:, i] = (a @ ylags + u_r[:, i].reshape(-1,1)).ravel()
                    else:
                        y_r[:, i] = (V @ determ[i, :].reshape(-1, 1) + a @ ylags + u_r[:, i].reshape(-1,1)).ravel()

                i = i + 1
                j = j + 1
                
            Beta_r, _, _, _, _, SIGMA_r = self.diagnostic_tester.VAR_estimation_with_exogenous(lags=self.lags,Constant=True,Trend=False,Exogenous=None,y=y_r.T)
            A_r = self.diagnostic_tester.Companion_matrix(Beta_r,lags=self.lags)

            if Orthogonal == True:
                B0inv_r = np.linalg.cholesky(SIGMA_r)
            else:
                Asumr = A_r[0:K, 0:K].copy()  
                for i in range(1, self.lags):
                    Asumr = Asumr + A_r[0:K, i*K:(i+1)*K].copy()  

                Asumr = np.eye(K) - Asumr  

                Xi_r = np.linalg.inv(Asumr)

                if initial_guess is None:
                    result = least_squares(lambda B0inv_flat: self.Restrictions(B0inv_flat, B0inv_R=B0inv_R, Upsilon_R=Upsilon_R, Xi=Xi_r, SIGMA=SIGMA_r,Beta=None),
                                        np.random.randn(K * K))
                else:
                    result = least_squares(lambda B0inv_flat: self.Restrictions(B0inv_flat, B0inv_R=B0inv_R, Upsilon_R=Upsilon_R, Xi=Xi_r, SIGMA=SIGMA_r,Beta=None),
                                        initial_guess.flatten())
                    
                B0inv_r = result.x.reshape((K, K))

                for i in range(0,K):
                    if B0inv_r[i,i] < 0:
                        B0inv_r[:,i] = -B0inv_r[:,i]

            e = np.abs(np.linalg.eigvals(A_r))

            tol_atol = 1e-4

            if np.allclose(B0inv_r @ B0inv_r.T, SIGMA_r, atol=tol_atol) or np.allclose(np.linalg.inv(B0inv_r) @ SIGMA_r @ np.linalg.inv(B0inv_r).T, np.eye(K), atol=tol_atol):
                if np.max(e)>0.999999999:
                    fail = fail + 1
                    total_trials = total_trials + 1
                else:
                    irfr = self.IRF(A=A_r,B0inv=B0inv_r)
                    IRFmat[trial-1,:] = np.reshape(self.cointegration_tester.Vectorization(irfr.T), (1,-1))
                    
                    VCr = self.FEVD(A=A_r,B0inv=B0inv_r) * 100
                    VCmat[trial-1,:] = np.reshape(self.cointegration_tester.Vectorization(VCr.T), (1,-1))
                    
                    trial = trial + 1
                    total_trials = total_trials + 1
            else:
                failed_to_estimate = failed_to_estimate + 1
                total_trials = total_trials + 1
            
            print(f"Trial: {trial-1} of {total_trials-1}  |  Failures due to solver: {failed_to_estimate} | Failures due to unstable eigenvalue: {fail}",end="\r")

        Bootstrap_type_used = "Wild" if Bootstrap_type == "Wild" else "non parametric"

        # Display total number of trials, failures and trials used
        print('\nBootstrap based on:',n_trials,'trials')
        print('Bootstrap type in use:',Bootstrap_type_used)
        print('Unstable replications:',fail)

        return IRFmat, VCmat
    
    def VEC_Bootstrap(self, Bootstrap = True, Bootstrap_type = "Wild", Orthogonal = True, 
                       iterations = 1000, B0inv_R = str, Upsilon_R = str, rank = int,
                       n_trials=int, Beta_theory = np.ndarray, Umat = None, Upsilon0 = None,
                       initial_guess=None, estimator = "Numerical",
                       linear_combinations=list, atol_number=1e-9):
        
        print("Initial settings and model checks")
        Beta, _, _, _, _, SIGMA = self.diagnostic_tester.VAR_estimation_with_exogenous(lags=self.lags,Constant=self.Constant,Trend=self.Trend,Exogenous=self.Exogenous)

        rank = int(rank)
        alpha, Gamma, SIGMA, u = self.VECMknown(beta=Beta_theory.T)

        V = Gamma[:,0]
        GAMMA = Gamma[:,1:self.K*(self.lags-1)+1] # first diff lags

        beta_perp=self.cointegration_tester.null(Beta_theory)[1]
        alpha_perp=self.cointegration_tester.null(alpha.T)[1]

        Gamma_sum = GAMMA[0:self.K,0:self.K]
        if self.lags > 1: 
            for i in range(1,self.lags-1): 
                Gamma_sum = Gamma_sum+GAMMA[0:self.K,i*self.K:i*self.K+self.K]

        help = np.linalg.inv(np.dot(np.dot(alpha_perp.T,(np.identity(self.K)-Gamma_sum)),beta_perp))
        Xi = np.dot(np.dot(beta_perp,help),alpha_perp.T)

        errors = 0
        if np.sum(np.abs(Beta_theory @ Xi)>1e-12) != 0: 
            print("      Identification of Xi failed at test I")
            errors = errors + 1 
        if np.sum(np.abs(np.linalg.eig(Xi.T @ Xi)[0])>1e-12) != self.K-rank: 
            print("      Identification of Xi failed at test II")
            errors = errors + 1 
        if errors == 0:
            print("      Xi satisfies assumptions")
        else:
            raise KeyError('Xi does not satisfies assumptions')

        if Umat is None:
            Umat=np.zeros((rank,self.K))
            for i in range(0, rank):
                Umat[i,self.K-i-1] = 1
                i = i + 1

        if np.isclose(np.linalg.det(np.dot(Umat, alpha)), 0, atol=1e-6):
            print('\nIdentification of transitory shock is NOT valid')
            raise KeyError('The determinant of U and Alpha is zero, identification failed!')
        else:
            print('      Identification of transitory shocks is valid')

        if initial_guess is None:
            initial_guess = np.random.randn(self.K * self.K)
        else:
            initial_guess = initial_guess.flatten()

        result = least_squares(lambda B0inv_flat: self.Restrictions(B0inv_flat, B0inv_R=B0inv_R, Upsilon_R=Upsilon_R, Xi=Xi, SIGMA=SIGMA,Beta=Beta_theory),
                                initial_guess, xtol=1e-1000000, gtol=1e-1000000, max_nfev=iterations)

        B0inv = result.x.reshape((self.K, self.K))
  
        for i in range(0,self.K):
            if B0inv[i,i] < 0:
                B0inv[:,i] = -B0inv[:,i]
        
        if np.allclose((B0inv @ B0inv.T - SIGMA), 0) and np.allclose((Beta_theory @ Xi @ B0inv), 0) and np.allclose((np.linalg.inv(B0inv) @ SIGMA @ np.linalg.inv(B0inv).T), np.eye(self.K),atol=atol_number) == True:
            print("      Solver works correctly.\n")
        else:
            print("\nSolver does not work correctly. Check the tests below:")
            print("    B_0^{-1}(B_0^{-1})' - \\Sigma = 0       ", np.allclose(B0inv @ B0inv.T - SIGMA, 0))
            print("    \\beta'\\Upsilon = 0                    ", np.allclose(Beta_theory @ Xi @ B0inv, 0))
            print("    (B_0^{-1} \\Sigma B_0^{-1})' = I_K      ", np.allclose(np.linalg.inv(B0inv) @ SIGMA @ np.linalg.inv(B0inv).T, np.eye(self.K)))
            raise KeyError("The least_squares solver failed to estimate B_0^{-1} correctly.")

        if Upsilon0 is not None and Umat is not None:
            Upsilon_0 = Upsilon0.T
            
            pipit=np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(Upsilon_0.T,Upsilon_0)),Upsilon_0.T),Xi),SIGMA),np.dot(np.dot(np.linalg.inv(np.dot(Upsilon_0.T,Upsilon_0)),Upsilon_0.T),Xi).T)            
            
            pimat=np.linalg.cholesky(pipit)
            Upsilon = np.dot(Upsilon_0,pimat)                       
            Fk=np.dot(np.dot(np.linalg.inv(np.dot(Upsilon.T,Upsilon)),Upsilon.T),Xi)

            xi=np.dot(alpha,np.linalg.inv(np.dot(Umat,alpha)))  

            i=0
            while i < self.K:
                j=0
                while j < rank:
                    if abs(xi[i,j]) < 1E-8: #just to make sure that elements are = 0
                        xi[i,j] = 0
                    j=j+1
                i=i+1

            qr=np.linalg.cholesky(np.dot(np.dot(xi.T,np.linalg.inv(SIGMA)),xi))                   
            Fr=np.dot(np.dot(np.linalg.inv(qr),xi.T),np.linalg.inv(SIGMA))

            invB0 = np.linalg.inv(np.vstack((Fk,Fr)))

            if np.allclose(abs(invB0), abs(B0inv)) == True:
                print("        The analytical solution is identical to the estimate")
                print("        abs(invB0) ≈ abs(B0inv_solve) ",np.allclose(abs(invB0), abs(B0inv)),"\n")
            else:
                print("        The analytical solution is NOT identical to the estimate")
                print("        abs(invB0) != abs(B0inv_solve)",np.allclose(abs(invB0), abs(B0inv)),"\n")

        IRFmat = np.zeros((n_trials,(self.K*self.K)*(self.horizon+1)), dtype=np.float64)
        VCmat  = np.zeros((n_trials,(self.K*self.K)*(self.horizon+1)), dtype=np.float64)

        trial = fail = total_trials = failed_to_estimate = Xi_error = 0
        t,_ = self.y_dataframe.shape

        A = self.vectovar(beta=Beta_theory.T,alpha=alpha)
        a = A[0:self.K,:]

        if self.Constant == True and self.Trend == True:
            V = Beta[len(Beta)-1-1:len(Beta),:].T
        elif self.Constant == True and self.Trend == False:
            V = Beta[len(Beta)-1-0:len(Beta),:].T
        elif self.Constant == False and self.Trend == True:
            V = Beta[len(Beta)-0-1:len(Beta),:].T
        else:
            V = Beta[len(Beta)-0-0:len(Beta),:].T
        
        y = self.y_dataframe.to_numpy()

        while trial <= n_trials:

            if Bootstrap == False:
                y0p = y[:self.lags-1,:self.K-1]  # in Python, indexing starts from 0
            else:
                pos = random.randint(self.lags+1, t - 1)  # in Python, randint's upper limit is inclusive
                y0p = y[(pos-self.lags+1):(pos+1), :]  # add 1 to pos for Python's exclusive upper limit in slicing
                
            # Draw with replacement from uHat
            indexur = np.random.randint(0, t-self.lags, size=t-self.lags)  # Using numpy.random.randint! Generate Tbig
            u_r = u[:, indexur]
            u_r = np.hstack((np.zeros((self.K, self.lags)), u_r))  # Add residuals for initial conditions

            # Handle deterministic components: Only constant and linear trend allowed
            if self.Constant == True:
                determ = np.ones((t, 1))
            else:
                determ = np.array([]).reshape(0, 1)  # define an empty column vector
            
            if self.Trend == True:
                added_array = np.arange(-self.lags+1, t-self.lags+1).reshape(-1, 1)
                
                if self.Constant == True:
                    determ = np.column_stack((determ,added_array))
                else:
                    determ = np.column_stack((added_array))
            
            i = self.lags
            j = 0
            y_r = np.zeros((self.K, t-self.lags))  # simulated y for t=1:Tbig
            y_r = np.hstack((y0p.T, y_r))  # Add initial values

            while i < t:   # Python uses 0-based indexing
                index = np.flip(np.arange(j, j+self.lags))
                ylags = self.cointegration_tester.Vectorization(y_r[:, index])
                ylags = np.reshape(ylags, (self.K,self.lags))
                ylags = self.cointegration_tester.Vectorization(ylags.T).reshape(-1,1)
                
                if Bootstrap_type == "Wild":
                    try:    # If Wild Gaussian BS, multiply residuals with random number
                        y_r[:, i] = (V @ determ[i, :].reshape(-1, 1) + a @ ylags + np.random.randn() * u_r[:, i].reshape(-1,1)).ravel()   # np.random.randn() generates a random number from a standard normal distribution
                    except:
                        print(y_r.shape)
                        assert False    
                else:       # Non-parametric BS (standard recursive bootstrap)
                    y_r[:, i] = (V @ determ[i, :].reshape(-1, 1) + a @ ylags + u_r[:, i].reshape(-1,1)).ravel()

                i = i + 1
                j = j + 1
            
            alpha_r, Gamma_r, SIGMA_r, _ = self.VECMknown(beta=Beta_theory.T,y=y_r.T)
            #BetaB, BetavecB, _,_,_,SIGMA_r = self.LSKnownBeta(yr.T,p,beta_boot) # sigma is also found here

            
            GAMMA_r = Gamma_r[:,1:self.K*(self.lags-1)+1] # first diff lags
            V_r = Gamma_r[:,0]

            beta_perp_r = self.cointegration_tester.null(Beta_theory)[1]
            alpha_perp_r = self.cointegration_tester.null(alpha_r.T)[1]
            
            Gamma_sum_r = GAMMA_r[0:self.K,0:self.K]

            if self.lags > 1: 
                for i in range(1,self.lags-1): 
                    Gamma_sum_r = Gamma_sum_r+GAMMA_r[0:self.K,i*self.K:i*self.K+self.K]

            help = np.linalg.inv(np.dot(np.dot(alpha_perp_r.T,(np.identity(self.K)-Gamma_sum_r)),beta_perp_r))
            Xi_r = np.dot(np.dot(beta_perp_r,help),alpha_perp_r.T)

            errors = 0

            if np.sum(np.abs(Beta_theory @ Xi_r)>1e-12) != 0: 
                print("\nIdentification of Xi failed at test 1")
                print(np.sum(np.abs(Beta_theory @ Xi_r)))
                errors = errors + 1 
            if np.sum(np.abs(np.linalg.eig(Xi_r.T @  Xi_r)[0])>1e-12) != self.K-rank: 
                print("\nIdentification of Xi failed at test 2")
                print(np.sum(np.abs(np.linalg.eig(Xi_r.T @  Xi_r)[0])))
                errors = errors + 1 
            if errors != 0:
                raise KeyError('Xi does not satisfies assumptions')

            if Orthogonal == True:
                B0inv_r = np.linalg.cholesky(SIGMA_r)
                if trial == 0:
                    print("Cholesky Decomposition/Orthogonalisation:\n")

                for i in range(0,self.K):
                    if B0inv_r[i,i] < 0:
                        B0inv_r[:,i] = -B0inv_r[:,i]
                
            else:
                if estimator == "Analytical" and Upsilon0 is not None:
                    if trial == 0:
                        print("Analytical:     Requirements: Upsilon_0 and possibly U.\n")

                    if Upsilon0 is not None:
                        if Umat is None:
                            if trial == 0:
                                print("     *: The U matrix was not specified; an automatically generated U matrix has been used instead.\n")
                            Umat_est = np.zeros((rank,self.K))
                            for i in range(0, rank):
                                Umat_est[i,self.K-i-1] = 1
                                i = i + 1
                        else:
                            Umat_est = Umat 
                        
                        Upsilon_0 = Upsilon0.T
                        
                        pipit_r = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(Upsilon_0.T,Upsilon_0)),Upsilon_0.T),Xi_r),SIGMA_r),np.dot(np.dot(np.linalg.inv(np.dot(Upsilon_0.T,Upsilon_0)),Upsilon_0.T),Xi_r).T)            
                        
                        pimat_r = np.linalg.cholesky(pipit_r)
                        Upsilon_r = np.dot(Upsilon_0,pimat_r)                       
                        Fk = np.dot(np.dot(np.linalg.inv(np.dot(Upsilon_r.T,Upsilon_r)),Upsilon_r.T),Xi_r)

                        xi = np.dot(alpha_r,np.linalg.inv(np.dot(Umat_est,alpha_r)))  

                        i=0
                        while i < self.K:
                            j=0
                            while j< rank:
                                if abs(xi[i,j]) < 1E-8: #just to make sure that elements are = 0
                                    xi[i,j] = 0
                                j=j+1
                            i=i+1

                        qr = np.linalg.cholesky(np.dot(np.dot(xi.T,np.linalg.inv(SIGMA_r)),xi))                   
                        Fr = np.dot(np.dot(np.linalg.inv(qr),xi.T),np.linalg.inv(SIGMA_r))

                        B0inv_r = np.linalg.inv(np.vstack((Fk,Fr)))
                    else:
                        raise KeyError("Missing the matrices U and Upsilon_0")
                else:
                    if trial == 0:
                        print("Numerical Solver (least_squares)\n")

                    if initial_guess is None:
                        if iterations is None:
                            result = least_squares(lambda B0inv_flat: self.Restrictions(B0inv_flat, B0inv_R=B0inv_R, Upsilon_R=Upsilon_R, Xi=Xi_r, SIGMA=SIGMA_r,Beta=Beta_theory),
                                                np.random.randn(self.K * self.K))
                        else:
                            result = least_squares(lambda B0inv_flat: self.Restrictions(B0inv_flat, B0inv_R=B0inv_R, Upsilon_R=Upsilon_R, Xi=Xi_r, SIGMA=SIGMA_r,Beta=Beta_theory),
                                                np.random.randn(self.K * self.K),xtol=1e-1000000, gtol=1e-1000000, max_nfev=iterations)
                    else:
                        if iterations is None:
                            result = least_squares(lambda B0inv_flat: self.Restrictions(B0inv_flat, B0inv_R=B0inv_R, Upsilon_R=Upsilon_R, Xi=Xi_r, SIGMA=SIGMA_r,Beta=Beta_theory),
                                                initial_guess.flatten())
                        else:
                            result = least_squares(lambda B0inv_flat: self.Restrictions(B0inv_flat, B0inv_R=B0inv_R, Upsilon_R=Upsilon_R, Xi=Xi_r, SIGMA=SIGMA_r,Beta=Beta_theory),
                                                initial_guess.flatten(),xtol=1e-1000000, gtol=1e-1000000, max_nfev=iterations)

                    B0inv_r = result.x.reshape((self.K, self.K))
                    for i in range(0,self.K):
                        if B0inv_r[i,i] < 0:
                            B0inv_r[:,i] = -B0inv_r[:,i]
            
            A_r = self.vectovar(beta=Beta_theory.T, alpha=alpha_r,Gamma=GAMMA_r)
            
            e = np.abs(np.linalg.eigvals(A_r))

            if np.allclose((B0inv_r @ B0inv_r.T - SIGMA_r), 0) and np.allclose((np.linalg.inv(B0inv_r) @ SIGMA_r @ np.linalg.inv(B0inv_r).T), np.eye(self.K),atol=atol_number) and (errors == 0) is True:
                if np.max(e)>1.0000000000001:
                    fail = fail + 1
                    total_trials = total_trials + 1

                else:
                    irfr = self.IRF(A=A_r,B0inv=B0inv_r)

                    ##############################################################
                    ################ Linear Combinations
                    ##############################################################

                    if isinstance(linear_combinations, dict):
                        variable_name = [
                            [s_temp.strip() for s_temp in u_temp.split(" - ")]
                            for values in linear_combinations.values()
                            for u_temp in values]

                        Placeholder = []
                                                                    
                        for d in range(len(variable_name)):
                            Placeholder.append(variable_name[d][0])
                            
                            for o in range(self.K):
                                left_index = self.K * (self.list_of_info.index(variable_name[d][0]) - 1) + o
                                right_index = self.K * (self.list_of_info.index(variable_name[d][1]) - 1) + o
                                
                                column_index = (self.list_of_info.index(variable_name[d][0]) - 1) + o * self.K
                                irfr[:, column_index] = irfr[:, left_index] - irfr[:, right_index]

                    IRFmat[trial-1,:] = np.reshape(self.cointegration_tester.Vectorization(irfr.T), (1,-1))
                    
                    VCr = self.FEVD(A=A_r,B0inv=B0inv_r) * 100
                    VCmat[trial-1,:] = np.reshape(self.cointegration_tester.Vectorization(VCr.T), (1,-1))
                    
                    trial = trial + 1
                    total_trials = total_trials + 1
            else:
                if np.allclose((B0inv_r @ B0inv_r.T - SIGMA_r), 0) and np.allclose((np.linalg.inv(B0inv_r) @ SIGMA_r @ np.linalg.inv(B0inv_r).T), np.eye(self.K),atol=atol_number) is False:
                    failed_to_estimate = failed_to_estimate + 1
                elif (errors == 0) is False:
                    Xi_error = Xi_error + 1 

                total_trials = total_trials + 1
            
            print(f"Trial: {trial-1} of {total_trials-1}  |  Failures due to solver: {failed_to_estimate} | Failures due to unstable eigenvalue: {fail}",end="\r")

        Bootstrap_type_used = "Wild" if Bootstrap_type == "Wild" else "Non parametric"
        
        if Orthogonal == True:
            Estimator_used = "Cholesky Decomposition/Orthogonalisation (Overwrites)"
        else:
            Estimator_used = "Analytical" if estimator == "Analytical" and Upsilon0 is not None else "Switched to Numerical due to a missing Upsilon_0" if estimator == "Analytical" else "Numerical"

        print('\n\n  Bootstrap based on:',n_trials,'trials')
        print('  Estimator type in use:',Estimator_used)
        
        if Umat is None and estimator == "Analytical" and Upsilon0 is not None:
            print("The U matrix for the analytical solution is not unique and has been generated automatically.")
        
        print('  Bootstrap type in use:',Bootstrap_type_used)
        print('\nTotal errors:',total_trials-trial)
        print('     Errors due to unstable replications:',fail)
        print('     Errors due to Xi not satisfying assumptions:', Xi_error)
        print('     Errors due to solver failure to converge:', failed_to_estimate)

        return IRFmat, VCmat

    def IRF_Combined_Plots(self, Baseline="irf", Bootstrap_Matrix="IRF_mat", z_values=[68], 
                                          confidence_type ="Efron", Info="Wild", shocks = None, responses = None,
                                          linear_combinations=None, normalise=None, Bootstrap=True):
        """
        Plot Impulse Response Functions with confidence intervals for a list of Z-values.

        Parameters:
            - Baseline         : np.ndarrays   | Array containing the true/baseline impulse response function value
            - Bootstrap_Matrix : np.ndarrays   | Matrix containing impulse response function values.
            - shock_variable   : int           | Index of the shock variable.
            - z_values         : list          | List of z confidence levels (e.g., [68, 90, 99]).
            - I                : int, optional | Sign flip for the shock (+1 or -1). Default is 1.
        """


        if shocks is None:
            shocks = range(1, self.K+1)
            type_shock = [1] * self.K
        else:
            type_shock = [1 if s > 0 else -1 if s < 0 else 0 for s in shocks]
            shocks = np.abs(shocks)
        
        if responses is None:
            responses = range(1, self.K+1)

        confidence_results = {}

        if confidence_type == "Efron":
            z_values = [1]
            alpha_value=0.2
            Type = "Efron"
        else:
            alpha_value=0.1
            Type = "Delta"
        

        
        if Bootstrap == True:
            if isinstance(linear_combinations, dict) or linear_combinations is not None:
                Baseline = Baseline.copy()
                irf_copy = Baseline.copy()
                variable_name = [
                    [s_temp.strip() for s_temp in u_temp.split(" - ")]
                    for values in linear_combinations.values()
                    for u_temp in values]

                for d in range(len(variable_name)):                    
                    for o in range(self.K):
                        
                        left_index = self.K * (self.list_of_info.index(variable_name[d][0]) - 1) + o
                        right_index = self.K * (self.list_of_info.index(variable_name[d][1]) - 1) + o
                        column_index = (self.list_of_info.index(variable_name[d][0]) - 1) + o * self.K

                        Baseline[:, column_index] = irf_copy[:, left_index] - irf_copy[:, right_index]

                        #print(f"irf[:, {column_index}] = irf[:, {left_index}] - irf[:, {right_index}]")

                name_map = {expr.split(" - ")[0].strip(): name for name, values in linear_combinations.items() for expr in values}
                shock_names = [name_map.get(v, v)  for v in self.list_of_info]

            for z in z_values:

                if confidence_type == "Efron":
                    CI = np.percentile(Bootstrap_Matrix, [5, 95], axis=0, method='nearest')
                    CILO = np.reshape(CI[0,:], [self.K*self.K,self.horizon+1]).T
                    CIUP = np.reshape(CI[1,:], [self.K*self.K,self.horizon+1]).T
                else: # Delta
                    IRFrstd = np.reshape(np.std(Bootstrap_Matrix.T, axis=1), (self.K*self.K, self.horizon + 1))
                    CILO = Baseline - stats.norm.ppf(1 - (1 - z / 100) / 2) * IRFrstd.T
                    CIUP = Baseline + stats.norm.ppf(1 - (1 - z / 100) / 2) * IRFrstd.T

                CI_LO_result = self.IRF_estimation(IRF_var=CILO, list_of_info=self.list_of_info, normalise=normalise)
                CI_UP_result = self.IRF_estimation(IRF_var=CIUP, list_of_info=self.list_of_info, normalise=normalise)
                confidence_results[z] = (CI_LO_result, CI_UP_result)

        IRF_result = self.IRF_estimation(IRF_var=Baseline, list_of_info=self.list_of_info, normalise=normalise)#list_of_info[3]) # None)
        
        if Bootstrap == True:
            CI_LO_result = self.IRF_estimation(IRF_var=CILO, list_of_info=self.list_of_info, normalise=normalise)#list_of_info[3]) # None)
            CI_UP_result = self.IRF_estimation(IRF_var=CIUP, list_of_info=self.list_of_info, normalise=normalise)#list_of_info[3]) # None)
            print(f"Impulse Response Function\nBootstrap ({Info}) Confidence Intervals ({Type})")
        else:
            print(f"Impulse Response Function")

        fig, axes = plt.subplots(len(shocks), len(responses), squeeze=False, figsize=(len(responses)*5, len(shocks)*2.5))

        for row, shock_idx in enumerate(shocks):   # shocks
            for col, response_idx in enumerate(responses):  # responses
                color_map = {68: 'red', 90: 'blue',99: 'green'}
                ax = axes[row, col]
                ax.plot(type_shock[row] * IRF_result[self.list_of_info[response_idx]][self.list_of_info[shock_idx]], color="teal", linewidth=1.75)
                for z, (CI_LO_result, CI_UP_result) in confidence_results.items():
                    if Bootstrap == True:
                        ax.fill_between(
                            range(len(CI_LO_result[self.list_of_info[response_idx]][self.list_of_info[shock_idx]])),
                            CI_LO_result[self.list_of_info[response_idx]][self.list_of_info[shock_idx]] * type_shock[row],
                            CI_UP_result[self.list_of_info[response_idx]][self.list_of_info[shock_idx]] * type_shock[row],
                            color=color_map.get(z, 'gray'), alpha=alpha_value)
                        ax.plot(
                            range(len(CI_LO_result[self.list_of_info[response_idx]][self.list_of_info[shock_idx]])),
                            CI_LO_result[self.list_of_info[response_idx]][self.list_of_info[shock_idx]] * type_shock[row],
                            color="black", alpha=1.0, linestyle=':', linewidth=1)
                        ax.plot(
                            range(len(CI_UP_result[self.list_of_info[response_idx]][self.list_of_info[shock_idx]])),
                            CI_UP_result[self.list_of_info[response_idx]][self.list_of_info[shock_idx]] * type_shock[row],
                            color="black", alpha=1.0, linestyle=':', linewidth=1)
                if col == 0:
                    if isinstance(linear_combinations, dict):
                       ax.set_ylabel(f'{shock_names[shock_idx]}', fontname='Times New Roman')
                    else:
                        ax.set_ylabel(f'{self.list_of_info[shock_idx]}', fontname='Times New Roman')
                if row == 0:
                    ax.set_title(f'{self.list_of_info[response_idx]}', fontname='Times New Roman')
                fig.text(0.5, 1.01, 'Response Variables', ha='center', va='center', fontsize=14,fontname='Times New Roman')
                fig.text(-0.01, 0.5, 'Structural Shock from Variable:', ha='center', va='center', rotation='vertical', fontsize=14,fontname='Times New Roman')
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_position('zero')
                ax.spines['bottom'].set_position(('data', 0))
                ax.yaxis.tick_left()
                ax.xaxis.tick_bottom()
                for label in ax.get_yticklabels() + ax.get_xticklabels():
                    label.set_fontproperties(font_props)
                ax.set_xlim(0.01, self.horizon)
                ymin, ymax = ax.get_ylim()
                if ymin > 0:
                    ax.set_ylim(0, ymax)
                elif ymax < 0:
                    ax.set_ylim(ymin, 0)
                ax.set_xlim(0.01, self.horizon)

        plt.tight_layout()
        plt.show()

    def FEVD_Combined_Plots(self, Baseline="fevd", Bootstrap_Matrix="VC_mat", z_values=[68], 
                                  confidence_type ="Efron", Info="Wild", shocks = list, responses = list):
        """
        Plot Impulse Response Functions with confidence intervals for a list of Z-values.

        Parameters:
            - Baseline         : np.ndarrays   | Array containing the true/baseline impulse response function value
            - Bootstrap_Matrix : np.ndarrays   | Matrix containing impulse response function values.
            - shock_variable   : int           | Index of the shock variable.
            - z_values         : list          | List of z confidence levels (e.g., [68, 90, 99]).
            - I                : int, optional | Sign flip for the shock (+1 or -1). Default is 1.
        """
        confidence_results = {}
        if shocks is None:
            shocks = range(1, self.K+1)
        
        if responses is None:
            responses = range(1, self.K+1)

        if confidence_type == "Efron":
            z_values = [5]
            alpha_value=0.2
            Type = "Efron"
        else:
            alpha_value=0.1
            Type = "Delta"
        
        for z in z_values:

            if confidence_type == "Efron":
                CI = np.percentile(Bootstrap_Matrix, [z, 100-z], axis=0, method='nearest')
                CILO = np.reshape(CI[0,:], [self.K*self.K,self.horizon+1]).T
                CIUP = np.reshape(CI[1,:], [self.K*self.K,self.horizon+1]).T
            # elif confidence_type == "Hall":
            #     CI = np.percentile(Bootstrap_Matrix, [(100 - z) / 2, (z + (100 - z) / 2)], axis=0)
            #     print()
            #     CILO = np.reshape(CI[0,:], [self.K*self.K,self.horizon+1]).T
            #     CIUP = np.reshape(CI[1,:], [self.K*self.K,self.horizon+1]).T
            else:
                IRFrstd = np.reshape(np.std(Bootstrap_Matrix.T, axis=1), (self.K*self.K, self.horizon + 1))
                CILO = Baseline - stats.norm.ppf(1 - (1 - z / 100) / 2) * IRFrstd.T
                CIUP = Baseline + stats.norm.ppf(1 - (1 - z / 100) / 2) * IRFrstd.T

            CI_LO_result = self.FEVD_estimation(FEVD_var=CILO, list_of_info=self.list_of_info, normalise=None)
            CI_UP_result = self.FEVD_estimation(FEVD_var=CIUP, list_of_info=self.list_of_info, normalise=None)
            confidence_results[z] = (CI_LO_result, CI_UP_result)

        IRF_result = self.FEVD_estimation(FEVD_var=Baseline, list_of_info=self.list_of_info, normalise=None)#list_of_info[3]) # None)
        CI_LO_result = self.FEVD_estimation(FEVD_var=CILO, list_of_info=self.list_of_info, normalise=None)#list_of_info[3]) # None)
        CI_UP_result = self.FEVD_estimation(FEVD_var=CIUP, list_of_info=self.list_of_info, normalise=None)#list_of_info[3]) # None)
        print(f"Forecast Error Variance Decomposition\nBootstrap ({Info}) confidence intervals ({Type})")

        fig, axes = plt.subplots(len(shocks), len(responses), squeeze=False, figsize=(len(responses)*5, len(shocks)*2.5))

        for row, shock_idx in enumerate(shocks):   # shocks
            for col, response_idx in enumerate(responses):  # responses
                color_map = {68: 'red', 90: 'blue',99: 'green'}
                ax = axes[row, col]
                ax.plot(IRF_result[self.list_of_info[shock_idx]][self.list_of_info[response_idx]]/100, color="teal", linewidth=1.75)
                for z, (CI_LO_result, CI_UP_result) in confidence_results.items():
                    ax.fill_between(
                        range(len(CI_LO_result[self.list_of_info[shock_idx]][self.list_of_info[response_idx]])),
                        CI_LO_result[self.list_of_info[shock_idx]][self.list_of_info[response_idx]]/100,
                        CI_UP_result[self.list_of_info[shock_idx]][self.list_of_info[response_idx]]/100,
                        color=color_map.get(z, 'royalblue'), alpha=alpha_value)
                    ax.plot(
                        range(len(CI_LO_result[self.list_of_info[shock_idx]][self.list_of_info[response_idx]])),
                        CI_LO_result[self.list_of_info[shock_idx]][self.list_of_info[response_idx]]/100,
                        color="black", alpha=1.0, linestyle=':', linewidth=1)
                    ax.plot(
                        range(len(CI_UP_result[self.list_of_info[shock_idx]][self.list_of_info[response_idx]])),
                        CI_UP_result[self.list_of_info[shock_idx]][self.list_of_info[response_idx]]/100,
                        color="black", alpha=1.0, linestyle=':', linewidth=1)
                if col == 0:
                    ax.set_ylabel(f'{self.list_of_info[shock_idx]}', fontname='Times New Roman')
                if row == 0:
                    ax.set_title(f'{self.list_of_info[response_idx]}', fontname='Times New Roman')
                fig.text(0.5, 1.01, 'Response Variables', ha='center', va='center', fontsize=14,fontname='Times New Roman')
                fig.text(-0.01, 0.5, 'Structural Shock from Variable:', ha='center', va='center', rotation='vertical', fontsize=14,fontname='Times New Roman')
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.00))
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_position('zero')
                ax.spines['bottom'].set_position(('data', 0))
                ax.yaxis.tick_left()
                ax.xaxis.tick_bottom()
                for label in ax.get_yticklabels() + ax.get_xticklabels():
                    label.set_fontproperties(font_props)
                ax.set_xlim(0.01, self.horizon)
                ymin, ymax = ax.get_ylim()
                if ymin > 0:
                    ax.set_ylim(0, ymax)
                elif ymax < 0:
                    ax.set_ylim(ymin, 0)
                ax.set_xlim(0.01, self.horizon)

        plt.tight_layout()
        plt.show()

    def Historical_Decomposition(self,VEC=False):
        """
        Input
        A = Companion matrix A
        mu = constant
        What = structural shocks
        B0inv = inv(B0)
        K = # variables
        p = # lags
        indep = dependent variables excluding constant

        Output
        HDinit = initial conditions
        HDconst = constant
        HDshocks(obs,shock,variable) = (nobs x shock & variable, i.e.,
        HDshock(:,1,1) contains the effect of shock 1 on variable 1;
        HDshock(:,2,1) contains the effect of shock 2 on variable 1 and so on.

        Written by Michael Bergman 2020 inspired by code written by Ambrogio Cesa Bianchi
        Updated and corrected 2023.09.12
        """
        if VEC == True:
            Beta = Beta

        else:
            if self.Trend is True:
                raise KeyError("Historical Decomposition does not support a trend")
            else:
                Beta, _, _, residuals, _, SIGMA = self.diagnostic_tester.VAR_estimation_with_exogenous(lags=self.lags,Constant=self.Constant,Trend=self.Trend,Exogenous=self.Exogenous)
                A = self.diagnostic_tester.Companion_matrix(Beta,lags=self.lags)
                B0inv = np.linalg.cholesky(SIGMA)

                if self.Constant is True:
                    mu = Beta[-1, :].reshape(-1, 1)
                else:
                    raise KeyError("Missing a constant")

                What = np.dot(np.linalg.inv(B0inv),residuals.T)

                indep = self.diagnostic_tester.lagmatrix(y_dataframe=self.y_dataframe, lags=self.lags)
                indep = indep[self.lags:, :]
        
        nobs,KK = indep.shape 
        jmat = self.Jmatrix()

        init_big = np.zeros([self.lags*self.K,nobs+1])
        init = np.zeros([self.K,nobs+1])
        init_big[:,0] = indep[0,0:self.lags*self.K].T
        init[:,0] = np.dot(jmat,init_big[:,0])
        
        i = 1
        while i<=nobs:
            init_big[:,i] = np.dot(A,init_big[:,i-1])
            init[:,i] = np.dot(jmat,init_big[:,i])
            i = i+1
        
        
        # Constant
        const_big = np.zeros([self.lags*self.K,nobs+1])
        const = np.zeros([self.K,nobs+1]) 
        CC = np.zeros([self.lags*self.K,1])
        
        if np.any(mu) == True:
            CC[0:self.K,:] = mu
            i = 1
            while i<=nobs:
                const_big[:,i] = np.array(CC).T + np.array(np.dot(A,const_big[:,i-1])).T
                const[:,i] = np.dot(jmat,const_big[:,i])
                i = i+1
        
            B_big = np.zeros([self.lags*self.K,self.K])
            B_big[0:self.K,:] = B0inv
            shock_big = np.zeros((self.lags*self.K,nobs+1,self.K))
            shock = np.zeros((self.K,nobs+1,self.K))
            j = 0
            while j<=self.K-1:
                What_big = np.zeros([self.K,nobs+1])
                What_big[j,1:nobs+1] = What[j,0:nobs]
                i = 0
                while i<=nobs:
                    shock_big[:,i,j] = np.dot(B_big,What_big[:,i]).T + np.dot(A,shock_big[:,i-1,j]).T
                    shock[:,i,j] = np.dot(jmat,shock_big[:,i,j])
                    i = i + 1
        
                j = j + 1 
        
        HDendo = init + const + np.sum(shock.T, axis=0).T 
        HDshock = np.zeros((nobs+1,self.K,self.K))
        
        i = 0 
        while i <= self.K-1:
            j = 0
            while j <= self.K-1:
                HDshock[1:nobs+self.lags,j,i] = shock[i,1:nobs+self.lags,j].T
                j = j + 1 
        
            i = i + 1 
                
        missing = np.full((self.K,self.lags),np.nan)
        missing2 = np.full((self.lags,self.K,self.K), np.nan)
        
        HDshock = np.vstack((missing2,HDshock[1:len(HDshock),:,:]))
        HDinit = init.T                       
        HDinit = np.vstack((missing[:,0:self.lags-1].T,HDinit))
        HDconst = const.T
        HDconst = np.hstack((missing,HDconst[1:len(HDconst),:].T))
        HDendo = np.hstack((missing,HDendo[:,1:len(HDendo.T)])).T
        
        
        return HDinit, HDconst, HDshock, HDendo

    def Local_Projection_irf(self, B0inv, multiplier=1.96):
        """
        Compute Local Projection Impulse Response Functions (LPIRF) with Newey-West adjusted standard errors.

        Parameters:
        - y (np.ndarray): Time series data of shape (T, K).
        - p (int): Number of lags.
        - h (int): Maximum horizon for the impulse response.
        - B0inv (np.ndarray): Inverse of the contemporaneous impact matrix of shape (K, K).
        - se_multiplier (float): Multiplier for standard errors (e.g., 1.96 for 95% confidence interval).

        Returns:
        - lpirf (np.ndarray): Estimated LPIRF coefficients.
        - upper_band (np.ndarray): Upper confidence band.
        - lower_band (np.ndarray): Lower confidence band.
        """
        y = pd.DataFrame(self.y_dataframe).to_numpy()
    
        
        p = self.lags
        h = self.horizon
        
        T, K = y.shape
        nw_lag = max(int(np.round(4 * (T / 100) ** (2 / 9))), 1)  # Newey-West lag length

        # Dependent variable starting from lag p
        y_dep = y[p:, :]
        num_obs = y_dep.shape[0]

        # First lag of y
        y_lag1 = y[p - 1:T - 1, :]

        # Construct regressor matrix X (constant term and lagged variables)
        X = np.ones((num_obs, 1))  # Constant term
        for lag in range(1, p):
            lagged_y = y[p - lag - 1:T - lag - 1, :]
            X = np.hstack((X, lagged_y))

        # Generate the leads for the projections
        num_projections = num_obs - h + 1  # Adjusted number of observations for projections
        yy = y_dep[0:num_projections, :]
        for j in range(1, h):
            lead_y = y_dep[j:num_projections + j, :]
            yy = np.hstack((yy, lead_y))


        # Resize regressors for the lost observations due to leads
        yy1 = y_lag1[0:num_projections, :]

        xx = X[0:num_projections, :]

        # Create the idempotent matrix px
        xx_inv = np.linalg.inv(xx.T @ xx)
        px = np.eye(num_projections) - xx @ xx_inv @ xx.T



        # IRF Coefficient Matrix
        gamma_x = np.linalg.inv(yy1.T @ px @ yy1) @ yy1.T @ px @ yy

        # Compute residuals
        residuals = px @ (yy - yy1 @ gamma_x)


        # Demean residuals
        residuals_demeaned = residuals - residuals.mean(axis=0)

        # Variance-Covariance Matrix of the IRF Coefficients
        v_matrix = residuals_demeaned.T @ residuals_demeaned

        # Newey-West adjustment
        for lag in range(1, int(nw_lag) + 1):
            weight = 1 - lag / (nw_lag + 1)
            lagged_residuals = residuals_demeaned[lag:, :]
            lead_residuals = residuals_demeaned[:-lag, :]
            gamma_k = lagged_residuals.T @ lead_residuals
            v_matrix += weight * (gamma_k + gamma_k.T)


        v_matrix /= (len(yy) - K)

        var_gamma_x = np.kron(v_matrix, np.linalg.inv(yy1.T @ px @ yy1))
        omega = (np.kron(np.eye(K * h), B0inv) @ var_gamma_x @ np.kron(np.eye(K * h), B0inv).T)

        # Structural Impulse-Response Coefficients
        gamma_x_struct = B0inv.T @ gamma_x

        

        # Organize the coefficients into lpirf
        gamy = gamma_x_struct[0, :].reshape(h, K).T
        for i in range(1, K):
            temp = gamma_x_struct[i, :].reshape(h, K).T
            gamy = np.vstack((gamy, temp))
        gamy = gamy.T


        # Compute lpirf
        initial_impact = B0inv.T.reshape(1, K * K)
        lpirf = np.vstack((initial_impact, gamy[0:h - 1, :]))

        # Compute standard errors
        sd_vector = np.sqrt(np.diag(omega))
        sd_matrix = sd_vector.reshape(K * h, K).T  # Shape: (K, h * K)

        # Organize standard errors
        sdd = sd_matrix[0, :].reshape(h, K).T
        for i in range(1, K):
            temp = sd_matrix[i, :].reshape(h, K).T
            sdd = np.vstack((sdd, temp))
        sdd = sdd.T
        zeros_row = np.zeros((1, lpirf.shape[1]))
        sdd = np.vstack((zeros_row, sdd[0:h - 1, :]))

        # Compute confidence bands
        upper_band = lpirf + multiplier * sdd
        lower_band = lpirf - multiplier * sdd

        return lpirf, upper_band, lower_band