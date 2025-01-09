""" The Impact of Technology Companies on Career Choices of Youth in Armenia """

import pandas as pd
import numpy as np
import re
import datetime
from dateutil import parser
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sys import displayhook
from statsmodels.regression.linear_model import OLSResults
from statsmodels.stats.outliers_influence import variance_inflation_factor

ysu_data_file = "./data/ysu_annual_applications_processed_data - English.csv"
tech_company_file = "./data/Armenian_ICT_sector_statistics.csv"
arm_pop_file = "./data/Armenian_population_2000-23.csv"
arm_gdp_file ="./data/Armenian_gdp_2000-23.csv"
# Create a function to process the variable in a desirable panel data format
def prep_var(file_path):
    var = pd.read_csv(file_path, na_values=np.nan) # 47 cross-sections
    var.columns = list(var.columns[:1]) + [pd.to_datetime(int(year), format="%Y") for year in var.columns[1:]]
    var = var.set_index("faculty").T
    var = var.reset_index().melt(id_vars=["index"])
    return(var)
 
"""_____Dependent Var. Number of annual faculty applications for Yerevan State University (YSU)_____"""
ysu_data = prep_var(ysu_data_file)
ysu_data = ysu_data.rename(columns={"index":"year", "value":"fac_app"})
ysu_data.info()
print(ysu_data.isnull().sum())

ysu_data = ysu_data.dropna(subset=["fac_app"])
ysu_data.info()

''' 1. By dividing  the "number of applicants for a faculty" with the "number of total applicants for all faculties", 
    we can get more homogeneous dataset, regardless of how many applicants there were applying for each year for each faculty. 
    2. By z-scorizing (standardizing), we can adjust for the level for each faculty.
    So, by 1. and 2. we can make the claim that:
    `Independent variables might have affected the dependent variable which is equal to ("above/below the average time-series share(%) level of 
    {STEM divided by other faculties} regardless of the the number of applicants.")`. '''

# group the dataframe by each year, divide each value of "fac_app" column by the sum of "fac_app"s values for each given year.
ysu_data = ysu_data.assign(rel_fac_app = ysu_data.groupby("year")["fac_app"].transform( lambda x: x/x.sum() ))
ysu_data = ysu_data.sort_values(by=["faculty", "year"])
ysu_data = ysu_data.assign(relative_faculty_app = ysu_data.groupby("faculty")["rel_fac_app"].transform( lambda x: x.diff() ))
"""_____Dependent Var_____"""

"""_____Independent Vars. Total number of ICT companies in Armenia, ICT trade volume, ICT share_____"""
it_com = pd.read_csv(tech_company_file)
it_com["year"] = it_com["year"].apply(lambda i: pd.to_datetime(int(i), format="%Y"))
it_com.rename(columns={"Armenian_ICT_sector_turnover,_million_drams":"trade_drams", "% of country's total services":"pct_in_gdp"}, inplace=True)
it_com["diff_(IT_share_in_gdp)"] = (it_com["pct_in_gdp"]/100).diff() # to convert percentages to numbers and take the difference
it_com["trade_pct_change"] = (it_com["trade_drams"]*1000000).pct_change()
it_com["company_count_pct_change"] = it_com["total_company_count"].pct_change()

arm_pop = pd.read_csv(arm_pop_file)
arm_pop["year"] = arm_pop["year"].apply(lambda i: pd.to_datetime(int(i), format="%Y"))
arm_pop['pop_pct_change'] = (arm_pop['pop'] * 1000).pct_change()
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='pop_pct_change', data=arm_pop)
arm_pop['pop_pct_change_stand'] = (arm_pop['pop_pct_change'] - arm_pop['pop_pct_change'].mean())/arm_pop['pop_pct_change'].std()
sns.lineplot(x='year', y='pop_pct_change_stand', data=arm_pop)

arm_gdp = pd.read_csv(arm_gdp_file)
arm_gdp["year"] = arm_gdp["year"].apply(lambda i: pd.to_datetime(int(i), format="%Y"))
arm_gdp['gdp_pct_change'] = (arm_gdp['gdp'] * 1000000).pct_change()
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='gdp_pct_change', data=arm_gdp)
arm_gdp['gdp_pct_change_stand'] = (arm_gdp['gdp_pct_change'] - arm_gdp['gdp_pct_change'].mean())/arm_gdp['gdp_pct_change'].std()
sns.lineplot(x='year', y='gdp_pct_change_stand', data=arm_gdp)

# remove the outliers of the year 2022
it_com = it_com.query("not (year == '2022-01-01')")

it_com["trade_volume_pct_change_stand_lag"] = ((it_com["trade_pct_change"] - it_com["trade_pct_change"].mean())/it_com["trade_pct_change"].std()).shift(2)
it_com["diff_(IT_share_in_gdp)_lag"] = ((it_com["diff_(IT_share_in_gdp)"] - it_com["diff_(IT_share_in_gdp)"].mean()) / it_com["diff_(IT_share_in_gdp)"].std()).shift(2)
it_com["company_count_pct_change_stand_lag"] = ((it_com["company_count_pct_change"] - it_com["company_count_pct_change"].mean())/it_com["company_count_pct_change"].std()).shift(2)
it_com.info()

"""_____END Independent Vars_____"""

panel_df = pd.merge(pd.merge(pd.merge(ysu_data, it_com, how='left', left_on = 'year', right_on='year'), arm_pop, how='left', left_on = 'year', right_on='year'), arm_gdp, how='left', left_on = 'year', right_on='year')

entire_stem_fac = ["Mathematics",
                    "Mechanics",
                    "Actuarial and Financial Mathematics",
                    "Informatics and Applied Mathematics, Applied Statistics and Data Science",
                    "Mathematics and Computer Sciences, Information Security",
                    "Physics, Applied Mathematics and Physics",
                    "Physics of Nuclear Reactors",
                    "Radiophysics and Electronics",
                    "Semiconductor Physics and Microelectronics",
                    "Telecommunications and Signal Processing",
                    "Biophysics, Bioinformatics",
                    "Chemistry",
                    "Ecological Chemistry, Biochemistry and Food Safety",
                    "Pharmaceutical Chemistry, Pharmacy",
                    "Biology",
                    "Geology",
                    "Geography",
                    "Cartography and Cadastre Work"]

len(entire_stem_fac) # 18 faculties

ict_stem_fac = ["Mathematics",
                "Actuarial and Financial Mathematics",
                "Informatics and Applied Mathematics, Applied Statistics and Data Science",
                "Mathematics and Computer Sciences, Information Security",
                "Physics, Applied Mathematics and Physics",
                "Radiophysics and Electronics",
                "Semiconductor Physics and Microelectronics",
                "Telecommunications and Signal Processing",
                "Biophysics, Bioinformatics"]

len(ict_stem_fac) # 9 faculties

fac_app_reg_name  = 'relative_faculty_app'
comp_count_reg_name  = 'company_count_pct_change_stand_lag'
it_gdp_reg_name  = 'diff_(IT_share_in_gdp)_lag'
it_turn_reg_name = 'trade_volume_pct_change_stand_lag'
it_log_reg_name = 'trade_log'
arm_pop_reg_name = 'pop_pct_change_stand'
arm_gdp_reg_name = 'gdp_pct_change_stand'

reg_var_list  = [fac_app_reg_name, it_turn_reg_name, arm_pop_reg_name]
panel_df = panel_df[['year'] + ['faculty'] + reg_var_list]
panel_df.describe()

panel_df = panel_df.query("not (year == '2022-01-01')")
stem_panel_df = panel_df.query("faculty in @ict_stem_fac") # includes 121 observations
stem_panel_df.rename(columns={'relative_faculty_app':'relative_stem_faculty_app'}, inplace=True)
stem_panel_df = stem_panel_df.reset_index(drop=True)
stem_panel_df['faculty'].nunique()
stem_panel_df.describe()

non_stem_panel_df = panel_df.query("faculty not in @ict_stem_fac") # includes 544 observations
non_stem_panel_df.rename(columns={'relative_faculty_app':'relative_non-stem_faculty_app'}, inplace=True)
non_stem_panel_df = non_stem_panel_df.reset_index(drop=True)
non_stem_panel_df['faculty'].nunique()
non_stem_panel_df.describe()

len(stem_panel_df)
stem_panel_df = stem_panel_df.dropna()

"""_____START PLOTS_____"""

# stem_panel_df.rename(columns={'relative_faculty_app':'relative_stem_faculty_app'}, inplace=True)
# non_stem_panel_df.rename(columns={'relative_faculty_app':'relative_non-stem_faculty_app'}, inplace=True)

# # Check correlation between the variables and the STEM faculty applications with the heatmap.
sns.set(font_scale=1.4)
htmp = sns.heatmap(stem_panel_df[stem_panel_df.columns[2:7]].corr(), vmin=-1, vmax=1, annot=True, fmt=".2f", linewidth=.5, cmap="vlag")
htmp.set_xticklabels(htmp.get_xmajorticklabels(), fontsize=16)
htmp.set_yticklabels(htmp.get_ymajorticklabels(), fontsize=16)
plt.show()

# # Check correlation between the variables and the non-STEM faculty applications with the heatmap.
sns.set(font_scale=1.4)
htmp = sns.heatmap(non_stem_panel_df[non_stem_panel_df.columns[2:7]].corr(), vmin=-1, vmax=1, annot=True, fmt=".2f", linewidth=.5, cmap="vlag")
htmp.set_xticklabels(htmp.get_xmajorticklabels(), fontsize=16)
htmp.set_yticklabels(htmp.get_ymajorticklabels(), fontsize=16)
plt.show()

# # Check linear relationship between the STEM-related faculties and independent variables, scatterplot
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=2)
sns.set_context("paper", font_scale=0.9) 
splot = sns.pairplot(stem_panel_df[stem_panel_df.columns[1:]], hue="faculty", grid_kws={"despine": False})
plt.show()

# # Check linear relationship between the non-STEM-related faculties and independent variables, scatterplot
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=2)
sns.set_context("paper", font_scale=0.9) 
splot = sns.pairplot(non_stem_panel_df[non_stem_panel_df.columns[1:]], hue="faculty", grid_kws={"despine": False})
plt.show()

"""_____END PLOTS_____"""

"""
Covariance type (refers to the 'c_type' variable's value while fitting the model)
    
    'HC0': White's (1980) heteroskedasticity robust standard errors.
    'HC1', 'HC2', 'HC3': MacKinnon and White's (1985) heteroskedasticity robust standard errors.
    'robust': White”s robust covariance
    'HAC' : heteroskedasticity-autocorrelation robust covariance 
"""

print(""" ____ START: Pooled OLS for STEM faculty applications ____""")
stem_panel_df = stem_panel_df.sort_values(by='year')
stem_panel_df = stem_panel_df.reset_index(drop=True)

# Generate a dummy variable for each faculty. 
for fac in stem_panel_df['faculty'].unique():
    stem_panel_df[ fac ] = ( stem_panel_df['faculty'] == fac ).apply( lambda x : int(x) )

c_type = 'cluster'
stem_endg = stem_panel_df[[reg_var_list[0]]] # using the STEM faculty applications as the dependent variable
type(stem_endg)
exog = stem_panel_df[stem_panel_df.columns[3:]]
type(exog)
stem_panel_df['year'] = pd.factorize(stem_panel_df['year'], sort = True) [0] + 1
stem_panel_df['faculty'] = pd.factorize(stem_panel_df['faculty'], sort = True) [0] + 1
mdl_stem = sm.OLS(stem_endg, sm.add_constant(exog))
fitted_mdl_stem = mdl_stem.fit(cov_type = 'cluster', cov_kwds={'groups' : np.array(stem_panel_df['year'])})
displayhook(fitted_mdl_stem.summary())

print("""____ START: Pooled OLS for non-STEM faculty applications ____""")

non_stem_panel_df = non_stem_panel_df.sort_values(by='year')
non_stem_panel_df = non_stem_panel_df.reset_index(drop=True)

# Generate a dummy variable for each faculty. 
for fac in non_stem_panel_df['faculty'].unique():
    non_stem_panel_df[ fac ] = ( non_stem_panel_df['faculty'] == fac ).apply( lambda x : int(x) )

non_stem_panel_df = non_stem_panel_df.dropna() # 475 non-null rows
non_stem_endg = non_stem_panel_df[reg_var_list[0]] # using the non-STEM faculty applications as the dependent variable
type(non_stem_panel_df)
exog = non_stem_panel_df[non_stem_panel_df.columns[3:]]
exog = exog.dropna()
type(exog)
non_stem_panel_df['year'] = pd.factorize(non_stem_panel_df['year'], sort = True) [0] + 1
non_stem_panel_df['faculty'] = pd.factorize(non_stem_panel_df['faculty'], sort = True) [0] + 1
mdl_non_stem = sm.OLS(non_stem_endg, sm.add_constant(exog))
fitted_mdl_non_stem = mdl_non_stem.fit(cov_type = 'cluster', cov_kwds={'groups' : np.array(non_stem_panel_df['year'])})
displayhook(fitted_mdl_non_stem.summary())
