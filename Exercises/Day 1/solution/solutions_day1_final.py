import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, f_oneway, kruskal




# Exercise 1 _________________________________________________________________
print("Exercise 1 - Brain _______________________________________________________________")
# 1. Load data
braindata = pd.read_csv("brainweight.txt", sep='\s+')

# Print the first few rows and column names to ensure correct loading
print(braindata.head())
print(braindata.columns)

# Ensure 'body' and 'brain' columns are present
if 'body' not in braindata.columns or 'brain' not in braindata.columns:
    raise KeyError("The required columns 'body' and 'brain' are not present in the DataFrame.")

# 1. Plot body vs brain
plt.figure(figsize=(10, 6))
plt.scatter(braindata['body'], braindata['brain'])
plt.xlabel('Body Weight')
plt.ylabel('Brain Weight')
plt.title('Body Weight vs Brain Weight')
plt.show()

# Calculate Pearson correlation
pearson_corr = braindata['brain'].corr(braindata['body'])
print(f'Pearson correlation: {pearson_corr}')

# Calculate Spearman correlation
spearman_corr, _ = spearmanr(braindata['brain'], braindata['body'])
print(f'Spearman correlation: {spearman_corr}')

# 2. Log transform brain and body weights
braindata['logbrain'] = np.log(braindata['brain'])
braindata['logbody'] = np.log(braindata['body'])

# Plot log-transformed body vs log-transformed brain
plt.figure(figsize=(10, 6))
plt.scatter(braindata['logbody'], braindata['logbrain'])
plt.xlabel('Log Body Weight')
plt.ylabel('Log Brain Weight')
plt.title('Log Body Weight vs Log Brain Weight')
plt.show()

# Calculate Pearson correlation on log-transformed data
log_pearson_corr = braindata['logbrain'].corr(braindata['logbody'])
print(f'Log Pearson correlation: {log_pearson_corr}')

# Calculate Spearman correlation on log-transformed data
log_spearman_corr, _ = spearmanr(braindata['logbrain'], braindata['logbody'])
print(f'Log Spearman correlation: {log_spearman_corr}')

# 3. Linear regression on log-transformed data
X = braindata[['logbody']]
y = braindata['logbrain']
model = sm.OLS(y, sm.add_constant(X)).fit()

# Print summary of the model
print(model.summary())

# 4. Plot log-transformed data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(braindata['logbody'], braindata['logbrain'])
plt.plot(braindata['logbody'], model.predict(sm.add_constant(braindata[['logbody']])), color='red')
plt.xlabel('Log Body Weight')
plt.ylabel('Log Brain Weight')
plt.title('Log Body Weight vs Log Brain Weight with Regression Line')
plt.show()

# 5. Plot residuals vs fitted values and QQ plot
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax[0, 0], line_kws={'color': 'red', 'lw': 1})
ax[0, 0].set_title('Residuals vs Fitted')
ax[0, 0].set_xlabel('Fitted values')
ax[0, 0].set_ylabel('Residuals')

# QQ plot
sm.qqplot(model.resid, line='45', ax=ax[0, 1])
ax[0, 1].set_title('QQ plot')

# Scale-Location
sns.scatterplot(x=model.fittedvalues, y=np.sqrt(np.abs(model.resid)), ax=ax[1, 0])
ax[1, 0].set_title('Scale-Location')
ax[1, 0].set_xlabel('Fitted values')
ax[1, 0].set_ylabel('√|Residuals|')

# Residuals distribution
sns.histplot(model.resid, kde=True, ax=ax[1, 1])
ax[1, 1].set_title('Residuals Distribution')
ax[1, 1].set_xlabel('Residuals')

plt.tight_layout()
plt.show()

# ### Comparison: Non-transformed
# 6. Linear regression on non-transformed data
X = braindata[['body']]
y = braindata['brain']
model = sm.OLS(y, sm.add_constant(X)).fit()

# Print summary of the non-transformed model
print(model.summary())

# Plot non-transformed data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(braindata['body'], braindata['brain'])
plt.plot(braindata['body'], model.predict(sm.add_constant(braindata[['body']])), color='red')
plt.xlabel('Body Weight')
plt.ylabel('Brain Weight')
plt.title('Body Weight vs Brain Weight with Regression Line')
plt.show()

# Plot residuals vs fitted values and QQ plot for non-transformed data
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Residuals vs Fitted for non-transformed data
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax[0], line_kws={'color': 'red', 'lw': 1})
ax[0].set_title('Residuals vs Fitted (Non-Transformed)')
ax[0].set_xlabel('Fitted values')
ax[0].set_ylabel('Residuals')

# QQ plot for non-transformed data
sm.qqplot(model.resid, line='45', ax=ax[1])
ax[1].set_title('QQ plot (Non-Transformed)')

plt.tight_layout()
plt.show()

# Exercise 2 _________________________________________________________________
print("Exercise 2 - Laborforce _______________________________________________________________")
# Load data
labor = pd.read_csv("labor.txt", sep='\s+')

# Boxplot for both columns
plt.figure(figsize=(10, 6))
plt.boxplot([labor['x1968'], labor['x1972']], labels=['1968', '1972'])
plt.title('Boxplot of Labor Data for 1968 and 1972')
plt.show()

# 1. Two-sample t-test assuming equal variances
t_stat, p_val = stats.ttest_ind(labor['x1968'], labor['x1972'], equal_var=True)
print(f'Two-sample t-test (equal variances): t-statistic = {t_stat}, p-value = {p_val}')

# 2. Paired t-test
t_stat_paired, p_val_paired = stats.ttest_rel(labor['x1968'], labor['x1972'])
print(f'Paired t-test: t-statistic = {t_stat_paired}, p-value = {p_val_paired}')

# 4. Mean difference
mean_diff = labor['x1972'].mean() - labor['x1968'].mean()
print(f'Mean difference (1972 - 1968): {mean_diff}')

# Exercise 3 _________________________________________________________________
print("Exercise 3 - Log transformation _______________________________________________________________")
# 1. Log of y
x = np.linspace(1.5, 2.5, 100)
logy = 3 * x + np.random.normal(scale=0.15, size=100)
y = np.exp(logy)

# Plot x vs y
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.legend()
plt.show()

# Linear regression on log(y)
X = sm.add_constant(x)
model = sm.OLS(np.log(y), X).fit()
print(model.summary())

# Extract coefficients
coef_alpha = model.params[0]
coef_beta = model.params[1]

# Fit model
fit_model = np.exp(coef_alpha + coef_beta * x)

# Plot fitted model
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.plot(x, fit_model, color='red', label='Fitted model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y with Fitted Model')
plt.legend()
plt.show()

# 2. Log-log
logx = np.linspace(1.5, 2.5, 100)
logy = 3 * logx + np.random.normal(scale=0.15, size=100)
y = np.exp(logy)
x = np.exp(logx)

# Plot log(x) vs y
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('log(x) vs y')
plt.legend()
plt.show()

# Linear regression on log(y) vs log(x)
X_log = sm.add_constant(np.log(x))
model_log = sm.OLS(np.log(y), X_log).fit()
print(model_log.summary())

# Extract coefficients
coef_alpha_log = model_log.params[0]
coef_beta_log = model_log.params[1]

# Fit model
fit_model_log = np.exp(coef_alpha_log + coef_beta_log * np.log(x))

# Plot fitted model
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.plot(x, fit_model_log, color='red', label='Fitted model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('log(x) vs y with Fitted Model')
plt.legend()
plt.show()

# 3. Log of x
logx = np.linspace(1.5, 2.5, 100)
y = 3 * logx + np.random.normal(scale=0.15, size=100)
x = np.exp(logx)

# Plot x vs y
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.legend()
plt.show()

# Linear regression on y vs log(x)
X_logx = sm.add_constant(np.log(x))
model_logx = sm.OLS(y, X_logx).fit()
print(model_logx.summary())

# Extract coefficients
coef_alpha_logx = model_logx.params[0]
coef_beta_logx = model_logx.params[1]

# Fit model
fit_model_logx = coef_alpha_logx + coef_beta_logx * np.log(x)

# Plot fitted model
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.plot(x, fit_model_logx, color='red', label='Fitted model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y with Fitted Model')
plt.legend()
plt.show()

# Exercise 4 _________________________________________________________________
print("Exercise 4 - Calcium _______________________________________________________________")
# Load data
calc = pd.read_csv("calcium.txt")

# Two sample setting
# 2. Q-Q plot for Calcium and Placebo
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sm.qqplot(calc['Decrease'].iloc[:9], line='q', marker='o')
plt.title('Q-Q plot for Calcium')
plt.subplot(1, 2, 2)
sm.qqplot(calc['Decrease'].iloc[9:], line='q', marker='o')
plt.title('Q-Q plot for Placebo')
plt.show()

# 3. Variance test
var_test_result = f_oneway(calc[calc['Treatment'] == 'Calcium']['Decrease'].dropna(),
                            calc[calc['Treatment'] == 'Placebo']['Decrease'].dropna())
print("Variance test p-value:", var_test_result.pvalue)

# 4. Boxplot
plt.figure(figsize=(8, 6))
calc.boxplot(column='Decrease', by='Treatment')
plt.title('Boxplot of Decrease by Treatment')
plt.xlabel('Treatment')
plt.ylabel('Decrease')
plt.show()

# 5. Two-sample t-test assuming equal variances
t_test_result = ttest_ind(calc[calc['Treatment'] == 'Calcium']['Decrease'].dropna(),
                          calc[calc['Treatment'] == 'Placebo']['Decrease'].dropna(), equal_var=True)
print("Two-sample t-test p-value (equal variances):", t_test_result.pvalue)

# 6. Wilcoxon rank-sum test (Mann-Whitney U test)
mannwhitneyu_test_result = mannwhitneyu(calc[calc['Treatment'] == 'Calcium']['Decrease'].dropna(),
                                         calc[calc['Treatment'] == 'Placebo']['Decrease'].dropna())
print("Wilcoxon rank-sum test p-value:", mannwhitneyu_test_result.pvalue)

# Conidia
cold = [1575, 2019, 1921, 2019, 2323]
medium = [2003, np.nan, 1510, 1991, 1720]
warm = [1742, 1764, np.nan, 1470, 1769]

# Boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([cold, medium, warm], labels=['cold', 'medium', 'warm'])
plt.title('Boxplot of Conidia Discharge by Temperature')
plt.xlabel('Temperature')
plt.ylabel('Discharge')
plt.show()

# ANOVA
conidia = pd.DataFrame({'temp': ['cold']*5 + ['medium']*5 + ['warm']*5,
                        'discharge': cold + medium + warm})
anova_model = sm.formula.ols('discharge ~ temp', data=conidia).fit()
print(anova_model.summary())

# Kruskal-Wallis test
kruskal_test_result = kruskal(cold, medium, warm)
print("Kruskal-Wallis test p-value:", kruskal_test_result.pvalue)