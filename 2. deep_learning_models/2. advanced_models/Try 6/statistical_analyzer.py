import numpy as np
from scipy import stats
from typing import Dict

class StatisticalAnalyzer:
    
    @staticmethod
    def calculate_confidence_intervals(data: np.ndarray, confidence: float = 0.95) -> Dict:
        alpha = 1 - confidence
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        
        t_ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
        
        return {
            'mean': float(mean),
            'se': float(se),
            'n': int(n),
            'confidence_level': confidence,
            'confidence_interval': (float(t_ci[0]), float(t_ci[1]))
        }
    
    @staticmethod
    def paired_ttest_analysis(before: np.ndarray, after: np.ndarray, alpha: float = 0.05) -> Dict:
        statistic, p_value = stats.ttest_rel(after, before)
        
        diff = after - before
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0
        
        n = len(diff)
        se = stats.sem(diff)
        ci_lower, ci_upper = stats.t.interval(1-alpha, n-1, loc=np.mean(diff), scale=se)
        
        def interpret_effect_size(d):
            if abs(d) < 0.2:
                return "negligible"
            elif abs(d) < 0.5:
                return "small"
            elif abs(d) < 0.8:
                return "medium"
            else:
                return "large"
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'effect_size_cohens_d': float(effect_size),
            'effect_size_interpretation': interpret_effect_size(effect_size),
            'mean_difference': float(np.mean(diff)),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'sample_size': int(n)
        }
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict:
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        both_correct = np.sum(correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        only_1_correct = np.sum(correct1 & ~correct2)
        only_2_correct = np.sum(~correct1 & correct2)
        
        if only_1_correct + only_2_correct == 0:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'contingency_table': [[both_correct, only_1_correct], 
                                    [only_2_correct, both_wrong]]
            }
        
        statistic = (abs(only_1_correct - only_2_correct) - 1)**2 / (only_1_correct + only_2_correct)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'contingency_table': [[both_correct, only_1_correct], 
                                [only_2_correct, both_wrong]]
        }
    
    @staticmethod
    def wilcoxon_signed_rank_test(before: np.ndarray, after: np.ndarray) -> Dict:
        statistic, p_value = stats.wilcoxon(after, before, alternative='two-sided')
        
        n = len(before)
        z_score = (statistic - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'z_score': float(z_score),
            'significant': p_value < 0.05,
            'sample_size': int(n)
        }
    
    @staticmethod
    def effect_size_cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    @staticmethod
    def anova_one_way(groups: list) -> Dict:
        statistic, p_value = stats.f_oneway(*groups)
        
        df_between = len(groups) - 1
        df_within = sum(len(group) for group in groups) - len(groups)
        
        return {
            'f_statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'df_between': df_between,
            'df_within': df_within,
            'total_samples': sum(len(group) for group in groups)
        }
    
    @staticmethod
    def correlation_analysis(x: np.ndarray, y: np.ndarray) -> Dict:
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        return {
            'pearson_correlation': float(pearson_r),
            'pearson_p_value': float(pearson_p),
            'pearson_significant': pearson_p < 0.05,
            'spearman_correlation': float(spearman_r),
            'spearman_p_value': float(spearman_p),
            'spearman_significant': spearman_p < 0.05
        }
    
    @staticmethod
    def power_analysis(effect_size: float, alpha: float = 0.05, power: float = 0.8) -> Dict:
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        required_n = ((z_alpha + z_beta) / effect_size) ** 2 * 2
        
        return {
            'required_sample_size': int(np.ceil(required_n)),
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'z_alpha': float(z_alpha),
            'z_beta': float(z_beta)
        }
    
    @staticmethod
    def normality_test(data: np.ndarray) -> Dict:
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        if len(data) >= 8:
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        else:
            ks_stat, ks_p = np.nan, np.nan
        
        return {
            'shapiro_wilk_statistic': float(shapiro_stat),
            'shapiro_wilk_p_value': float(shapiro_p),
            'shapiro_normal': shapiro_p > 0.05,
            'kolmogorov_smirnov_statistic': float(ks_stat) if not np.isnan(ks_stat) else None,
            'kolmogorov_smirnov_p_value': float(ks_p) if not np.isnan(ks_p) else None,
            'ks_normal': ks_p > 0.05 if not np.isnan(ks_p) else None,
            'sample_size': len(data)
        }
    
    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray, statistic_func=np.mean, 
                                    n_bootstrap: int = 1000, confidence: float = 0.95) -> Dict:
        np.random.seed(42)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            'original_statistic': float(statistic_func(data)),
            'bootstrap_mean': float(np.mean(bootstrap_stats)),
            'bootstrap_std': float(np.std(bootstrap_stats)),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'confidence_level': confidence,
            'n_bootstrap': n_bootstrap
        }
    
    @staticmethod
    def independent_ttest(group1: np.ndarray, group2: np.ndarray, equal_var: bool = True) -> Dict:
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        effect_size = StatisticalAnalyzer.effect_size_cohen_d(group1, group2)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size_cohens_d': float(effect_size),
            'mean_group1': float(np.mean(group1)),
            'mean_group2': float(np.mean(group2)),
            'std_group1': float(np.std(group1, ddof=1)),
            'std_group2': float(np.std(group2, ddof=1)),
            'n_group1': len(group1),
            'n_group2': len(group2)
        }
    
    @staticmethod
    def mann_whitney_u_test(group1: np.ndarray, group2: np.ndarray) -> Dict:
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        n1, n2 = len(group1), len(group2)
        z_score = (statistic - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'z_score': float(z_score),
            'significant': p_value < 0.05,
            'median_group1': float(np.median(group1)),
            'median_group2': float(np.median(group2)),
            'n_group1': n1,
            'n_group2': n2
        }
    
    @staticmethod
    def chi_square_test(observed: np.ndarray, expected: np.ndarray = None) -> Dict:
        if expected is None:
            statistic, p_value, dof, expected = stats.chi2_contingency(observed)
        else:
            statistic, p_value = stats.chisquare(observed, expected)
            dof = len(observed) - 1
        
        return {
            'chi_square_statistic': float(statistic),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < 0.05,
            'expected_frequencies': expected.tolist() if hasattr(expected, 'tolist') else expected
        }
    
    @staticmethod
    def bonferroni_correction(p_values: list, alpha: float = 0.05) -> Dict:
        p_values = np.array(p_values)
        corrected_alpha = alpha / len(p_values)
        significant = p_values < corrected_alpha
        
        return {
            'original_p_values': p_values.tolist(),
            'corrected_alpha': corrected_alpha,
            'significant_tests': significant.tolist(),
            'number_significant': int(np.sum(significant)),
            'total_tests': len(p_values)
        }
    
    @staticmethod
    def false_discovery_rate(p_values: list, alpha: float = 0.05) -> Dict:
        p_values = np.array(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        m = len(p_values)
        significant = np.zeros(m, dtype=bool)
        
        for i in range(m-1, -1, -1):
            if sorted_p_values[i] <= (i+1) * alpha / m:
                significant[sorted_indices[:i+1]] = True
                break
        
        return {
            'original_p_values': p_values.tolist(),
            'fdr_alpha': alpha,
            'significant_tests': significant.tolist(),
            'number_significant': int(np.sum(significant)),
            'total_tests': m
        }