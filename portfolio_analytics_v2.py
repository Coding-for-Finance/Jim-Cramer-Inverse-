
import pandas as pd
import numpy as np
import statsmodels.api as sm


class PortfolioAnalysis:
    def __init__(self, returns, ann_factor, benchmark=None, rf=None):
        """
        Initialize the PortfolioAnalysis class with returns (DataFrame or Series),
        annualization factor, and optional benchmark and risk-free rate.
        """
        self.returns = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
        self.ann_factor = ann_factor
        self.rf = rf if rf is not None else pd.Series(0, index=returns.index)
        self.benchmark = benchmark if benchmark is not None else pd.Series(0, index=returns.index)

        self.prepare_data()

    def prepare_data(self):
        """
        Prepare data for performance metric calculations.
        Adjust for risk-free rate and combine strategies with the benchmark.
        """
        # Adjust for risk-free rate
        self.adjusted_returns = self.returns.subtract(self.rf, axis=0)
        self.adjusted_benchmark = self.benchmark.subtract(self.rf, axis=0)

        # Combine adjusted returns and benchmark
        self.combined_data = self.adjusted_returns.copy()
        self.combined_data['benchmark'] = self.adjusted_benchmark

        self.n_periods = len(self.returns)
        self.navs = (1 + self.combined_data).cumprod()

    def calculate_performance_metrics(self):
        """
        Calculate various performance metrics for each strategy using vectorized operations.
        """
        # NAVs and drawdowns
        cumulative_max = self.navs.cummax()
        drawdowns = self.navs / cumulative_max - 1
        self.max_drawdowns = drawdowns.min()
        self.avg_drawdowns = drawdowns.mean()

        # Other metrics
        self.g_avgs = (1 + self.combined_data).prod() ** (self.ann_factor / self.n_periods) - 1
        self.volatilities = self.combined_data.std() * np.sqrt(self.ann_factor)
        self.sharpe_ratios = self.g_avgs / self.volatilities
        self.skewnesses = self.combined_data.skew()
        self.kurtoses = self.combined_data.kurtosis()

    def calculate_sortino_ratio(self):
        """
        Calculate Sortino ratio.
        """
        downside_returns = self.returns_all[self.returns_all < 0]
        downside_risk = np.sqrt(np.mean(np.power(downside_returns, 2)))
        return np.mean(self.returns_all) / downside_risk

    def calculate_alpha_beta(self):
        """
        Calculate alpha and beta for each strategy.
        """
        self.alphas, self.betas = [], []
        if self.benchmark is not None:
            for strat in self.adjusted_returns.columns:
                model = sm.OLS(self.adjusted_returns[strat].values, sm.add_constant(self.adjusted_benchmark.values)).fit()
                self.alphas.append(model.params[0])
                self.betas.append(model.params[1])

    def summary_with_benchmark(self):
        """
        Generate summary with benchmark.
        """
        self.calculate_performance_metrics()

        # Calculate benchmark statistics
        benchmark_g_avg = (1 + self.adjusted_benchmark).prod() ** (self.ann_factor / self.n_periods) - 1

        # Prepare summary DataFrame
        summary = pd.DataFrame(columns=[
            'Geometric Average - Excess (%)', 'Volatility Annual (%)',
            'Sharpe Ratio', 'Information Ratio', 'Skewness',
            'Excess Kurtosis', 'Beta', 'Alpha (%)'
        ])

        # Fill in the summary statistics for each strategy
        for strategy in self.adjusted_returns.columns:
            strategy_stats = self.calculate_strategy_stats(self.adjusted_returns[strategy], benchmark_g_avg)
            summary.loc[strategy] = strategy_stats

        # Add benchmark statistics as a row
        benchmark_stats = self.calculate_strategy_stats(self.benchmark, benchmark_g_avg)
        summary.loc['Benchmark'] = benchmark_stats

        return summary.fillna(0)

    def calculate_strategy_stats(self, strategy_returns, benchmark_g_avg=None):
        """
        Calculate various statistics for a given strategy or benchmark.
        """
        g_avg = (1 + strategy_returns).prod() ** (self.ann_factor / self.n_periods) - 1
        volatility = strategy_returns.std() * np.sqrt(self.ann_factor)
        sharpe_ratio = g_avg / volatility
        information_ratio = None
        skewness = strategy_returns.skew()
        kurtosis = strategy_returns.kurtosis()
        beta = alpha = None

        if benchmark_g_avg is not None:
            information_ratio = (g_avg - benchmark_g_avg) / volatility

        if self.benchmark is not None:
            model = sm.OLS(strategy_returns.values, sm.add_constant(self.benchmark.values)).fit()
            alpha, beta = model.params

        return [
            round(g_avg * 100, 2), round(volatility * 100, 2),
            round(sharpe_ratio, 2), round(information_ratio, 2) if information_ratio is not None else None,
            round(skewness, 2), round(kurtosis, 2),
            round(beta, 2) if beta is not None else None,
            round(alpha * 100, 2) if alpha is not None else None
        ]

    def summary_without_benchmark(self):
        """
        Generate summary without benchmark.
        """
        self.calculate_performance_metrics()
        summary = pd.DataFrame(columns=[
            'Geometric Average - Excess (%)', 'Volatility Annual (%)',
            'Sharpe Ratio', 'Sortino Ratio', 'Max. Drawdown (%)',
            'Avg. Drawdown (%)', 'Skewness', 'Excess Kurtosis'
        ])

        # Fill in the summary statistics for each strategy
        for strategy in self.adjusted_returns.columns:
            strategy_stats = self.calculate_strategy_stats(self.adjusted_returns[strategy])
            summary.loc[strategy] = strategy_stats

        return summary.fillna(0)