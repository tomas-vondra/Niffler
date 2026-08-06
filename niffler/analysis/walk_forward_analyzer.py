import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Type
from dateutil.relativedelta import relativedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from niffler.strategies.base_strategy import BaseStrategy
from niffler.backtesting.backtest_engine import BacktestEngine
from niffler.backtesting.backtest_result import BacktestResult
from niffler.backtesting.cost_model import CostModel
from niffler.optimization.parameter_space import ParameterSpace
from niffler.optimization.optimizer_factory import create_optimizer, get_available_optimizers
from .analysis_result import AnalysisResult, log_failure_rate


# Analysis modes.
#
# MODE_WALK_FORWARD is the real thing: parameters are fitted on an in-sample training
# window and then evaluated on the immediately following, untouched out-of-sample window.
#
# MODE_SEGMENTED_IN_SAMPLE re-runs one fixed parameter set over consecutive slices of the
# same data the parameters were originally fitted on. That is *not* walk-forward analysis
# and it validates nothing, so it is named honestly and is never the default.
MODE_WALK_FORWARD = 'walk_forward'
MODE_SEGMENTED_IN_SAMPLE = 'segmented_in_sample'

VALID_MODES = (MODE_WALK_FORWARD, MODE_SEGMENTED_IN_SAMPLE)

# Minimum number of bars a window must contain to be usable.
MIN_TRAIN_BARS = 30
MIN_TEST_BARS = 30


@dataclass
class WalkForwardWindow:
    """A single train/test split of the walk-forward schedule."""

    window_number: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_start: Optional[pd.Timestamp] = None
    train_end: Optional[pd.Timestamp] = None

    @property
    def has_training_window(self) -> bool:
        """True when this window carries an in-sample training period."""
        return self.train_start is not None and self.train_end is not None


@dataclass
class WalkForwardFold:
    """
    Result of one completed walk-forward fold.

    Carries the training period, the out-of-sample test period, the parameters that were
    chosen on the training data, and the in-sample vs out-of-sample performance used to
    derive the walk-forward efficiency ratio.
    """

    fold_number: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    parameters: Dict[str, Any]
    test_result: BacktestResult
    n_test_bars: int
    train_start: Optional[pd.Timestamp] = None
    train_end: Optional[pd.Timestamp] = None
    n_train_bars: int = 0
    train_return_pct: Optional[float] = None
    test_return_pct: float = 0.0
    efficiency_ratio: Optional[float] = None
    in_sample: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable summary of the fold (without the full backtest result)."""
        return {
            'fold_number': self.fold_number,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
            'parameters': self.parameters,
            'n_train_bars': self.n_train_bars,
            'n_test_bars': self.n_test_bars,
            'train_return_pct': self.train_return_pct,
            'test_return_pct': self.test_return_pct,
            'efficiency_ratio': self.efficiency_ratio,
            'in_sample': self.in_sample,
        }


class WalkForwardAnalyzer:
    """
    Walk-forward analysis implementation for strategy validation.

    In ``walk_forward`` mode (the default) each fold optimises the strategy parameters on
    an in-sample training window and then evaluates *those* parameters on the immediately
    following out-of-sample test window, which the optimiser never saw. The ratio between
    out-of-sample and in-sample performance (the walk-forward efficiency ratio) is the
    actual output of the exercise: it measures how much of the fitted edge survives on
    unseen data.

    In ``segmented_in_sample`` mode a single fixed parameter set is re-run over consecutive
    slices of the data. This is a segmented in-sample backtest, not a validation - every
    slice is data the parameters were already fitted on - and it must be requested
    explicitly.
    """

    def __init__(self,
                 strategy_class: Type[BaseStrategy],
                 parameter_space: Optional[ParameterSpace] = None,
                 optimal_parameters: Optional[Dict[str, Any]] = None,
                 train_window_months: int = 12,
                 test_window_months: int = 6,
                 step_months: int = 3,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 mode: str = MODE_WALK_FORWARD,
                 anchored: bool = False,
                 optimization_method: str = 'grid',
                 optimization_metric: str = 'total_return',
                 n_jobs: Optional[int] = None,
                 max_results_in_memory: int = 1000,
                 cost_model: Optional[CostModel] = None):
        """
        Initialize Walk-Forward Analyzer.

        Args:
            strategy_class: Strategy class to analyze
            parameter_space: Search space re-optimised on every training window.
                Required for ``walk_forward`` mode.
            optimal_parameters: Fixed parameter set. Only used (and required) by
                ``segmented_in_sample`` mode.
            train_window_months: Months of in-sample data used to fit each fold
            test_window_months: Months of out-of-sample data used to evaluate each fold
            step_months: Months to step forward between folds
            initial_capital: Starting capital for backtests
            commission: Commission rate for trades
            mode: ``walk_forward`` (default) or ``segmented_in_sample``
            anchored: If True the training window always starts at the first bar of the
                dataset (anchored walk-forward). If False the training window rolls.
            optimization_method: Optimizer used on each training window ('grid', 'random')
            optimization_metric: Metric the optimizer selects parameters by
            n_jobs: Number of parallel jobs (None = auto-detect)
            max_results_in_memory: Maximum number of results to keep in memory
            cost_model: Transaction cost model applied to every fill, in both the
                per-fold optimisation and the out-of-sample evaluation. Passing it
                to only one of the two would measure a strategy fitted in one
                market against results from another.

        Raises:
            ValueError: If the configuration is inconsistent (e.g. ``walk_forward`` mode
                without a ``parameter_space``).
        """
        self.strategy_class = strategy_class
        self.parameter_space = parameter_space
        self.optimal_parameters = optimal_parameters
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.initial_capital = initial_capital
        self.commission = commission
        self.mode = mode
        self.anchored = anchored
        self.optimization_method = optimization_method
        self.optimization_metric = optimization_metric
        self.n_jobs = n_jobs or min(mp.cpu_count(), 4)  # Default to max 4 processes
        self.max_results_in_memory = max_results_in_memory
        self.cost_model = cost_model

        self._validate_parameters()

        if self.mode == MODE_SEGMENTED_IN_SAMPLE:
            self._validate_strategy_parameters()
            # Reusable instances - the parameters never change in this mode.
            self._strategy = self.strategy_class(**self.optimal_parameters)
        else:
            self._strategy = None

        self._backtest_engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
            cost_model=self.cost_model
        )

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got '{self.mode}'")

        if self.mode == MODE_WALK_FORWARD:
            if self.parameter_space is None:
                raise ValueError(
                    "walk_forward mode requires a parameter_space: parameters must be "
                    "re-optimised on each training window for the test window to be "
                    "genuinely out-of-sample. Pass parameter_space=..., or explicitly "
                    f"request mode='{MODE_SEGMENTED_IN_SAMPLE}' with optimal_parameters "
                    "to re-run one fixed parameter set over consecutive in-sample slices."
                )
            if self.train_window_months <= 0:
                raise ValueError("train_window_months must be positive")
            if self.optimization_method not in get_available_optimizers():
                available = ', '.join(get_available_optimizers())
                raise ValueError(f"optimization_method must be one of: {available}")
        else:
            if not self.optimal_parameters:
                raise ValueError(
                    f"mode='{MODE_SEGMENTED_IN_SAMPLE}' requires optimal_parameters"
                )

        if self.test_window_months <= 0:
            raise ValueError("test_window_months must be positive")
        if self.step_months <= 0:
            raise ValueError("step_months must be positive")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.commission < 0:
            raise ValueError("commission cannot be negative")

    def _validate_strategy_parameters(self) -> None:
        """Validate that the fixed parameters are compatible with the strategy class."""
        try:
            self.strategy_class(**self.optimal_parameters)
            logging.info("Strategy parameter validation successful")
        except Exception as e:
            raise ValueError(f"Invalid parameters for {self.strategy_class.__name__}: {e}")

    def analyze(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> AnalysisResult:
        """
        Perform walk-forward analysis on the given data.

        Args:
            data: Historical price data with OHLCV columns
            symbol: Symbol identifier

        Returns:
            AnalysisResult containing every completed fold, the aggregate out-of-sample
            metrics, the efficiency-ratio statistics and the failure accounting.

        Raises:
            ValueError: If the data is unusable or no fold completed successfully
        """
        if data.empty or len(data) < 100:
            raise ValueError("Insufficient data for walk-forward analysis")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        total_window_months = self.test_window_months
        if self.mode == MODE_WALK_FORWARD:
            total_window_months += self.train_window_months

        min_required_periods = max(2, (total_window_months * 30) // 7)  # Rough estimate
        if len(data) < min_required_periods:
            raise ValueError(
                f"Need at least {min_required_periods} data points for "
                f"{total_window_months}-month train+test windows"
            )

        logging.info(f"Starting {self.mode} analysis for {symbol}")
        if self.mode == MODE_WALK_FORWARD:
            logging.info(f"Parameter space: {self.parameter_space.parameters}")
            logging.info(f"Train window: {self.train_window_months} months "
                         f"({'anchored' if self.anchored else 'rolling'})")
        else:
            logging.warning(
                "Running in segmented_in_sample mode: a single fixed parameter set is "
                "re-run over consecutive slices of the same data it was fitted on. "
                "These results are NOT out-of-sample."
            )
            logging.info(f"Using parameters: {self.optimal_parameters}")
        logging.info(f"Test window: {self.test_window_months} months")
        logging.info(f"Step size: {self.step_months} months")

        if self.step_months < self.test_window_months:
            overlap_pct = (
                (self.test_window_months - self.step_months) / self.test_window_months * 100.0
            )
            logging.warning(
                f"step_months ({self.step_months}) is smaller than test_window_months "
                f"({self.test_window_months}): consecutive out-of-sample windows overlap by "
                f"{overlap_pct:.0f}%. Folds are NOT independent observations; the pooled "
                f"metrics deduplicate repeated bars (see 'oos_overlap_pct'), but the "
                f"per-fold counters still treat every fold as one sample. Set "
                f"step_months >= test_window_months for non-overlapping folds."
            )

        windows = self._generate_walk_forward_periods(data)
        logging.info(f"Generated {len(windows)} walk-forward windows")

        if not windows:
            raise ValueError("No valid walk-forward periods found")

        if self.n_jobs == 1:
            folds, failed = self._run_folds_sequential(data, symbol, windows)
        else:
            folds, failed = self._run_folds_parallel(data, symbol, windows)

        attempted = len(windows)
        failure_rate = log_failure_rate(
            f"Walk-forward analysis for {symbol}", attempted, failed
        )

        if not folds:
            raise ValueError(
                f"No successful walk-forward folds ({failed}/{attempted} failed)"
            )

        results = [fold.test_result for fold in folds]

        combined_metrics = self._calculate_combined_metrics(results)
        combined_metrics.update(self._calculate_efficiency_metrics(folds))
        combined_metrics.update({
            'attempted_folds': float(attempted),
            'failed_folds': float(failed),
            'failure_rate_pct': failure_rate * 100.0,
        })

        stability_metrics = self._calculate_stability_metrics(results)

        return AnalysisResult(
            analysis_type='walk_forward',
            strategy_name=self.strategy_class.__name__,
            symbol=symbol,
            analysis_start_date=data.index[0],
            analysis_end_date=data.index[-1],
            individual_results=results,
            combined_metrics=combined_metrics,
            analysis_parameters={
                'mode': self.mode,
                'anchored': self.anchored,
                'parameter_space': (self.parameter_space.parameters
                                    if self.parameter_space is not None else None),
                'optimal_parameters': self.optimal_parameters,
                'optimization_method': self.optimization_method,
                'optimization_metric': self.optimization_metric,
                'cost_model': (self.cost_model.description
                               if self.cost_model is not None else None),
                'train_window_months': (self.train_window_months
                                        if self.mode == MODE_WALK_FORWARD else None),
                'test_window_months': self.test_window_months,
                'step_months': self.step_months,
                'n_periods': len(folds),
                'attempted_folds': attempted,
                'failed_folds': failed,
                'failure_rate': failure_rate,
            },
            stability_metrics=stability_metrics,
            metadata={
                'test_periods': [(f.test_start, f.test_end) for f in folds],
                'train_periods': [(f.train_start, f.train_end) for f in folds],
                'folds': [f.to_dict() for f in folds],
            },
            attempted_runs=attempted,
            failed_runs=failed,
        )

    def _run_folds_sequential(self, data: pd.DataFrame, symbol: str,
                              windows: List[WalkForwardWindow]) -> Tuple[List[WalkForwardFold], int]:
        """
        Run walk-forward folds sequentially.

        Args:
            data: Full historical dataset
            symbol: Symbol identifier
            windows: Train/test windows to evaluate

        Returns:
            Tuple of (completed folds, number of failed windows)
        """
        folds: List[WalkForwardFold] = []
        failed = 0

        for i, window in enumerate(windows):
            if i % 10 == 0:
                logging.info(
                    f"Fold {i + 1}/{len(windows)}: test "
                    f"{window.test_start.date()} to {window.test_end.date()}"
                )
                if len(folds) > self.max_results_in_memory:
                    folds = self._manage_memory_efficient_results(folds)

            try:
                fold = self._run_single_fold(data, window, symbol)
            except Exception as e:
                logging.warning(f"Error in fold {window.window_number}: {e}")
                failed += 1
                continue

            if fold is None:
                failed += 1
                continue

            folds.append(fold)

        return folds, failed

    def _run_folds_parallel(self, data: pd.DataFrame, symbol: str,
                            windows: List[WalkForwardWindow]) -> Tuple[List[WalkForwardFold], int]:
        """
        Run walk-forward folds in parallel, one process per fold.

        The per-fold optimizer always runs single-threaded here so that fold-level and
        optimizer-level process pools are never nested.

        Args:
            data: Full historical dataset
            symbol: Symbol identifier
            windows: Train/test windows to evaluate

        Returns:
            Tuple of (completed folds, number of failed windows)
        """
        folds: List[WalkForwardFold] = []
        failed = 0

        logging.info(f"Running {len(windows)} folds using {self.n_jobs} parallel jobs")

        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_window = {}
                for window in windows:
                    try:
                        future = executor.submit(
                            self._run_single_fold_static,
                            data, window, symbol,
                            self.strategy_class, self.parameter_space,
                            self.optimal_parameters, self.mode,
                            self.optimization_method, self.optimization_metric,
                            self.initial_capital, self.commission, self.cost_model
                        )
                        future_to_window[future] = window
                    except Exception as e:
                        logging.warning(f"Failed to submit fold {window.window_number}: {e}")
                        failed += 1

                completed = 0
                for future in as_completed(future_to_window):
                    window = future_to_window[future]
                    completed += 1

                    if completed % 10 == 0:
                        logging.info(f"Completed {completed}/{len(windows)} folds")

                    try:
                        fold = future.result(timeout=600)
                        if fold is not None:
                            folds.append(fold)
                        else:
                            failed += 1
                    except Exception as e:
                        logging.warning(
                            f"Fold {window.window_number} "
                            f"({window.test_start.date()}-{window.test_end.date()}) failed: {e}"
                        )
                        failed += 1

        except Exception as e:
            logging.error(f"Critical error in parallel walk-forward: {e}")
            raise

        folds.sort(key=lambda f: f.fold_number)
        return folds, failed

    def _manage_memory_efficient_results(self, folds: List[WalkForwardFold]) -> List[WalkForwardFold]:
        """
        Trim the fold list to bound memory usage.

        Args:
            folds: Folds accumulated so far

        Returns:
            The chronologically most recent half of the folds
        """
        if len(folds) <= self.max_results_in_memory:
            return folds

        keep_count = self.max_results_in_memory // 2
        logging.info(f"Memory management: trimming {len(folds)} folds to {keep_count}")

        # Keep the most recent folds. Keeping the *best* folds would bias every
        # aggregate metric computed afterwards.
        folds.sort(key=lambda f: f.fold_number)
        return folds[-keep_count:]

    def _generate_walk_forward_periods(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Generate the train/test schedule for the analysis.

        Each window places the out-of-sample test period immediately after its training
        period, so the optimizer never sees the data it is graded on.

        Args:
            data: Full historical dataset

        Returns:
            List of WalkForwardWindow objects (empty if the data cannot cover one window)
        """
        windows: List[WalkForwardWindow] = []
        data_start = data.index[0]
        data_end = data.index[-1]

        current_date = data_start
        window_number = 0

        while True:
            if self.mode == MODE_WALK_FORWARD:
                train_start = data_start if self.anchored else current_date
                train_end = current_date + relativedelta(months=self.train_window_months)
                test_start = train_end
            else:
                train_start = None
                train_end = None
                test_start = current_date

            test_end = test_start + relativedelta(months=self.test_window_months)

            if test_end > data_end:
                break

            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            has_enough_test = len(test_data) >= MIN_TEST_BARS

            if self.mode == MODE_WALK_FORWARD:
                train_data = data[(data.index >= train_start) & (data.index < train_end)]
                has_enough_train = len(train_data) >= MIN_TRAIN_BARS
            else:
                has_enough_train = True

            if has_enough_test and has_enough_train:
                window_number += 1
                windows.append(WalkForwardWindow(
                    window_number=window_number,
                    test_start=test_start,
                    test_end=test_end,
                    train_start=train_start,
                    train_end=train_end,
                ))

            current_date += relativedelta(months=self.step_months)

        return windows

    def _run_single_fold(self, data: pd.DataFrame, window: WalkForwardWindow,
                         symbol: str) -> Optional[WalkForwardFold]:
        """
        Run one fold: fit on the training window, evaluate on the test window.

        Args:
            data: Full historical dataset
            window: Train/test window to evaluate
            symbol: Symbol identifier

        Returns:
            The completed fold, or None if the window did not hold enough data
        """
        return self._execute_fold(
            data=data,
            window=window,
            symbol=symbol,
            strategy_class=self.strategy_class,
            parameter_space=self.parameter_space,
            optimal_parameters=self.optimal_parameters,
            mode=self.mode,
            optimization_method=self.optimization_method,
            optimization_metric=self.optimization_metric,
            initial_capital=self.initial_capital,
            commission=self.commission,
            cost_model=self.cost_model,
            backtest_engine=self._backtest_engine,
        )

    @staticmethod
    def _run_single_fold_static(data: pd.DataFrame, window: WalkForwardWindow, symbol: str,
                                strategy_class: Type[BaseStrategy],
                                parameter_space: Optional[ParameterSpace],
                                optimal_parameters: Optional[Dict[str, Any]],
                                mode: str, optimization_method: str, optimization_metric: str,
                                initial_capital: float, commission: float,
                                cost_model: Optional[CostModel] = None
                                ) -> Optional[WalkForwardFold]:
        """Static entry point for parallel fold execution (must be picklable)."""
        try:
            return WalkForwardAnalyzer._execute_fold(
                data=data,
                window=window,
                symbol=symbol,
                strategy_class=strategy_class,
                parameter_space=parameter_space,
                optimal_parameters=optimal_parameters,
                mode=mode,
                optimization_method=optimization_method,
                optimization_metric=optimization_metric,
                initial_capital=initial_capital,
                commission=commission,
                cost_model=cost_model,
                backtest_engine=None,
            )
        except Exception as e:
            logging.warning(f"Fold {window.window_number} failed: {e}")
            return None

    @staticmethod
    def _execute_fold(data: pd.DataFrame,
                      window: WalkForwardWindow,
                      symbol: str,
                      strategy_class: Type[BaseStrategy],
                      parameter_space: Optional[ParameterSpace],
                      optimal_parameters: Optional[Dict[str, Any]],
                      mode: str,
                      optimization_method: str,
                      optimization_metric: str,
                      initial_capital: float,
                      commission: float,
                      cost_model: Optional[CostModel] = None,
                      backtest_engine: Optional[BacktestEngine] = None) -> Optional[WalkForwardFold]:
        """
        Fit parameters on the training window and evaluate them out-of-sample.

        Args:
            data: Full historical dataset
            window: Train/test window to evaluate
            symbol: Symbol identifier
            strategy_class: Strategy class under analysis
            parameter_space: Search space (walk_forward mode only)
            optimal_parameters: Fixed parameters (segmented_in_sample mode only)
            mode: Analysis mode
            optimization_method: Optimizer used on the training window
            optimization_metric: Metric the optimizer sorts by
            initial_capital: Starting capital for backtests
            commission: Commission rate for trades
            cost_model: Transaction cost model applied to every fill
            backtest_engine: Engine to reuse; a fresh one is created when None

        Returns:
            The completed fold, or None if either window held too little data
        """
        test_data = data[(data.index >= window.test_start) & (data.index < window.test_end)].copy()
        if len(test_data) < MIN_TEST_BARS:
            logging.warning(
                f"Fold {window.window_number}: insufficient test data ({len(test_data)} rows)"
            )
            return None

        engine = backtest_engine or BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            cost_model=cost_model
        )

        train_return_pct: Optional[float] = None
        n_train_bars = 0

        if mode == MODE_WALK_FORWARD:
            train_data = data[(data.index >= window.train_start) &
                              (data.index < window.train_end)].copy()
            if len(train_data) < MIN_TRAIN_BARS:
                logging.warning(
                    f"Fold {window.window_number}: insufficient training data "
                    f"({len(train_data)} rows)"
                )
                return None

            n_train_bars = len(train_data)
            parameters, train_return_pct = WalkForwardAnalyzer._optimize_on_training_window(
                train_data=train_data,
                strategy_class=strategy_class,
                parameter_space=parameter_space,
                optimization_method=optimization_method,
                optimization_metric=optimization_metric,
                initial_capital=initial_capital,
                commission=commission,
                cost_model=cost_model,
            )
            if parameters is None:
                logging.warning(
                    f"Fold {window.window_number}: optimization produced no valid parameters"
                )
                return None
        else:
            parameters = dict(optimal_parameters)

        strategy = strategy_class(**parameters)
        test_result = engine.run_backtest(strategy, test_data, symbol)

        efficiency_ratio = WalkForwardAnalyzer._calculate_efficiency_ratio(
            train_return_pct, n_train_bars,
            test_result.total_return_pct, len(test_data)
        )

        test_result.metadata = {
            'period_number': window.window_number,
            'fold_number': window.window_number,
            'parameters_used': parameters,
            'train_start': window.train_start,
            'train_end': window.train_end,
            'test_start': window.test_start,
            'test_end': window.test_end,
            'train_data_points': n_train_bars,
            'test_data_points': len(test_data),
            'train_return_pct': train_return_pct,
            'efficiency_ratio': efficiency_ratio,
            'in_sample': mode != MODE_WALK_FORWARD,
        }

        return WalkForwardFold(
            fold_number=window.window_number,
            test_start=window.test_start,
            test_end=window.test_end,
            parameters=parameters,
            test_result=test_result,
            n_test_bars=len(test_data),
            train_start=window.train_start,
            train_end=window.train_end,
            n_train_bars=n_train_bars,
            train_return_pct=train_return_pct,
            test_return_pct=test_result.total_return_pct,
            efficiency_ratio=efficiency_ratio,
            in_sample=mode != MODE_WALK_FORWARD,
        )

    @staticmethod
    def _optimize_on_training_window(train_data: pd.DataFrame,
                                     strategy_class: Type[BaseStrategy],
                                     parameter_space: ParameterSpace,
                                     optimization_method: str,
                                     optimization_metric: str,
                                     initial_capital: float,
                                     commission: float,
                                     cost_model: Optional[CostModel] = None
                                     ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """
        Fit parameters on a single in-sample training window.

        Args:
            train_data: In-sample slice to optimise on
            strategy_class: Strategy class under analysis
            parameter_space: Search space
            optimization_method: Optimizer name ('grid', 'random')
            optimization_metric: Metric the optimizer sorts by
            initial_capital: Starting capital for backtests
            commission: Commission rate for trades
            cost_model: Transaction cost model applied to every candidate fill, so
                the parameters are fitted in the same market they are graded in

        Returns:
            Tuple of (best parameters, in-sample total return percentage). Both are None
            when the optimizer found no valid combination.
        """
        optimizer = create_optimizer(
            method=optimization_method,
            strategy_class=strategy_class,
            parameter_space=parameter_space,
            data=train_data,
            initial_capital=initial_capital,
            commission=commission,
            sort_by=optimization_metric,
            n_jobs=1,  # never nest process pools inside a fold
            cost_model=cost_model,
        )

        results = optimizer.optimize()
        if not results:
            return None, None

        best = results[0]
        return dict(best.parameters), best.backtest_result.total_return_pct

    @staticmethod
    def _calculate_efficiency_ratio(train_return_pct: Optional[float], n_train_bars: int,
                                    test_return_pct: float, n_test_bars: int) -> Optional[float]:
        """
        Calculate the walk-forward efficiency ratio for one fold.

        The ratio compares out-of-sample to in-sample performance on a per-bar basis so
        that differently sized train and test windows remain comparable. A ratio near or
        above 1.0 means the fitted edge survived out-of-sample; a ratio near 0 (or
        negative) means the parameters were curve-fitted.

        Args:
            train_return_pct: In-sample total return percentage (None in in-sample mode)
            n_train_bars: Number of bars in the training window
            test_return_pct: Out-of-sample total return percentage
            n_test_bars: Number of bars in the test window

        Returns:
            The efficiency ratio, or None when it is undefined (no training window, or
            non-positive in-sample performance, for which the ratio is meaningless).
        """
        if train_return_pct is None or n_train_bars <= 0 or n_test_bars <= 0:
            return None

        train_rate = train_return_pct / n_train_bars
        if train_rate <= 0:
            # A negative or flat in-sample result makes the ratio uninterpretable:
            # dividing by it would turn losses into "efficient" positive numbers.
            return None

        test_rate = test_return_pct / n_test_bars
        return test_rate / train_rate

    def _calculate_efficiency_metrics(self, folds: List[WalkForwardFold]) -> Dict[str, float]:
        """
        Aggregate the per-fold efficiency ratios.

        Args:
            folds: Completed folds

        Returns:
            Dictionary of efficiency statistics (empty-ish when no ratio was defined)
        """
        ratios = [f.efficiency_ratio for f in folds if f.efficiency_ratio is not None]

        metrics: Dict[str, float] = {
            'folds_with_efficiency_ratio': float(len(ratios)),
        }

        if not ratios:
            return metrics

        metrics.update({
            'mean_efficiency_ratio': float(np.mean(ratios)),
            'median_efficiency_ratio': float(np.median(ratios)),
            'std_efficiency_ratio': float(np.std(ratios)),
            'worst_efficiency_ratio': float(np.min(ratios)),
            'best_efficiency_ratio': float(np.max(ratios)),
            'folds_above_half_efficiency_pct': (
                sum(1 for r in ratios if r >= 0.5) / len(ratios) * 100.0
            ),
        })

        train_returns = [f.train_return_pct for f in folds if f.train_return_pct is not None]
        if train_returns:
            metrics['avg_train_return_pct'] = float(np.mean(train_returns))

        return metrics

    def _resolve_combined_periods_per_year(self, results: List[Any]) -> float:
        """
        Annualisation factor for the pooled out-of-sample return series.

        Taken from the same inference the per-fold backtests use, so the pooled Sharpe
        and the per-fold Sharpe ratios share one convention. Hardcoding 252 here would
        understate an hourly-crypto pooled Sharpe by sqrt(8760/252) against the per-fold
        numbers reported beside it.

        Args:
            results: Out-of-sample backtest results

        Returns:
            Bars per year inferred from the first fold that carries a usable index
        """
        for result in results:
            portfolio_values = getattr(result, 'portfolio_values', None)
            if portfolio_values is not None and len(portfolio_values) > 1:
                return self._backtest_engine.resolve_periods_per_year(portfolio_values.index)
        return BacktestEngine._DEFAULT_PERIODS_PER_YEAR

    def _pool_out_of_sample_returns(self, results: List[Any]) -> Tuple[pd.Series, int, int]:
        """
        Pool the per-bar out-of-sample returns of every fold, without double counting.

        Consecutive folds overlap whenever ``step_months < test_window_months`` (which
        the defaults do: a 6-month test window stepped 3 months at a time repeats half
        of every window). Concatenating them blindly counts those bars twice, which
        inflates the apparent sample size and smooths the dispersion of exactly the
        statistics walk-forward analysis exists to stress. Each timestamp is therefore
        kept only once, from the first fold that covers it.

        Args:
            results: Out-of-sample backtest results, in fold order

        Returns:
            Tuple of (deduplicated return series, number of bars kept, number of
            overlapping bars dropped)
        """
        pooled: List[pd.Series] = []
        seen: set = set()
        duplicates = 0

        for result in results:
            portfolio_values = getattr(result, 'portfolio_values', None)
            if portfolio_values is None or len(portfolio_values) == 0:
                continue

            period_returns = portfolio_values.pct_change().dropna()
            if period_returns.empty:
                continue

            # Without timestamps overlap cannot be detected, so such folds are pooled
            # whole rather than silently dropped.
            if isinstance(period_returns.index, pd.DatetimeIndex):
                mask = [ts not in seen for ts in period_returns.index]
                duplicates += len(period_returns) - sum(mask)
                seen.update(period_returns.index)
                period_returns = period_returns[mask]

            if not period_returns.empty:
                pooled.append(period_returns)

        if not pooled:
            return pd.Series(dtype=float), 0, duplicates

        combined = pd.concat(pooled)
        return combined, len(combined), duplicates

    def _calculate_combined_metrics(self, results: List[Any]) -> Dict[str, float]:
        """
        Calculate combined metrics across all out-of-sample test windows.

        The pooled bar returns are deduplicated first (see
        :meth:`_pool_out_of_sample_returns`) and annualised with the factor inferred
        from the data's own bar frequency, so ``combined_sharpe_ratio`` is directly
        comparable with the per-fold ``sharpe_ratio`` values sitting next to it.

        Args:
            results: Out-of-sample backtest results

        Returns:
            Dictionary of aggregate metrics
        """
        if not results:
            return {}

        returns = [r.total_return for r in results]
        return_pcts = [r.total_return_pct for r in results]
        sharpe_ratios = [r.sharpe_ratio or 0.0 for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]
        total_trades = [r.total_trades for r in results]

        # Calculate combined portfolio performance over the independent OOS bars
        combined_returns, independent_bars, overlapping_bars = (
            self._pool_out_of_sample_returns(results)
        )
        total_pooled_bars = independent_bars + overlapping_bars

        combined_sharpe = 0.0
        if len(combined_returns) > 1 and combined_returns.std() > 0:
            periods_per_year = self._resolve_combined_periods_per_year(results)
            combined_sharpe = (
                np.sqrt(periods_per_year) * combined_returns.mean() / combined_returns.std()
            )

        metrics = {
            'total_periods': len(results),
            'independent_oos_bars': float(independent_bars),
            'overlapping_oos_bars': float(overlapping_bars),
            'oos_overlap_pct': (
                overlapping_bars / total_pooled_bars * 100.0 if total_pooled_bars else 0.0
            ),
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'avg_return_pct': np.mean(return_pcts),
            'median_return_pct': np.median(return_pcts),
            'std_return_pct': np.std(return_pcts),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'combined_sharpe_ratio': combined_sharpe,
            'avg_max_drawdown': np.mean(max_drawdowns),
            # max_drawdown is reported as a negative percentage, so the *worst*
            # drawdown is the minimum, not the maximum.
            'worst_max_drawdown': min(max_drawdowns) if max_drawdowns else 0.0,
            'best_max_drawdown': max(max_drawdowns) if max_drawdowns else 0.0,
            'avg_win_rate': np.mean(win_rates),
            'avg_trades_per_period': np.mean(total_trades),
            'positive_return_periods': sum(1 for r in returns if r > 0),
            'positive_return_pct': (sum(1 for r in returns if r > 0) / len(returns)) * 100,
            'profitable_periods': sum(1 for r in return_pcts if r > 0),
            'profitable_periods_pct': (sum(1 for r in return_pcts if r > 0) / len(return_pcts)) * 100
        }

        return metrics

    def _calculate_stability_metrics(self, results: List[Any]) -> Dict[str, float]:
        """
        Calculate metrics measuring performance stability across periods.

        Args:
            results: Out-of-sample backtest results

        Returns:
            Dictionary of stability metrics
        """
        if not results:
            return {}

        # Performance stability metrics
        returns = [r.total_return for r in results]
        return_pcts = [r.total_return_pct for r in results]
        sharpe_ratios = [r.sharpe_ratio or 0.0 for r in results]

        stability_metrics = {
            'return_volatility': np.std(returns),
            'return_pct_volatility': np.std(return_pcts),
            'sharpe_volatility': np.std(sharpe_ratios),
            'return_consistency': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0,
            'return_pct_consistency': np.mean(return_pcts) / np.std(return_pcts) if np.std(return_pcts) > 0 else 0.0,
            'temporal_stability': self._calculate_temporal_stability(results)
        }

        # Calculate rolling correlations if we have enough periods
        if len(results) >= 4:
            stability_metrics.update(self._calculate_rolling_stability(results))

        return stability_metrics

    def _calculate_temporal_stability(self, results: List[Any]) -> float:
        """Calculate temporal stability - how consistent performance is over time."""
        if len(results) < 3:
            return 1.0

        returns = [r.total_return for r in results]

        # Calculate period-to-period changes
        changes = [returns[i+1] - returns[i] for i in range(len(returns) - 1)]

        # Count direction changes (positive to negative or vice versa)
        direction_changes = 0
        for i in range(len(changes) - 1):
            if (changes[i] > 0) != (changes[i+1] > 0):
                direction_changes += 1

        # Calculate temporal stability (lower direction changes = higher stability)
        max_possible_changes = len(changes) - 1
        if max_possible_changes > 0:
            stability = 1.0 - (direction_changes / max_possible_changes)
        else:
            stability = 1.0

        return stability

    def _calculate_rolling_stability(self, results: List[Any]) -> Dict[str, float]:
        """Calculate rolling performance stability metrics."""
        returns = [r.total_return for r in results]

        # Calculate rolling average stability
        window_size = min(4, len(returns) // 2)
        rolling_means = []

        for i in range(len(returns) - window_size + 1):
            window_returns = returns[i:i + window_size]
            rolling_means.append(np.mean(window_returns))

        rolling_stability = {
            'rolling_mean_stability': 1.0 / (1.0 + np.std(rolling_means)) if len(rolling_means) > 1 else 1.0,
            'trend_consistency': self._calculate_trend_consistency(returns)
        }

        return rolling_stability

    def _calculate_trend_consistency(self, returns: List[float]) -> float:
        """Calculate how consistent the performance trend is across periods."""
        if len(returns) < 3:
            return 1.0

        # Calculate period-to-period changes
        changes = [returns[i+1] - returns[i] for i in range(len(returns) - 1)]

        # Count direction changes (positive to negative or vice versa)
        direction_changes = 0
        for i in range(len(changes) - 1):
            if (changes[i] > 0) != (changes[i+1] > 0):
                direction_changes += 1

        # Calculate trend consistency (lower direction changes = higher consistency)
        max_possible_changes = len(changes) - 1
        if max_possible_changes > 0:
            consistency = 1.0 - (direction_changes / max_possible_changes)
        else:
            consistency = 1.0

        return consistency
