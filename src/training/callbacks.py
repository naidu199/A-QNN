"""
Training Callbacks
===================

Callback classes for monitoring and controlling QNN training.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import pickle
from abc import ABC, abstractmethod


class Callback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """
        Called at the end of each training epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary with training metrics

        Returns:
            True to continue training, False to stop
        """
        pass

    def on_training_start(self, logs: Dict[str, Any]) -> None:
        """Called at the start of training."""
        pass

    def on_training_end(self, logs: Dict[str, Any]) -> None:
        """Called at the end of training."""
        pass


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Example:
        >>> early_stop = EarlyStopping(
        ...     monitor='val_loss',
        ...     patience=5,
        ...     min_delta=0.001
        ... )
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'auto',
        restore_best: bool = True
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min', 'max', or 'auto'
            restore_best: Whether to restore best weights at end
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        # Determine mode
        if mode == 'auto':
            if 'loss' in monitor:
                self.mode = 'min'
            else:
                self.mode = 'max'
        else:
            self.mode = mode

        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """Check if training should stop."""
        current = logs.get(self.monitor)

        if current is None:
            return True

        if self._is_improvement(current):
            self.best_value = current
            self.best_weights = logs.get('weights', None)
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return False

        return True

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == 'min':
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta

    def on_training_end(self, logs: Dict[str, Any]) -> None:
        """Restore best weights if configured."""
        if self.restore_best and self.best_weights is not None:
            logs['restored_weights'] = self.best_weights


class ModelCheckpoint(Callback):
    """
    Save model at specified intervals or when metric improves.

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='checkpoints/model_{epoch}.pkl',
        ...     monitor='val_accuracy',
        ...     save_best_only=True
        ... )
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'auto',
        save_interval: int = 1
    ):
        """
        Initialize model checkpoint callback.

        Args:
            filepath: Path pattern for saving (can include {epoch})
            monitor: Metric to monitor for best model
            save_best_only: Only save when metric improves
            mode: 'min', 'max', or 'auto'
            save_interval: Save every N epochs
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_interval = save_interval

        # Determine mode
        if mode == 'auto':
            self.mode = 'min' if 'loss' in monitor else 'max'
        else:
            self.mode = mode

        self.best_value = float('inf') if self.mode == 'min' else float('-inf')

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """Save model if conditions are met."""
        current = logs.get(self.monitor)

        should_save = False

        if not self.save_best_only:
            if epoch % self.save_interval == 0:
                should_save = True
        else:
            if current is not None and self._is_improvement(current):
                self.best_value = current
                should_save = True

        if should_save:
            filepath = self.filepath.format(epoch=epoch)
            self._save_model(filepath, logs)

        return True

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == 'min':
            return current < self.best_value
        else:
            return current > self.best_value

    def _save_model(self, filepath: str, logs: Dict[str, Any]) -> None:
        """Save model state."""
        state = {
            'weights': logs.get('weights'),
            'circuit': logs.get('circuit'),
            'metrics': {k: v for k, v in logs.items()
                       if k not in ['weights', 'circuit']}
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)


class TrainingLogger(Callback):
    """
    Log training progress to file and/or console.

    Example:
        >>> logger = TrainingLogger(
        ...     log_file='training.log',
        ...     metrics=['loss', 'accuracy', 'val_accuracy']
        ... )
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        log_interval: int = 1,
        verbose: bool = True
    ):
        """
        Initialize training logger.

        Args:
            log_file: Optional file path for logging
            metrics: Metrics to log
            log_interval: Log every N epochs
            verbose: Print to console
        """
        self.log_file = log_file
        self.metrics = metrics or ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        self.log_interval = log_interval
        self.verbose = verbose
        self.history: List[Dict] = []

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def on_training_start(self, logs: Dict[str, Any]) -> None:
        """Log training start."""
        message = "Training started"
        if self.verbose:
            print(message)
        self._write_to_file(message)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """Log epoch metrics."""
        self.history.append({'epoch': epoch, **logs})

        if epoch % self.log_interval == 0:
            metric_strs = []
            for metric in self.metrics:
                if metric in logs:
                    metric_strs.append(f"{metric}: {logs[metric]:.4f}")

            message = f"Epoch {epoch}: " + ", ".join(metric_strs)

            if self.verbose:
                print(message)
            self._write_to_file(message)

        return True

    def on_training_end(self, logs: Dict[str, Any]) -> None:
        """Log training completion and save history."""
        message = "Training completed"
        if self.verbose:
            print(message)
        self._write_to_file(message)

        # Save full history
        if self.log_file:
            history_file = self.log_file.replace('.log', '_history.json')
            with open(history_file, 'w') as f:
                json.dump(self.history, f, indent=2)

    def _write_to_file(self, message: str) -> None:
        """Write message to log file."""
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')


class LearningRateScheduler(Callback):
    """
    Adjust learning rate during training (for gradient-based methods).
    """

    def __init__(
        self,
        schedule: str = 'cosine',
        initial_lr: float = 0.1,
        min_lr: float = 0.001,
        decay_epochs: int = 50
    ):
        """
        Initialize learning rate scheduler.

        Args:
            schedule: Type of schedule ('cosine', 'exponential', 'step')
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            decay_epochs: Number of epochs for full decay
        """
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_epochs = decay_epochs

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """Update learning rate."""
        if self.schedule == 'cosine':
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * epoch / self.decay_epochs))
        elif self.schedule == 'exponential':
            decay_rate = (self.min_lr / self.initial_lr) ** (1 / self.decay_epochs)
            lr = self.initial_lr * (decay_rate ** epoch)
        elif self.schedule == 'step':
            step_size = self.decay_epochs // 3
            lr = self.initial_lr * (0.5 ** (epoch // step_size))
            lr = max(lr, self.min_lr)
        else:
            lr = self.initial_lr

        logs['learning_rate'] = lr
        return True


class BarrenPlateauMonitor(Callback):
    """
    Monitor for barren plateau conditions during training.

    Detects if the training is exhibiting barren plateau symptoms:
    - Very small gradients
    - Flat cost landscape
    - No improvement over many epochs
    """

    def __init__(
        self,
        gradient_threshold: float = 1e-5,
        variance_threshold: float = 1e-4,
        window_size: int = 10
    ):
        """
        Initialize barren plateau monitor.

        Args:
            gradient_threshold: Threshold for gradient magnitude
            variance_threshold: Threshold for cost variance
            window_size: Window for computing statistics
        """
        self.gradient_threshold = gradient_threshold
        self.variance_threshold = variance_threshold
        self.window_size = window_size
        self.cost_history: List[float] = []
        self.gradient_history: List[float] = []
        self.is_barren = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> bool:
        """Check for barren plateau conditions."""
        cost = logs.get('loss', logs.get('cost'))
        gradient_norm = logs.get('gradient_norm', None)

        if cost is not None:
            self.cost_history.append(cost)

        if gradient_norm is not None:
            self.gradient_history.append(gradient_norm)

        # Check conditions
        if len(self.cost_history) >= self.window_size:
            recent_costs = self.cost_history[-self.window_size:]
            cost_variance = np.var(recent_costs)

            if cost_variance < self.variance_threshold:
                logs['barren_plateau_warning'] = True
                logs['cost_variance'] = cost_variance
                self.is_barren = True
                print(f"Warning: Potential barren plateau detected (variance={cost_variance:.2e})")

        if len(self.gradient_history) >= self.window_size:
            recent_grads = self.gradient_history[-self.window_size:]
            mean_grad = np.mean(recent_grads)

            if mean_grad < self.gradient_threshold:
                logs['vanishing_gradient_warning'] = True
                logs['mean_gradient'] = mean_grad
                self.is_barren = True
                print(f"Warning: Vanishing gradients detected (mean={mean_grad:.2e})")

        return True
