"""Shared logging configuration for repurchase_cycle pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    controller_cfg: Optional[Dict[str, Any]] = None,
    module_name: str = "repurchase_cycle"
) -> logging.Logger:
    """
    Setup logging based on pipeline_controller config.
    
    Parameters
    ----------
    controller_cfg : dict, optional
        pipeline_controller section from config (contains logging settings).
    module_name : str
        Logger name for the calling module.
    
    Returns
    -------
    logging.Logger
    """
    if controller_cfg is None:
        controller_cfg = {}
    
    log_cfg = controller_cfg.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    save_path = log_cfg.get("save_path")
    
    level = getattr(logging, level_str, logging.INFO)
    
    logger = logging.getLogger(module_name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if save_path is specified)
    if save_path:
        log_dir = Path(save_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / "pipeline.log", encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger