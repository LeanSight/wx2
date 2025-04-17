"""
Utilities and decorators for the transcription system.
"""
import time
import logging
import functools
import importlib
from typing import Dict, Any, Callable
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

# Configure console
console = Console()

# Logging configuration with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
            enable_link_path=False,
            markup=True  # Enable Rich markup interpretation
        )
    ]
)
logger = logging.getLogger("transcriber")

# Nueva función para establecer nivel de logging dinámicamente
def set_log_level(level: str) -> None:
    """
    Establece el nivel de logging para la aplicación.
    
    Args:
        level: Nivel de logging ('debug', 'info', 'warning', 'error', 'critical')
    """
    # Mapeo de nombres de nivel a constantes de logging
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    # Obtener el nivel numérico
    numeric_level = levels.get(level.lower(), logging.INFO)
    
    # Configurar tanto el logger raíz como el logger específico
    logging.getLogger().setLevel(numeric_level)
    logger.setLevel(numeric_level)
    
    logger.debug(f"Log level set to: {level.upper()}")

# Utility for dynamic import
def import_module(module_name: str) -> Any:
    """Dynamically imports a module and logs it."""
    logger.debug(f"Importing: {module_name}")
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f"Error importing {module_name}: {e}")
        raise

# Decorator to import modules just before executing the function
def with_imports(*module_names: str) -> Callable:
    """
    Decorator that imports modules just before executing the function.
    
    Args:
        *module_names: Names of modules to import
    
    Returns:
        Decorated function that will have the imported modules available
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            dynamic_imports: Dict[str, Any] = {}
            for name in module_names:
                module_alias = name.split(".")[-1]
                dynamic_imports[module_alias] = import_module(name)
            kwargs["dynamic_imports"] = dynamic_imports
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Decorator to measure execution time
def log_time(func: Callable) -> Callable:
    """Decorator that measures the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        end_time: float = time.time()
        logger.info(f"{func.__name__} - Time: {end_time - start_time:.2f}s")
        return result
    return wrapper

# Helper function to display progress bar
def with_progress_bar(description: str, func: Callable) -> Any:
    """
    Executes a function while displaying a progress bar.
    
    Args:
        description: Description for the progress bar
        func: Function to execute
        
    Returns:
        Result of the function
    """
    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(f"[yellow]{description}", total=None)
        return func()

# Function to format file paths consistently
def format_path(path):
    """Applies consistent formatting to file paths in logs."""
    return f"[cyan]{path}[/]"