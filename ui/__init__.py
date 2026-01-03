try:
    from .dashboard import run_dashboard
except ImportError:
    from dashboard import run_dashboard

__all__ = ["run_dashboard"]
