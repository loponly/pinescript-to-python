"""Scheduler for automated optimization runs."""

import schedule
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import threading
import json
import os

from optimization.optimization_engine import OptimizationEngine
from optimization.optimization_config import OptimizationConfig, PARAMETER_GRIDS
from .database_manager import DatabaseManager
from .report_generator import ReportGenerator


class OptimizationScheduler:
    """
    Scheduler for automated optimization runs.
    
    Features:
    - Daily/weekly/monthly optimization runs
    - Configurable parameters per schedule
    - Email notifications (optional)
    - Result archiving
    - Automatic cleanup
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        config_file: str = "scheduler_config.json"
    ):
        """
        Initialize optimization scheduler.
        
        Args:
            db_manager: Database manager instance
            config_file: Configuration file path
        """
        self.db_manager = db_manager or DatabaseManager()
        self.report_generator = ReportGenerator(self.db_manager)
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Scheduler state
        self._running = False
        self._scheduler_thread = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scheduler.log'),
                logging.StreamHandler()
            ]
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load scheduler configuration."""
        default_config = {
            'schedules': {
                'daily': {
                    'enabled': False,
                    'time': '02:00',
                    'stocks': 20,
                    'parameter_grid': 'quick',
                    'timeframes': ['1h', '4h']
                },
                'weekly': {
                    'enabled': True,
                    'day': 'sunday',
                    'time': '01:00',
                    'stocks': 50,
                    'parameter_grid': 'comprehensive',
                    'timeframes': ['15m', '1h', '4h', '1d']
                },
                'monthly': {
                    'enabled': False,
                    'day': 1,
                    'time': '00:00',
                    'stocks': 100,
                    'parameter_grid': 'comprehensive',
                    'timeframes': ['5m', '15m', '1h', '4h', '1d']
                }
            },
            'notifications': {
                'email_enabled': False,
                'email_recipients': [],
                'webhook_url': None
            },
            'data_management': {
                'keep_results_days': 90,
                'auto_cleanup': True,
                'archive_old_results': True
            },
            'optimization_settings': {
                'max_workers': 4,
                'timeout_minutes': 120,
                'min_data_points': 1000
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults
                self._deep_update(default_config, loaded_config)
                return default_config
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}, using defaults")
        
        # Save default config
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup_schedules(self) -> None:
        """Setup all configured schedules."""
        schedules_config = self.config.get('schedules', {})
        
        # Daily schedule
        daily_config = schedules_config.get('daily', {})
        if daily_config.get('enabled', False):
            schedule.every().day.at(daily_config.get('time', '02:00')).do(
                self._run_scheduled_optimization,
                'daily',
                daily_config
            )
            self.logger.info(f"Daily optimization scheduled at {daily_config.get('time')}")
        
        # Weekly schedule
        weekly_config = schedules_config.get('weekly', {})
        if weekly_config.get('enabled', False):
            day = weekly_config.get('day', 'sunday').lower()
            time_str = weekly_config.get('time', '01:00')
            
            schedule_obj = getattr(schedule.every(), day)
            schedule_obj.at(time_str).do(
                self._run_scheduled_optimization,
                'weekly',
                weekly_config
            )
            self.logger.info(f"Weekly optimization scheduled on {day} at {time_str}")
        
        # Monthly schedule
        monthly_config = schedules_config.get('monthly', {})
        if monthly_config.get('enabled', False):
            # For monthly, we'll use a custom check in the scheduler loop
            self.logger.info("Monthly optimization schedule configured")
    
    def start_scheduler(self) -> None:
        """Start the scheduler in a background thread."""
        if self._running:
            self.logger.warning("Scheduler already running")
            return
        
        self._running = True
        self.setup_schedules()
        
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        self.logger.info("Optimization scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        schedule.clear()
        self.logger.info("Optimization scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Run pending schedules
                schedule.run_pending()
                
                # Check for monthly schedule manually
                self._check_monthly_schedule()
                
                # Sleep for a minute
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def _check_monthly_schedule(self) -> None:
        """Check if monthly schedule should run."""
        monthly_config = self.config.get('schedules', {}).get('monthly', {})
        
        if not monthly_config.get('enabled', False):
            return
        
        now = datetime.now()
        target_day = monthly_config.get('day', 1)
        target_time = monthly_config.get('time', '00:00')
        
        # Check if today is the target day and we haven't run this month
        if now.day == target_day:
            # Check if we've already run this month
            last_run_file = 'last_monthly_run.txt'
            last_run_month = None
            
            if os.path.exists(last_run_file):
                try:
                    with open(last_run_file, 'r') as f:
                        last_run_month = f.read().strip()
                except Exception:
                    pass
            
            current_month = now.strftime('%Y-%m')
            
            if last_run_month != current_month:
                # Parse target time
                try:
                    target_hour, target_minute = map(int, target_time.split(':'))
                    
                    # Check if current time is past target time
                    if now.hour >= target_hour and now.minute >= target_minute:
                        self._run_scheduled_optimization('monthly', monthly_config)
                        
                        # Mark as run for this month
                        with open(last_run_file, 'w') as f:
                            f.write(current_month)
                            
                except Exception as e:
                    self.logger.error(f"Error parsing monthly schedule time: {e}")
    
    def _run_scheduled_optimization(
        self,
        schedule_type: str,
        schedule_config: Dict[str, Any]
    ) -> None:
        """
        Run scheduled optimization.
        
        Args:
            schedule_type: Type of schedule ('daily', 'weekly', 'monthly')
            schedule_config: Schedule configuration
        """
        try:
            self.logger.info(f"Starting {schedule_type} optimization run")
            start_time = datetime.now()
            
            # Create optimization configuration
            opt_config = OptimizationConfig(
                max_stocks=schedule_config.get('stocks', 50),
                timeframes=schedule_config.get('timeframes', ['1h', '4h']),
                max_workers=self.config.get('optimization_settings', {}).get('max_workers', 4),
                min_data_points=self.config.get('optimization_settings', {}).get('min_data_points', 1000),
                output_dir=f"optimization_results_{schedule_type}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Get parameter grid
            grid_name = schedule_config.get('parameter_grid', 'quick')
            parameter_grid = PARAMETER_GRIDS.get(grid_name, PARAMETER_GRIDS['quick'])
            
            # Create optimization engine
            engine = OptimizationEngine(opt_config)
            
            # Create run record in database
            run_id = self.db_manager.create_optimization_run(
                run_name=f"{schedule_type}_optimization_{start_time.strftime('%Y%m%d_%H%M%S')}",
                total_stocks=opt_config.max_stocks,
                total_timeframes=len(opt_config.timeframes),
                parameter_combinations=self._count_combinations(parameter_grid),
                metadata={
                    'schedule_type': schedule_type,
                    'parameter_grid': grid_name,
                    'config': schedule_config
                }
            )
            
            # Run optimization
            results = engine.run_full_optimization(parameter_grid)
            
            # Save results to database
            for summary in results.summaries:
                self.db_manager.save_optimization_summary(summary)
            
            # Mark run as completed
            self.db_manager.complete_optimization_run(run_id, 'completed')
            
            # Generate reports
            self._generate_scheduled_reports(schedule_type, results, start_time)
            
            # Send notifications
            self._send_notifications(schedule_type, results, start_time)
            
            # Cleanup if configured
            if self.config.get('data_management', {}).get('auto_cleanup', True):
                self._cleanup_old_data()
            
            duration = datetime.now() - start_time
            self.logger.info(f"Completed {schedule_type} optimization in {duration}")
            
        except Exception as e:
            self.logger.error(f"Failed to run {schedule_type} optimization: {e}")
            
            # Mark run as failed
            try:
                self.db_manager.complete_optimization_run(run_id, 'failed')
            except:
                pass
    
    def _generate_scheduled_reports(
        self,
        schedule_type: str,
        results: Any,
        start_time: datetime
    ) -> None:
        """Generate reports for scheduled run."""
        try:
            timestamp = start_time.strftime('%Y%m%d_%H%M%S')
            report_dir = f"reports_{schedule_type}_{timestamp}"
            
            self.report_generator.generate_all_reports(report_dir)
            self.logger.info(f"Reports generated in {report_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
    
    def _send_notifications(
        self,
        schedule_type: str,
        results: Any,
        start_time: datetime
    ) -> None:
        """Send notifications about completed run."""
        notifications_config = self.config.get('notifications', {})
        
        if not notifications_config.get('email_enabled', False):
            return
        
        try:
            # Create notification message
            stats = results.get_performance_statistics()
            message = f"""
            {schedule_type.title()} Optimization Completed
            
            Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
            Duration: {datetime.now() - start_time}
            
            Results:
            - Total Strategies Tested: {stats.get('total_combinations_tested', 0):,}
            - Profitable Strategies: {stats.get('profitable_strategies', 0):,}
            - Best Profit Factor: {stats.get('best_profit_factor', 0):.2f}
            - Average Win Rate: {stats.get('avg_win_rate', 0):.1%}
            """
            
            # Send email (placeholder - would need actual email implementation)
            self._send_email_notification(
                subject=f"{schedule_type.title()} Optimization Results",
                message=message
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send notifications: {e}")
    
    def _send_email_notification(self, subject: str, message: str) -> None:
        """Send email notification (placeholder implementation)."""
        # This would integrate with an email service
        self.logger.info(f"Email notification: {subject}")
        # Implementation would depend on email service (SMTP, SendGrid, etc.)
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data based on configuration."""
        try:
            days_to_keep = self.config.get('data_management', {}).get('keep_results_days', 90)
            deleted_count = self.db_manager.cleanup_old_results(days_to_keep)
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old records")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def _count_combinations(self, parameter_grid: Dict[str, List[Any]]) -> int:
        """Count parameter combinations."""
        count = 1
        for values in parameter_grid.values():
            count *= len(values)
        return count
    
    def manual_run(
        self,
        run_type: str = 'manual',
        custom_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Manually trigger an optimization run.
        
        Args:
            run_type: Type of manual run
            custom_config: Custom configuration for the run
        """
        if custom_config is None:
            custom_config = self.config.get('schedules', {}).get('weekly', {})
        
        self._run_scheduled_optimization(run_type, custom_config)
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'running': self._running,
            'next_run_time': schedule.next_run(),
            'scheduled_jobs': len(schedule.jobs),
            'config_file': self.config_file,
            'schedules': {
                name: config.get('enabled', False)
                for name, config in self.config.get('schedules', {}).items()
            }
        }
    
    def update_schedule_config(
        self,
        schedule_name: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update schedule configuration.
        
        Args:
            schedule_name: Name of schedule to update
            updates: Configuration updates
        """
        if schedule_name not in self.config.get('schedules', {}):
            raise ValueError(f"Unknown schedule: {schedule_name}")
        
        self.config['schedules'][schedule_name].update(updates)
        self._save_config(self.config)
        
        # Restart scheduler to apply changes
        if self._running:
            self.stop_scheduler()
            self.start_scheduler()
        
        self.logger.info(f"Updated {schedule_name} schedule configuration")
