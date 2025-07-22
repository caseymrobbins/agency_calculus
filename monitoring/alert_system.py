# monitoring/alert_system.py

"""
Component: Alert System
Purpose: Monitors brittleness scores and triggers immediate notifications 
         when a country's risk level crosses critical thresholds.

Inputs:
- A BrittlenessPrediction object from the brittleness_predictor module.

Outputs:
- Dispatches alerts via configured channels (e.g., email, SMS).
- Generates formatted reports for alerts.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

# This dataclass is imported from brittleness_predictor.py
# Re-defined here for standalone functionality and clarity.
from dataclasses import dataclass, field

@dataclass
class BrittlenessPrediction:
    """A simplified local version of the prediction object for this module."""
    country_code: str
    brittleness_score: float
    risk_level: str
    trajectory: str
    confidence_interval: List[float] = field(default_factory=list)
    top_risk_factors: List[Dict[str, float]] = field(default_factory=list)
    days_to_critical: int = None

# --- Alert System Configuration ---
# In production, this would be loaded from a config file or environment variables.
ALERT_CONFIG = {
    'channels': ['email', 'sms'],
    'recipients': {
        'HIGH': {
            'email': ['level1-analysts@agency-monitor.org'],
            'sms': [],
        },
        'CRITICAL': {
            'email': ['level1-analysts@agency-monitor.org', 'crisis-response-team@agency-monitor.org'],
            'sms': ['+15551234567'],
        }
    },
    'alert_threshold': 7.0,  # Trigger alerts for any score > 7.0 [cite: 2]
    'cooldown_period_hours': 6, # Prevent alert spam for the same country.
}

class AlertSystem:
    """
    Manages the logic for checking predictions and dispatching alerts.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.last_alert_times: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__)

    def _send_email(self, recipient: str, subject: str, body: str):
        """Placeholder for sending an email alert."""
        # In production, this would integrate with a service like SendGrid or AWS SES.
        # import smtplib; server = smtplib.SMTP(...) etc.
        self.logger.info(f"EMAIL ALERT SENT to {recipient}")
        self.logger.debug(f"Subject: {subject}\nBody:\n{body}")

    def _send_sms(self, recipient: str, body: str):
        """Placeholder for sending an SMS alert."""
        # In production, this would integrate with a service like Twilio.
        # from twilio.rest import Client; client.messages.create(...) etc.
        self.logger.info(f"SMS ALERT SENT to {recipient}")
        self.logger.debug(f"SMS Body: {body}")

    def _generate_report(self, prediction: BrittlenessPrediction) -> Dict[str, str]:
        """Generates a formatted, human-readable report from a prediction object."""
        subject = f"🚨 {prediction.risk_level} ALERT: {prediction.country_code} Brittleness Score is {prediction.brittleness_score:.2f}"
        
        body_lines = [
            f"**Agency Monitor Alert for {prediction.country_code}**",
            f"Generated at: {datetime.now().isoformat()}",
            "---",
            f"**Brittleness Score**: {prediction.brittleness_score:.2f} / 10.0",
            f"**Confidence Interval**: [{prediction.confidence_interval[0]:.2f}, {prediction.confidence_interval[1]:.2f}]",
            f"**Risk Level**: {prediction.risk_level}",
            f"**Trajectory**: {prediction.trajectory}",
        ]

        if prediction.days_to_critical is not None:
            body_lines.append(f"**Estimated Days to Critical (>8.0)**: {prediction.days_to_critical}")

        body_lines.append("\n**Top Contributing Risk Factors:**")
        for factor in prediction.top_risk_factors:
            line = f"- **{factor['feature']}**: {factor['value']:.3f} (Contribution: {factor['contribution']:.3f})"
            body_lines.append(line)

        sms_body = f"ALERT: {prediction.country_code} brittleness at {prediction.brittleness_score:.2f} ({prediction.risk_level}). Trajectory: {prediction.trajectory}."

        return {'subject': subject, 'body': "\n".join(body_lines), 'sms': sms_body}

    def check_and_alert(self, prediction: BrittlenessPrediction):
        """
        Main method to check a prediction and dispatch alerts if necessary.
        """
        country_code = prediction.country_code
        score = prediction.brittleness_score

        if score < self.config['alert_threshold']:
            self.logger.info(f"Score for {country_code} ({score:.2f}) is below threshold. No alert.")
            return

        # Check for cooldown period to avoid spamming
        cooldown = timedelta(hours=self.config['cooldown_period_hours'])
        last_alert_time = self.last_alert_times.get(country_code)
        if last_alert_time and (datetime.now() - last_alert_time < cooldown):
            self.logger.info(f"Alert for {country_code} is in cooldown. Suppressing.")
            return

        self.logger.warning(f"ALERT TRIGGERED for {country_code}. Score: {score:.2f}, Level: {prediction.risk_level}")
        
        # Generate the report content
        report = self._generate_report(prediction)
        
        # Get the correct recipient list for the risk level
        recipients = self.config['recipients'].get(prediction.risk_level)
        if not recipients:
            self.logger.warning(f"No recipients configured for risk level '{prediction.risk_level}'.")
            return

        # Dispatch alerts through configured channels
        if 'email' in self.config['channels']:
            for email_recipient in recipients.get('email', []):
                self._send_email(email_recipient, report['subject'], report['body'])
        
        if 'sms' in self.config['channels']:
            for sms_recipient in recipients.get('sms', []):
                self._send_sms(sms_recipient, report['sms'])

        # Update the timestamp to enforce the cooldown
        self.last_alert_times[country_code] = datetime.now()

# --- Example Usage & Testing ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. Initialize the alert system
    alert_system = AlertSystem(ALERT_CONFIG)
    print("--- Initializing Alert System ---")

    # 2. Scenario: No Alert (Score is below threshold)
    print("\n--- Scenario 1: No Alert ---")
    low_risk_pred = BrittlenessPrediction(
        country_code='SE',
        brittleness_score=4.5,
        risk_level='MEDIUM',
        trajectory='STABLE'
    )
    alert_system.check_and_alert(low_risk_pred)

    # 3. Scenario: Critical Alert
    print("\n--- Scenario 2: Critical Alert ---")
    critical_pred = BrittlenessPrediction(
        country_code='HT',
        brittleness_score=8.7,
        risk_level='CRITICAL',
        trajectory='CRITICAL_DECLINE',
        confidence_interval=[8.2, 9.2],
        top_risk_factors=[
            {'feature': 'cascade_risk', 'value': 0.8, 'contribution': 0.24},
            {'feature': 'systemic_stress', 'value': 0.72, 'contribution': 0.21}
        ],
        days_to_critical=45
    )
    alert_system.check_and_alert(critical_pred)

    # 4. Scenario: Cooldown Period
    print("\n--- Scenario 3: Cooldown ---")
    print("Attempting to send a second alert for Haiti immediately...")
    alert_system.check_and_alert(critical_pred)

    # 5. Scenario: High-level alert (different recipients)
    print("\n--- Scenario 4: High-Level Alert ---")
    high_risk_pred = BrittlenessPrediction(
        country_code='VE',
        brittleness_score=7.8,
        risk_level='HIGH',
        trajectory='DETERIORATING',
        confidence_interval=[7.1, 8.5],
        top_risk_factors=[
            {'feature': 'political_economic_crisis', 'value': 0.9, 'contribution': 0.28}
        ]
    )
    alert_system.check_and_alert(high_risk_pred)