import time
import smtplib
from email.mime.text import MIMEText

LOG_FILE = "predictions.log"
ERROR_THRESHOLD = 1     # Trigger alert if >= 3 errors in the interval
CHECK_INTERVAL = 10       # Seconds between checks

# Email configuration
EMAIL_FROM = "ladenidhi00@gmail.com"
EMAIL_PASSWORD = "zkcp onku wjdy nroz"  # Use Gmail App Password
EMAIL_TO = "ladenidhi00@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def send_email_alert(error_count):
    msg = MIMEText(
        f"⚠️ Alert! High error rate detected: {error_count} errors in the last {CHECK_INTERVAL} seconds."
    )
    msg["Subject"] = "MLOps Capstone Prediction Service Alert"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
            print("Alert email sent!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def monitor_log():
    last_size = 0
    while True:
        try:
            with open(LOG_FILE, "r") as f:
                f.seek(last_size)
                lines = f.readlines()
                last_size = f.tell()

            # Count errors in new log lines
            error_count = sum(1 for line in lines if "ERROR" in line)

            if error_count >= ERROR_THRESHOLD:
                send_email_alert(error_count)

        except FileNotFoundError:
            print("Log file not found. Waiting...")
        except Exception as e:
            print(f"Monitoring error: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    print("Starting log monitoring...")
    monitor_log()