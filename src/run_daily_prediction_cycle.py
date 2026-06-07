from run_daily_monitor import main as run_daily_monitor
from prediction_tracking import run_prediction_tracking_cycle


def main():
    run_daily_monitor()
    outputs = run_prediction_tracking_cycle()
    print("Prediction tracking complete.")
    print(outputs)


if __name__ == "__main__":
    main()
