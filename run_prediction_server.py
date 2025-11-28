import os
import subprocess
import time
import sys

def main():
    """
    This script starts the MLflow prediction server and keeps it running.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))

    print("==========================================")
    print("Starting ML Prediction Server")
    print("==========================================")
    print("")

    # Activate virtualenv before running this script.
    # For example: source .venv/bin/activate

    # Run the deployment pipeline to start the server
    print("Running deployment pipeline to start the prediction server...")
    print("This may take a few minutes...")

    # Stop any existing services first to ensure a clean start
    print("Stopping any existing prediction services...")
    subprocess.run(
        [sys.executable, "run_deployment.py", "--stop-service"],
        cwd=project_dir,
        check=False  # Don't fail if no service is running
    )
    print("✓ Existing services stopped.")


    # Run the deployment process
    process = subprocess.Popen(
        [sys.executable, "run_deployment.py"],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Print the output of the deployment script in real-time
    prediction_url = ""
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            if "accepts inference requests at" in line:
                # The next line should contain the URL
                try:
                    prediction_url = next(process.stdout).strip()
                except StopIteration:
                    pass
            print(line, end='')
    
    process.wait()

    if process.returncode != 0:
        print("\n\nFailed to start the prediction server.")
        return

    print("\n\n==========================================")
    print("ML Prediction Server is running.")
    if prediction_url:
        print(f"Prediction URL: {prediction_url}")
    print("You can now run 'python sample_predict.py' in another terminal to get predictions.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("==========================================")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping the prediction server...")
        subprocess.run(
            [sys.executable, "run_deployment.py", "--stop-service"],
            cwd=project_dir,
            check=True
        )
        print("✓ Prediction server stopped.")

if __name__ == "__main__":
    main()