import os
import wandb
from ultralytics import YOLO

def run_training(run_number):

    print(f"Starting training run {run_number}...")
    try:
        #initialize wandb
        wandb.init(project="yolo_auto_train", entity="wildanaziz-braw", name=f"yolo_auto_train_{run_number}")

        # Load a model
        model = YOLO("yolov8n.pt")

        # train load model
        model.train(data="data/data.yaml", epochs=5, imgsz=640, name=f"yolo_auto_train_{run_number}")

        print(f"Training run {run_number} completed successfully.")
    
    except Exception as e:
        print(f"An error occurred during training run {run_number}: {e}")
    
    finally:
        # Finish the wandb run
        wandb.finish()

if __name__ == "__main__":
    total_runs = 3  # Specify the number of training runs
    for run in range(1, total_runs + 1):
        run_training(run)
