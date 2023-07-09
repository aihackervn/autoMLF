from contextlib import contextmanager
from ultralytics import YOLO
from pathlib import Path
import sys
import shutil
import logging
import os


def initialize_folder():
    """Initialize the folder for storing weights and models.

    Raises:
        OSError: if the folder cannot be created.
    """
    home = get_model_home()
    OCRHomePath = home / ".template_model"

    if not OCRHomePath.exists():
        OCRHomePath.mkdir(parents=True, exist_ok=True)


def get_model_home():
    """Get the home directory for storing weights and models.

    Returns:
        Path: the home directory.
    """
    return Path(os.getenv("ITC_AI_HOME", default=str(Path.home())))


def save_project(delete_model=True):
    initialize_folder()
    home = get_model_home()
    project_path = home / ".template_model"

    if delete_model:
        train_dir = project_path / "train"
        if train_dir.exists():
            shutil.rmtree(str(train_dir))

    model_path = project_path / "train/weights/best.pt"
    return str(model_path), str(project_path)


@contextmanager
def suppress_output():
    # Redirect stdout and stderr to null
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def auto_train(epoch: int, image_size: int, model_name: str, data_config: str):

    with suppress_output():
        model = YOLO('yolov8{}.yaml'.format(model_name))
        model = YOLO('yolov8{}.pt'.format(model_name))
        model_path, project_path = save_project()
        model.train(data='{}'.format(data_config), epochs=epoch, imgsz=image_size, project=str(project_path), verbose=False)
    val_result = model.val(data='{}'.format(data_config), project=str(project_path))

    logging.info('Training Finished. Enjoy your model.')

    if val_result is not None:
        logging.info('Total Validation Result: {}'.format(val_result))


if __name__ == '__main__':
    pass
