from autoML.AutoTraining.auto_trainer import auto_train
from autoML.AutoTraining.auto_dir import auto_path
from autoML.AutoDetect.auto_detect import matching


def training_framework(dataset_url: str, epochs: int, image_size: int, model_type: str, label_list: list,
                       val_split_size: int):
    training_data_path = auto_path(dataset_zip=dataset_url, label_objects=label_list, val_split_size=val_split_size)
    auto_train(epoch=epochs, image_size=image_size, model_name=model_type, data_config=training_data_path)


def detect_framework(image, task=None, img_show=False):
    if task == 'OCR':
        detail = matching(image, task=task, img_show=img_show)
        return detail
    elif task == 'ObjectDetection':
        detail = matching(image, task=task, img_show=img_show)
        return detail


def automl_framework(dataset_url, epochs, image_size, model_type, label_list,
                     val_split_size, image, task, img_show, flags):
    if flags == 'training':
        if dataset_url and epochs and image_size and model_type and label_list and val_split_size:
            print('[INFO] Starting training framework ...')
            training_framework(dataset_url, epochs, image_size, model_type, label_list, val_split_size)
    elif flags == 'inference':
        if image and task and img_show:
            print('[INFO] Starting inference framework ...')
            detect_framework(image, task, img_show)


def start_terminal_loop():
    reset = True  # Set reset to True initially to trigger the prompt

    while True:
        if reset:
            print("Select an option:")
            print("1. Training")
            print("2. Inference")
            print("3. Reset")
            print("0. Exit")
        else:
            print("Select an option:")
            print("1. Training")
            print("2. Inference")
            print("0. Exit")

        try:
            choice = int(input("Enter your choice: "))

            if choice == 1:
                dataset_url = input("Enter the dataset URL: ")
                epochs = None
                image_size = None
                val_split_size = None

                while epochs is None:
                    try:
                        epochs = int(input("Enter the number of epochs: "))
                    except ValueError:
                        print("Invalid input. Please enter a valid integer for epochs.")

                while image_size is None:
                    try:
                        image_size = int(input("Enter the image size: "))
                    except ValueError:
                        print("Invalid input. Please enter a valid integer for image size.")

                model_type = input("Enter the model type: ")
                label_list = input("Enter the label list (comma-separated): ").split(",")

                while val_split_size is None:
                    try:
                        val_split_size = float(input("Enter the validation split size: "))
                    except ValueError:
                        print("Invalid input. Please enter a valid floating-point value for validation split size.")

                automl_framework(dataset_url=dataset_url,
                                 epochs=epochs,
                                 image_size=image_size,
                                 model_type=model_type,
                                 label_list=label_list,
                                 val_split_size=val_split_size,
                                 image=None,
                                 task=None,
                                 img_show=False,
                                 flags='training')
                reset = False

            elif choice == 2:
                image = str(input("Enter the image (comma-separated pixel values): "))
                task = input("Enter the task (OCR/ObjectDetection): ")
                img_show = input("Enter the value for img_show (True/False): ").lower() == "true"

                automl_framework(dataset_url=None,
                                 epochs=None,
                                 image_size=None,
                                 model_type=None,
                                 label_list=None,
                                 val_split_size=None,
                                 image=image,
                                 task=task,
                                 img_show=img_show,
                                 flags='inference')
                reset = False

            elif choice == 3:
                if reset:
                    reset = False
                    print("[INFO] Process reset.")
                else:
                    reset = True
                    print("[INFO] Reset option disabled.")

            elif choice == 0:
                break

            else:
                print("Invalid choice. Please try again.")

        except ValueError:
            print("Invalid input. Please enter a valid integer choice.")

        input("Press Enter to continue...")
        print()


if __name__ == '__main__':
    start_terminal_loop()
