import os
import yaml
import gdown
import shutil
import zipfile
from sklearn.model_selection import train_test_split
from pathlib import Path


def initialize_folder():
    """Initialize the folder for storing weights and models.

    Raises:
        OSError: if the folder cannot be created.
    """
    home = get_model_home()
    OCRHomePath = home + "/.template_dataset"

    if not os.path.exists(OCRHomePath):
        os.makedirs(OCRHomePath, exist_ok=True)


def get_model_home():
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory.
    """
    return str(os.getenv("ITC_AI_HOME", default=str(Path.home())))


def save_dir(url, dataset_name):
    url_ = url
    initialize_folder()
    home = get_model_home()
    if os.path.exists(home + "/.template_dataset/{}.zip".format(dataset_name)):
        os.remove(home + "/.template_dataset/{}.zip".format(dataset_name))
    if not os.path.isfile(home + "/.template_dataset"):
        output = home + "/.template_dataset/{}.zip".format(dataset_name)
        gdown.download(url_, output, quiet=False)
    dataset_file = home + "/.template_dataset/{}.zip".format(dataset_name)
    dataset_path = home + "/.template_dataset"
    return dataset_file, dataset_path


def process_api_result(dataset_zip: str):
    extracted_folder = None
    dataset_file, dataset_path = save_dir(dataset_zip, dataset_name='datasets_template')
    dataset_dir = os.path.splitext(dataset_file)[0]
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)
    with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    if os.path.exists(os.path.join(dataset_dir, '__MACOSX')):
        shutil.rmtree(os.path.join(dataset_dir, '__MACOSX'))

    # Find the extracted folder and rename it to 'dataset_template'
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            extracted_folder = item_path
            break

    if extracted_folder:
        renamed_folder_path = os.path.join(dataset_dir, 'datasets')
        os.rename(extracted_folder, renamed_folder_path)

        images_dir = os.path.join(renamed_folder_path, 'images')
        labels_dir = os.path.join(renamed_folder_path, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for root, _, files in os.walk(renamed_folder_path):
            for file in files:
                if not os.path.basename(file).startswith('._'):
                    file_path = os.path.join(root, file)
                    if file.lower().endswith(('.jpg', '.png')):
                        destination_path = os.path.join(images_dir, file)
                        if not os.path.exists(destination_path):
                            shutil.move(file_path, images_dir)
                    elif file.lower().endswith('.txt'):
                        destination_path = os.path.join(labels_dir, file)
                        if not os.path.exists(destination_path):
                            shutil.move(file_path, labels_dir)

        return dataset_dir, 'datasets', dataset_path
    else:
        raise ValueError("Extracted folder not found.")


def auto_path(dataset_zip: str, label_objects: list, val_split_size: float):
    print('[INFO]: Preparing dataset for training mode ...')
    dataset_dir, dataset_name, dataset_path = process_api_result(dataset_zip)
    source_dir = os.path.join(dataset_dir, dataset_name)
    training_data_path = os.path.join(dataset_path, 'auto_training.yaml')
    auto_update_yaml_file(source_dir, yaml_path=training_data_path, label_objects=label_objects)
    train_dir = os.path.join(source_dir, 'train')
    val_dir = os.path.join(source_dir, 'val')

    # Delete existing train and val directories
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_image_dir = os.path.join(train_dir, 'images')
    train_label_dir = os.path.join(train_dir, 'labels')
    val_image_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    image_folder = os.path.join(source_dir, 'images')
    image_files = [file for file in os.listdir(image_folder) if file.endswith(('.jpg', '.png'))]
    train_files, val_files = train_test_split(image_files, test_size=val_split_size, random_state=42)

    for file in train_files:
        image_src_path = os.path.join(image_folder, file)
        image_dst_path = os.path.join(train_image_dir, file)
        shutil.copy(image_src_path, image_dst_path)

        label_folder = os.path.join(source_dir, 'labels')
        label_file = os.path.splitext(file)[0] + '.txt'
        label_src_path = os.path.join(label_folder, label_file)
        label_dst_path = os.path.join(train_label_dir, label_file)
        shutil.copy(label_src_path, label_dst_path)

    for file in val_files:
        image_src_path = os.path.join(image_folder, file)
        image_dst_path = os.path.join(val_image_dir, file)
        shutil.copy(image_src_path, image_dst_path)

        label_folder = os.path.join(source_dir, 'labels')
        label_file = os.path.splitext(file)[0] + '.txt'
        label_src_path = os.path.join(label_folder, label_file)
        label_dst_path = os.path.join(val_label_dir, label_file)
        shutil.copy(label_src_path, label_dst_path)

    image_folder = os.path.join(source_dir, 'images')
    label_folder = os.path.join(source_dir, 'labels')

    # Delete existing images and labels directories
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    if os.path.exists(label_folder):
        shutil.rmtree(label_folder)
    return training_data_path


def auto_update_yaml_file(datasets_path: str, yaml_path: str, label_objects: list):
    train_dir = './train'
    val_dir = './val'
    path = datasets_path

    classes = {}

    config = {
        'names': classes,
        'path': path,
        'train': train_dir,
        'val': val_dir
    }
    # label_objects = label_objects[0].split()
    for label in label_objects:
        label = label.strip()
        if label not in classes.values():
            new_class_id = max(classes.keys()) + 1 if classes else 0
            classes[new_class_id] = label

    config['names'] = classes
    if os.path.exists(yaml_path):
        os.remove(yaml_path)
    with open(yaml_path, 'w') as yaml_file:
        yaml.safe_dump(config, yaml_file)


if __name__ == '__main__':
    pass
