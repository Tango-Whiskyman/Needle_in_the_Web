import time

def get_logging_path(experiment_id, directory: str = "", prefix: str = "log"):
    now = time.strftime("%Y%m%d_%H%M%S")
    if directory != "":
        directory = directory + "/"
    return f"experiments/{str(experiment_id)}/logs/{directory}{prefix}_{now}.log"

def log_message(title: str, message: str, log_path: str):
    with open(log_path, "a") as f:
        f.write("*" * 10 + title + "*" * 10 + "\n\n")
        f.write(message + "\n\n")