import os


def data_path(chr):
    return os.path.join("data", "clean", "cleaned_broken_sentences", f"{chr}.txt")


model_save_path = os.path.join("models")
log_path = os.path.join("logs")
report_path = os.path.join("reports")