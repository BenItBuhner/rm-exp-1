import importlib
import inspect


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split("@")
    module = importlib.import_module(prefix + module_path)
    return getattr(module, class_name)


def get_source_path(identifier: str, prefix: str = "models."):
    module_path, _ = identifier.split("@")
    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)
