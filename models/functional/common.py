import models

__all__ = ['find']


def find(name):
    model = getattr(models, name, None)
    return model if model is not None and issubclass(model, models.BaseModel) else None
