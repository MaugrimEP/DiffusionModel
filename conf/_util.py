from dataclasses import field


def return_factory(param):
    return field(default_factory=lambda: param)
