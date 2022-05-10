from json import loads

__all__ = ['json', 'group_by_attributes']


def json(filename: str) -> list:
    return [*map(loads, open(filename, 'r').readlines())]


def group_by_attributes(
        sinks: list[dict], attributes: list[str] = None) -> dict[str, list]:
    if attributes is None:
        attributes = ['pid', 'mass', 'x1', 'x2', 'x3', 'v1', 'v2', 'v3']
    return {attr: [sink[attr] for sink in sinks] for attr in attributes}
