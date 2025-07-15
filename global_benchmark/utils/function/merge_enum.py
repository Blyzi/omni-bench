from enum import Enum


def merge_enums(*enums, name="MergedEnum"):
    merged_values = {}
    for enum_class in enums:
        for member in enum_class:
            merged_values[member.name] = member.value

    return Enum(name, merged_values)
