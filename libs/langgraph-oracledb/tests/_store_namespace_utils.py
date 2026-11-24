"""Shared data for the store namespace-matching tests."""

from __future__ import annotations


def namespace_matching_fixture(root: str) -> list[tuple[str, ...]]:
    return [
        (root, "user_1"),
        (root, "user_1", "prefs"),
        (root, "user_10"),
        (root, "userX1"),
        (root, "user%"),
        (root, "user%", "prefs"),
        (root, "users", "a", "bob"),
        (root, "users", "b", "bob"),
        (root, "users", "a", "x", "bob"),
        (root, "users", "a", "xbob"),
        (root, "users", "a_b", "bob"),
    ]


def namespace_matching_search_prefixes(root: str) -> list[tuple[str, ...]]:
    return [
        (root,),
        (root, "user_1"),
        (root, "user"),
        (root, "user%"),
        (root, "users"),
        (root, "users", "a"),
        (root, "users", "a", "bob"),
    ]


def namespace_matching_list_queries(root: str) -> list[dict[str, tuple[str, ...]]]:
    return [
        {"prefix": (root, "user_1")},
        {"prefix": (root, "user")},
        {"prefix": (root, "user%")},
        {"prefix": (root, "users", "a")},
        {"prefix": (root,), "suffix": ("bob",)},
        {"prefix": (root,), "suffix": ("a", "bob")},
        {"prefix": (root, "users", "*", "bob")},
        {"prefix": (root, "*", "prefs")},
        {"prefix": (root,), "suffix": ("*", "bob")},
        {"prefix": (root,), "suffix": ("a", "*", "bob")},
        {"prefix": (root, "*", "*", "*", "*", "*")},
    ]
