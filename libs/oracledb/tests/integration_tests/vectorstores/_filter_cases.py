# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""The existing OracleVS filter contract, shared by every backend."""

FILTER_TEXTS = ["Strawberry", "Banana", "Blueberry"]

FILTER_METADATA = [
    {
        "id": "1",
        "name": "Jason",
        "age": 45,
        "address": [
            {"street": "25 A street", "city": "Mono Vista", "zip": 94088, "state": "CA"}
        ],
        "drinks": "tea",
    },
    {
        "id": "2",
        "name": "Mary",
        "age": 50,
        "address": [
            {
                "street": "15 C street",
                "city": "Mono Vista",
                "zip": 97090,
                "state": "OR",
            },
            {
                "street": "30 ABC avenue",
                "city": "Markstown",
                "zip": 90001,
                "state": "CA",
            },
        ],
    },
    {"id": "3", "name": "Mark", "age": 65, "drinks": ["soda", "tea"]},
]

FILTER_CASES = [
    ({"drinks": {"$exists": True}}, ["1", "3"]),
    ({"address.zip": 94088}, ["1"]),
    ({"name": {"$eq": "Jason"}}, ["1"]),
    ({"drinks": {"$ne": "tea"}}, ["3"]),
    ({"drinks": {"$eq": ["soda", "tea"]}}, ["3"]),
    ({"drinks": {"$ne": ["soda", "tea"]}}, ["1"]),
    (
        {
            "address[0]": {
                "$eq": {
                    "street": "25 A street",
                    "city": "Mono Vista",
                    "zip": 94088,
                    "state": "CA",
                }
            }
        },
        ["1"],
    ),
    (
        {
            "address[0]": {
                "$ne": {
                    "street": "25 A street",
                    "city": "Mono Vista",
                    "zip": 94088,
                    "state": "CA",
                }
            }
        },
        ["2"],
    ),
    ({"$or": [{"drinks": {"$exists": False}}, {"drinks": {"$ne": "tea"}}]}, ["2", "3"]),
    (
        {"$or": [{"drinks": {"$exists": False}}, {"drinks": {"$ne": ["soda", "tea"]}}]},
        ["1", "2"],
    ),
    ({"age": {"$gt": 45, "$lt": 55}}, ["2"]),
    ({"age": {"$gt": 45}}, ["2", "3"]),
    ({"age": {"$lt": 55}}, ["1", "2"]),
    ({"age": {"$gte": 65}}, ["3"]),
    ({"age": {"$lte": 50}}, ["1", "2"]),
    ({"age": {"$between": [49, 51]}}, ["2"]),
    ({"name": {"$startsWith": "Mar"}}, ["2", "3"]),
    ({"name": {"$hasSubstring": "ar"}}, ["2", "3"]),
    ({"name": {"$instr": "ar"}}, ["2", "3"]),
    ({"name": {"$regex": ".*ar.*"}}, ["2", "3"]),
    ({"name": {"$like": "%ar%"}}, ["2", "3"]),
    ({"name": {"$in": ["Mark", "Mary"]}}, ["2", "3"]),
    ({"name": {"$nin": ["Mark", "Mary"]}}, ["1"]),
    ({"drinks": {"$all": ["tea", "soda"]}}, ["3"]),
    ({"drinks": {"$all": ["tea"]}}, ["1", "3"]),
    ({"drinks": {"$not": {"$all": ["tea", "soda"]}}}, ["1", "2"]),
    ({"address[*].zip": {"$in": [94088, 1]}}, ["1"]),
    ({"address[1].zip": 90001}, ["2"]),
    ({"drinks[0,1]": "soda"}, ["3"]),
    ({"drinks[1 to 2]": "soda"}, []),
    ({"drinks": "tea"}, ["1", "3"]),
    ({"drinks[*]": "tea"}, ["1", "3"]),
    ({"name": "Jason"}, ["1"]),
    ({"address.zip": {"$not": {"$eq": "90001"}}}, ["1", "3"]),
    ({"age": {"$not": {"$gt": 46, "$lt": 65}}}, ["1", "3"]),
    ({"$and": [{"name": {"$startsWith": "Ja"}}, {"drinks": "tea"}]}, ["1"]),
    ({"name": {"$startsWith": "Ja"}, "drinks": "tea"}, ["1"]),
    ({"$or": [{"drinks": "soda"}, {"address.zip": {"$lt": 94000}}]}, ["2", "3"]),
    ({"$nor": [{"drinks": "soda"}, {"address.zip": {"$lt": 94000}}]}, ["1"]),
    (
        {
            "$and": [
                {"age": {"$gte": 60}},
                {"$or": [{"name": "Jason"}, {"drinks": {"$in": ["tea", "soda"]}}]},
            ]
        },
        ["3"],
    ),
    (
        {
            "$or": [
                {"$and": [{"name": "Jason"}, {"drinks": {"$in": ["tea", "soda"]}}]},
                {"$nor": [{"age": {"$lt": 65}}, {"name": "Jason"}]},
            ]
        },
        ["1", "3"],
    ),
    (
        {
            "age": 65,
            "$or": [
                {"$and": [{"name": "Jason"}, {"drinks": {"$in": ["tea", "soda"]}}]},
                {"$nor": [{"age": {"$lt": 65}}, {"name": "Jason"}]},
            ],
        },
        ["3"],
    ),
    (
        {
            "age": 65,
            "name": {"$regex": "*rk"},
            "$or": [
                {"$and": [{"name": "Jason"}, {"drinks": {"$in": ["tea", "soda"]}}]},
                {"$nor": [{"age": {"$lt": 65}}, {"name": "Jason"}]},
            ],
        },
        ["3"],
    ),
]
