import sys

import tomllib
from packaging.version import parse as parse_version
import re

MIN_VERSION_LIBS = ["langchain-core"]


def get_min_version(version: str) -> str:
    # case ^x.x.x
    _match = re.match(r"^\^(\d+(?:\.\d+){0,2})$", version)
    if _match:
        return _match.group(1)

    # case >=x.x.x,<y.y.y
    _match = re.match(r"^>=(\d+(?:\.\d+){0,2}),<(\d+(?:\.\d+){0,2})$", version)
    if _match:
        _min = _match.group(1)
        _max = _match.group(2)
        assert parse_version(_min) < parse_version(_max)
        return _min

    # case x.x.x
    _match = re.match(r"^(\d+(?:\.\d+){0,2})$", version)
    if _match:
        return _match.group(1)

    raise ValueError(f"Unrecognized version format: {version}")


def get_min_version_from_toml(toml_path: str):
    # Parse the TOML file
    with open(toml_path, "rb") as file:
        toml_data = tomllib.load(file)

    # Get the dependencies from [project] section (PEP 621 format)
    if "project" not in toml_data or "dependencies" not in toml_data["project"]:
        raise ValueError("Could not find dependencies in [project] section")

    dependencies_list = toml_data["project"]["dependencies"]

    # Parse dependencies list into a dictionary
    # Format: "package-name>=x.x.x,<y.y.y" or "package-name>=x.x.x; python_version < '3.10'"
    dependencies = {}
    for dep in dependencies_list:
        # Remove environment markers (everything after semicolon)
        dep_without_marker = dep.split(";")[0].strip()

        # Extract package name and version spec
        match = re.match(r"^([a-zA-Z0-9_-]+)(.*)$", dep_without_marker)
        if match:
            pkg_name = match.group(1)
            version_spec = match.group(2)

            # If this package already exists, collect both version specs
            if pkg_name in dependencies:
                # Store as a list to handle multiple version constraints
                if isinstance(dependencies[pkg_name], list):
                    dependencies[pkg_name].append(version_spec)
                else:
                    dependencies[pkg_name] = [dependencies[pkg_name], version_spec]
            else:
                dependencies[pkg_name] = version_spec

    # Initialize a dictionary to store the minimum versions
    min_versions = {}

    # Iterate over the libs in MIN_VERSION_LIBS
    for lib in MIN_VERSION_LIBS:
        # Check if the lib is present in the dependencies
        if lib in dependencies:
            # Get the version string(s)
            version_spec = dependencies[lib]

            # Handle list format (multiple version constraints for different Python versions)
            if isinstance(version_spec, list):
                # Extract all version strings from the list and find the minimum
                versions = []
                for spec in version_spec:
                    if spec:
                        versions.append(get_min_version(spec))

                # If we found versions, use the minimum one
                if versions:
                    min_version = min(versions, key=parse_version)
                    min_versions[lib] = min_version
            elif isinstance(version_spec, str) and version_spec:
                # Handle simple string format
                min_version = get_min_version(version_spec)
                min_versions[lib] = min_version

    return min_versions


# Get the TOML file path from the command line argument
toml_file = sys.argv[1]

# Call the function to get the minimum versions
min_versions = get_min_version_from_toml(toml_file)

print(" ".join([f"{lib}=={version}" for lib, version in min_versions.items()]))
