import sys

from packaging.version import parse as parse_version
import re

try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib

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
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    for dep in dependencies_list:
        # Check if there's a Python version marker
        if ";" in dep:
            dep_without_marker, marker = dep.split(";", 1)
            dep_without_marker = dep_without_marker.strip()
            marker = marker.strip()

            # Check if this dependency applies to current Python version
            # Handle python_version < '3.10' and python_version >= '3.10' markers
            applies_to_current = True
            if "python_version" in marker:
                if "<" in marker and not ">=" in marker:
                    # python_version < 'X.Y'
                    match = re.search(r"python_version\s*<\s*['\"](\d+\.\d+)['\"]", marker)
                    if match:
                        max_version = match.group(1)
                        applies_to_current = parse_version(python_version) < parse_version(max_version)
                elif ">=" in marker:
                    # python_version >= 'X.Y'
                    match = re.search(r"python_version\s*>=\s*['\"](\d+\.\d+)['\"]", marker)
                    if match:
                        min_version_marker = match.group(1)
                        applies_to_current = parse_version(python_version) >= parse_version(min_version_marker)

            if not applies_to_current:
                continue
        else:
            dep_without_marker = dep.strip()

        # Extract package name and version spec
        match = re.match(r"^([a-zA-Z0-9_-]+)(.*)$", dep_without_marker)
        if match:
            pkg_name = match.group(1)
            version_spec = match.group(2)
            dependencies[pkg_name] = version_spec

    # Initialize a dictionary to store the minimum versions
    min_versions = {}

    # Iterate over the libs in MIN_VERSION_LIBS
    for lib in MIN_VERSION_LIBS:
        # Check if the lib is present in the dependencies
        if lib in dependencies:
            version_spec = dependencies[lib]
            if version_spec:
                min_version = get_min_version(version_spec)
                min_versions[lib] = min_version

    return min_versions


# Get the TOML file path from the command line argument
toml_file = sys.argv[1]

# Call the function to get the minimum versions
min_versions = get_min_version_from_toml(toml_file)

print(" ".join([f"{lib}=={version}" for lib, version in min_versions.items()]))
