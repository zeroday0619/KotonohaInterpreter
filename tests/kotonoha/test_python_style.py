"""Repository-level enforcement for the Python source conventions."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
SOURCE_ROOT: Final = PROJECT_ROOT / "src"
SCAN_DIRECTORIES: Final = ("eval", "scripts", "spikes", "src", "tests")
SNAKE_CASE: Final = re.compile(r"^_*[a-z][a-z0-9_]*$|^__[a-z][a-z0-9_]*__$")
PASCAL_CASE: Final = re.compile(r"^[A-Z][A-Za-z0-9]*$")


def _python_paths() -> tuple[Path, ...]:
    paths = [PROJECT_ROOT / "hatch_build.py"]
    for directory in SCAN_DIRECTORIES:
        paths.extend((PROJECT_ROOT / directory).rglob("*.py"))
    return tuple(sorted(paths))


def _is_dataclass(
    class_node: ast.ClassDef,
    /,
) -> bool:
    for decorator in class_node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "dataclass":
            return True
    return False


def _is_pydantic_model(
    class_node: ast.ClassDef,
    /,
) -> bool:
    return any(
        isinstance(base, ast.Name) and base.id in {"BaseModel", "BaseSettings"}
        for base in class_node.bases
    )


def _has_slots(
    class_node: ast.ClassDef,
    /,
) -> bool:
    if _is_dataclass(class_node):
        return any(
            isinstance(decorator, ast.Call)
            and any(
                keyword.arg == "slots"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in decorator.keywords
            )
            for decorator in class_node.decorator_list
        )
    return any(
        isinstance(item, ast.AnnAssign)
        and isinstance(item.target, ast.Name)
        and item.target.id == "__slots__"
        or isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__slots__"
            for target in item.targets
        )
        for item in class_node.body
    )


def _annotation_name(
    annotation: ast.expr,
    /,
) -> str:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        return annotation.value.id
    return ""


def _validate_initialization_module(
    path: Path,
    /,
    tree: ast.Module,
) -> list[str]:
    violations: list[str] = []
    for item in tree.body:
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            continue
        if (
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        ):
            continue
        if (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == "__all__"
            and isinstance(item.value, ast.Tuple)
        ):
            continue
        violations.append(f"{path}: initialization modules contain only imports and tuple __all__")
    return violations


def _validate_function(
    path: Path,
    /,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
) -> list[str]:
    location = f"{path}:{function.lineno}:{function.name}"
    violations: list[str] = []
    parameters = (
        function.args.posonlyargs
        + function.args.args
        + function.args.kwonlyargs
        + ([function.args.vararg] if function.args.vararg is not None else [])
        + ([function.args.kwarg] if function.args.kwarg is not None else [])
    )
    if function.name not in {"_", "N_"} and not SNAKE_CASE.fullmatch(function.name):
        violations.append(f"{location}: function name is not snake_case")
    if function.returns is None:
        violations.append(f"{location}: return type is missing")
    if parameters and not function.args.posonlyargs:
        violations.append(f"{location}: positional-only separator is missing")
    for index, parameter in enumerate(parameters):
        if parameter.annotation is None and not (
            index == 0 and parameter.arg in {"self", "cls"}
        ):
            violations.append(f"{location}: {parameter.arg} has no type annotation")
    declaration_line = source_lines[function.lineno - 1].strip()
    if parameters:
        if not declaration_line.endswith("("):
            violations.append(f"{location}: parameter declaration must start on separate lines")
        parameter_lines = [parameter.lineno for parameter in parameters]
        if len(parameter_lines) != len(set(parameter_lines)):
            violations.append(f"{location}: each parameter must use a separate line")
        header_end = min(item.lineno for item in function.body) - 1
        if not any(
            source_lines[line_number - 1].strip() == "/,"
            for line_number in range(function.lineno + 1, header_end + 1)
        ):
            violations.append(f"{location}: standalone positional-only separator is missing")
    elif "() ->" not in declaration_line:
        violations.append(f"{location}: zero-parameter declaration must remain on one line")
    mutable_nodes = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
    for default in (*function.args.defaults, *function.args.kw_defaults):
        if isinstance(default, mutable_nodes):
            violations.append(f"{location}: mutable parameter default is prohibited")
    return violations


def _validate_class(
    path: Path,
    /,
    class_node: ast.ClassDef,
) -> list[str]:
    location = f"{path}:{class_node.lineno}:{class_node.name}"
    violations: list[str] = []
    if not PASCAL_CASE.fullmatch(class_node.name):
        violations.append(f"{location}: class name is not PascalCase")
    if not _has_slots(class_node):
        violations.append(f"{location}: __slots__ is missing")
    model_fields = _is_dataclass(class_node) or _is_pydantic_model(class_node)
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            violations.append(f"{location}: class variable requires ClassVar or Final")
        elif (
            isinstance(item, ast.AnnAssign)
            and item.value is not None
            and isinstance(item.target, ast.Name)
            and item.target.id != "__slots__"
            and not model_fields
            and _annotation_name(item.annotation) not in {"ClassVar", "Final"}
        ):
            violations.append(
                f"{location}: {item.target.id} requires ClassVar or Final"
            )
    return violations


def test_python_source_conventions() -> None:
    violations: list[str] = []
    for path in _python_paths():
        relative_path = path.relative_to(PROJECT_ROOT)
        if (
            path.is_relative_to(SOURCE_ROOT)
            and path.name not in {"__init__.py", "__main__.py"}
            and not path.name.startswith("_")
        ):
            violations.append(f"{relative_path}: implementation module must be private")
        source = path.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(relative_path))
        if path.name == "__init__.py":
            violations.extend(_validate_initialization_module(relative_path, tree))
            continue
        if not any(
            isinstance(item, ast.ImportFrom)
            and item.module == "__future__"
            and any(alias.name == "annotations" for alias in item.names)
            for item in tree.body
        ):
            violations.append(f"{relative_path}: future annotations import is missing")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level > 1:
                    violations.append(f"{relative_path}:{node.lineno}: parent-relative import")
                if any(alias.name == "*" for alias in node.names):
                    violations.append(f"{relative_path}:{node.lineno}: wildcard import")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                violations.extend(
                    _validate_function(relative_path, node, source_lines)
                )
            elif isinstance(node, ast.ClassDef):
                violations.extend(_validate_class(relative_path, node))
    assert not violations, "\n".join(violations)
